from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import yfinance as yf
import pickle
import os
import warnings
import logging
from pathlib import Path
from src.data.preprocessing import VolatilityProcessor

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

app = FastAPI(title="TimeMixer Volatility API")

# Mount the static folder so we can serve HTML/CSS/JS seamlessly
app.mount("/static", StaticFiles(directory="static"), name="static")


class PredictRequest(BaseModel):
    ticker: str
    horizon: int
    asset_class: str


# --- Build Ticker Hierarchy on Startup ---
TICKERS_DICT = {}
FILE_MAPPING = {
    "stock": "stock.csv",
    "etf": "index_etf.csv",
    "forex": "forex.csv",
    "crypto": "crypto.csv"
}

try:
    for ac, filename in FILE_MAPPING.items():
        path = Path("Data") / filename
        if path.exists():
            df = pd.read_csv(path, usecols=['Ticker'])
            TICKERS_DICT[ac] = sorted(list(df['Ticker'].unique()))
    print(f"✅ Loaded ticker dictionary: {sum(len(v) for v in TICKERS_DICT.values())} tickers total.")
except Exception as e:
    print(f"⚠️ Error loading tickers: {e}")


@app.get("/")
def serve_index():
    # Return the beautiful dashboard UI
    return FileResponse("static/index.html")


@app.get("/api/tickers")
def get_tickers():
    return {"assets": TICKERS_DICT}


@app.post("/api/predict")
def predict_volatility(req: PredictRequest):
    try:
        # 1. Download ~6mo of live data
        v_ticker = req.ticker
        if req.asset_class == "forex" and not v_ticker.endswith("=X"):
            v_ticker += "=X"
        elif req.asset_class == "crypto":
            if v_ticker.endswith("USD") and not v_ticker.endswith("-USD"):
                v_ticker = v_ticker[:-3] + "-USD"
            elif not v_ticker.endswith("-USD"):
                v_ticker += "-USD"

        yf_ticker = yf.Ticker(v_ticker)
        df_live = yf_ticker.history(period="6mo").reset_index()

        if len(df_live) < 100:
            raise HTTPException(status_code=400, detail="Insufficient live market data.")

        # Clean yfinance columns
        df_live = df_live[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df_live['Date'] = pd.to_datetime(df_live['Date']).dt.tz_localize(None)
        df_live['Ticker'] = req.ticker

        # 2. Compute true Yang-Zhang target volatility
        proc = VolatilityProcessor()
        vol_live = proc.compute_volatility(df_live)
        nf_input = proc.to_neuralforecast_format(vol_live)

        # Baseline: Average of the rolling 96 window
        recent_96 = float(vol_live['y'].tail(96).mean())

        # 3. Load the mathematically proven model
        model_path = Path(f"models/timemixer/{req.asset_class}/{req.ticker}/h{req.horizon}.pkl")
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model weights not found for this horizon.")
        
        import torch
        _original_torch_load = torch.load
        try:
            torch.load = lambda *a, **k: _original_torch_load(*a, **{**k, "map_location": "cpu"})
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        finally:
            torch.load = _original_torch_load

        # Overwrite baked-in GPU trainer config to enforce pure CPU inference on Render
        if hasattr(model, 'models'):
            for m in model.models:
                if hasattr(m, 'trainer_kwargs'):
                    m.trainer_kwargs['accelerator'] = 'cpu'
                    if 'devices' in m.trainer_kwargs:
                        del m.trainer_kwargs['devices']

        # 4. Instant inference
        forecast = model.predict(df=nf_input)

        # Format output payload
        dates = forecast['ds'].dt.strftime('%Y-%m-%d').tolist()
        values = [max(0.0, float(v)) for v in forecast['TimeMixer'].tolist()]

        # Extract history for context line
        hist_dates = vol_live['ds'].tail(96).dt.strftime('%Y-%m-%d').tolist()
        hist_values = vol_live['y'].tail(96).tolist()

        return {
            "baseline": recent_96,
            "forecast": {
                "dates": dates,
                "values": values
            },
            "history": {
                "dates": hist_dates,
                "values": hist_values
            }
        }

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

