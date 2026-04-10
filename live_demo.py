import warnings
import logging
import os
import pickle
import yfinance as yf
import pandas as pd

# Silence underlying library warnings
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from src.data.preprocessing import VolatilityProcessor

def main():
    print("1. Downloading live AAPL data from Yahoo Finance...")
    ticker = yf.Ticker("AAPL")
    df_live = ticker.history(period="6mo").reset_index()

    df_live = df_live[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df_live['Date'] = pd.to_datetime(df_live['Date']).dt.tz_localize(None)
    df_live['Ticker'] = "AAPL"

    print("2. Calculating Yang-Zhang Volatility on live data...")
    proc = VolatilityProcessor()
    vol_live = proc.compute_volatility(df_live)
    nf_input = proc.to_neuralforecast_format(vol_live)

    # Calculate the recent historical baseline (e.g., from the last 96 days)
    # We use this to compare if the future is "riskier" or "safer" than what we're used to
    historical_avg = vol_live['y'].tail(96).mean()

    print("3. Loading the pre-trained TimeMixer model...")
    model_path = "models/timemixer/stock/AAPL/h12.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("4. Predicting the future volatility...")
    forecast = model.predict(df=nf_input)

    print("\n=================================================================")
    print(" 🔮 AAPL VOLATILITY FORECAST FOR THE NEXT 12 TRADING DAYS")
    print("=================================================================")
    print(f" Baseline: Recent 96-Day Average Volatility is {historical_avg:.4f}\n")
    
    print(" Date           Forecast      Risk Level")
    print("-----------------------------------------------------------------")
    
    # Print the forecast with Terminal Colors!
    for _, row in forecast.iterrows():
        date_str = str(row['ds'].date())
        pred_vol = row['TimeMixer']
        
        # Color Logic based on comparing to the historical average
        if pred_vol > historical_avg * 1.05:
            # More than 5% above average
            color = "\033[91m"  # Red
            status = "🔴 HIGH RISK (Above Avg)"
        elif pred_vol < historical_avg * 0.95:
            # More than 5% below average
            color = "\033[92m"  # Green
            status = "🟢 LOW RISK (Below Avg)"
        else:
            # Within 5% of average
            color = "\033[93m"  # Yellow
            status = "🟡 NORMAL (Near Avg)"
        
        reset = "\033[0m"
        print(f" {date_str}     {color}{pred_vol:.4f}{reset}        {status}")
        
    print("=================================================================\n")

if __name__ == "__main__":
    main()
