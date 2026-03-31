import yfinance as yf
import pandas as pd
from pathlib import Path

# Provide standard save paths
DATA_DIR = Path("Data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# YFinance Tickers mapped to our Clean Tickers 
TICKERS = {
    "stock": {
        "AAPL": "AAPL", "AMZN": "AMZN", "BRK-B": "BRK-B", "GOOGL": "GOOGL",
        "JNJ": "JNJ", "META": "META", "MSFT": "MSFT", "NVDA": "NVDA", 
        "TSLA": "TSLA", "V": "V"
    },
    "index_etf": {
        "EEM": "EEM", "EFA": "EFA", "GLD": "GLD", "GOVT": "GOVT",
        "IWM": "IWM", "QQQ": "QQQ", "SCHD": "SCHD", "SPY": "SPY",
        "VTI": "VTI", "VWO": "VWO"
    },
    "forex": {
        "AUDJPY=X": "AUDJPY", "AUDUSD=X": "AUDUSD", "EURGBP=X": "EURGBP",
        "EURJPY=X": "EURJPY", "EURUSD=X": "EURUSD", "GBPJPY=X": "GBPJPY",
        "GBPUSD=X": "GBPUSD", "USDCAD=X": "USDCAD", "USDCHF=X": "USDCHF",
        "USDJPY=X": "USDJPY"
    },
    "crypto": {
        "ADA-USD": "ADAUSD", "BCH-USD": "BCHUSD", "BNB-USD": "BNBUSD",
        "BTC-USD": "BTCUSD", "DOGE-USD": "DOGEUSD", "DOT-USD": "DOTUSD",
        "ETH-USD": "ETHUSD", "LTC-USD": "LTCUSD", "SOL-USD": "SOLUSD",
        "XRP-USD": "XRPUSD"
    }
}

START_DATE = "2010-01-01"
END_DATE = "2026-03-31"  # Inclusive up to Mar 30

for group, ticker_dict in TICKERS.items():
    print(f"\n--- Downloading {group} data ---")
    yf_tickers = list(ticker_dict.keys())
    
    # Download all tickers for group in one API call
    data = yf.download(yf_tickers, start=START_DATE, end=END_DATE, group_by='ticker', auto_adjust=False, progress=False)
    
    all_rows = []
    
    # yfinance returns different structures, handling carefully
    for yf_ticker, clean_ticker in ticker_dict.items():
        if len(yf_tickers) == 1:
            df = data.copy()
        else:
            try:
                # Grouped by ticker, the columns are MultiIndex, grab specific ticker
                df = data[yf_ticker].copy()
            except KeyError:
                print(f"Skipping {yf_ticker} (not returned by YF)")
                continue

        df = df.dropna(subset=['Close'])
        if df.empty:
            print(f"  Warning: No date range returned for {yf_ticker}")
            continue
            
        df = df.reset_index()
        
        # Ensure column names match our expected schema (Date, Open, High, Low, Close, Volume)
        df["Ticker"] = clean_ticker
        
        # Sometimes Volume goes missing for fiat forex pairs, ensure it exists
        if "Volume" not in df.columns:
            df["Volume"] = 0
            
        try:
            df = df[["Ticker", "Date", "Open", "High", "Low", "Close", "Volume"]]
            all_rows.append(df)
        except KeyError as e:
            print(f"Skipping {yf_ticker} due to missing column: {e}")
        
    if all_rows:
        combined_df = pd.concat(all_rows, ignore_index=True)
        # Formatting 'Date' nicely
        combined_df['Date'] = combined_df['Date'].dt.strftime('%Y-%m-%d')
        
        combined_df = combined_df.sort_values(by=["Ticker", "Date"])
        
        out_path = DATA_DIR / f"{group}.csv"
        # Overwrite existing files
        combined_df.to_csv(out_path, index=False)
        
        print(f"Saved {group} to '{out_path}'")
        print(f"  Rows: {len(combined_df)}, Tickers: {combined_df['Ticker'].nunique()}")
        print(f"  Range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
    else:
        print(f"Failed to compile group: {group}")

print("\nAll historical data refreshed over 2010 - 2026!")
