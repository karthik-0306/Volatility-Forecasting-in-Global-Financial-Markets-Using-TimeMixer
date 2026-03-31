"""
src/data/preprocessing.py
──────────────────────────
Volatility estimation and data preparation for NeuralForecast.

Two estimators:
  - Yang-Zhang (PRIMARY target `y`) : theoretically superior, uses full OHLC
  - Rolling Std                     : paper's baseline, close-price only

This is the core improvement over the original paper:
  Paper → Rolling Std → simple, ignores overnight gaps & intraday swings
  Ours  → Yang-Zhang  → 7-8x more efficient, uses all OHLCV information

Usage:
    from src.data.preprocessing import VolatilityProcessor
    proc = VolatilityProcessor()
    vol_df = proc.compute_volatility(raw_df)          # Yang-Zhang by default
    train, test = proc.train_test_split(vol_df)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)


class VolatilityProcessor:
    """
    Computes volatility series and prepares data for NeuralForecast.

    Primary target  : Yang-Zhang annualized volatility (column 'y')
    Secondary column: Rolling Std (column 'rolling_vol') — kept for comparison
    """

    def __init__(self):
        self.window      = cfg.data.volatility.rolling_window        # 21 days
        self.ann_factor  = cfg.data.volatility.annualization_factor  # 252
        self.ticker_col  = cfg.data.ticker_column                    # 'Ticker'
        self.date_col    = cfg.data.date_column                      # 'Date'
        self.split_date  = cfg.data.split_date                       # '2024-01-01'

    # ═══════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════

    def compute_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Yang-Zhang (primary) and Rolling Std (comparison) for all tickers.

        Args:
            df : Raw OHLCV DataFrame from DataLoader.

        Returns:
            DataFrame with columns:
            [unique_id, ds, y, rolling_vol]
            where 'y' = Yang-Zhang volatility (the forecast target)
        """
        log.info(f"Computing volatility | window={self.window}d | primary=Yang-Zhang")

        results = []
        for ticker, group in df.groupby(self.ticker_col):
            group = group.sort_values(self.date_col).copy()

            row = pd.DataFrame({
                self.date_col: group[self.date_col].values,
                "yz_vol":      self._yang_zhang(group).values,
                "rolling_vol": self._rolling_std(group).values,
                "unique_id":   ticker,
            })
            results.append(row)

        if not results:
            log.warning("No volatility results computed (empty input or insufficient data).")
            return pd.DataFrame(columns=["ds", "unique_id", "y", "yz_vol", "rolling_vol"])

        out = pd.concat(results, ignore_index=True)
        out = out.rename(columns={self.date_col: "ds"})

        # 'y' = primary target = Yang-Zhang
        out["y"] = out["yz_vol"]
        out = out.dropna(subset=["y"])

        log.info(
            f"  ✓ {len(out):,} rows | "
            f"{out['unique_id'].nunique()} tickers | "
            f"cols: {list(out.columns)}"
        )
        return out

    def train_test_split(
        self,
        vol_df: pd.DataFrame,
        split_date: str = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Temporal train/test split (no random shuffling — time series rule).

        Args:
            vol_df     : Output of compute_volatility().
            split_date : Cutoff date. Train < date, Test >= date.

        Returns:
            (train_df, test_df)
        """
        if split_date is None:
            split_date = self.split_date

        train = vol_df[vol_df["ds"] < split_date].copy()
        test  = vol_df[vol_df["ds"] >= split_date].copy()

        log.info(
            f"Split at {split_date} | "
            f"Train: {len(train):,} rows | Test: {len(test):,} rows"
        )
        return train, test

    def to_neuralforecast_format(self, vol_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return only [unique_id, ds, y] — the exact format NeuralForecast expects.
        """
        return vol_df[["unique_id", "ds", "y"]].copy()

    # ═══════════════════════════════════════════════════════════
    # Volatility Estimators
    # ═══════════════════════════════════════════════════════════

    def _yang_zhang(self, df: pd.DataFrame) -> pd.Series:
        """
        Yang-Zhang (2000) — minimum variance unbiased volatility estimator.

        Uses full OHLC data:
          σ²_YZ = σ²_overnight + k·σ²_open-to-close + (1-k)·σ²_intraday

        where:
          overnight  = log(Open_t / Close_{t-1})  ← gap between sessions
          open-close = log(Close_t / Open_t)       ← daytime trend
          intraday   = Rogers-Satchell component   ← H/L range movement
          k          = 0.34 / (1.34 + (N+1)/(N-1))

        ~7-8x more statistically efficient than close-to-close estimator.
        """
        N  = self.window
        k  = 0.34 / (1.34 + (N + 1) / (N - 1))

        o  = np.log(df["Open"]  / df["Close"].shift(1))  # overnight return
        c  = np.log(df["Close"] / df["Open"])             # open-to-close return
        h  = np.log(df["High"]  / df["Open"])
        lo = np.log(df["Low"]   / df["Open"])

        rs = h * (h - c) + lo * (lo - c)                 # Rogers-Satchell intraday

        σ2_overnight  = o.rolling(N).var(ddof=0)
        σ2_open_close = c.rolling(N).var(ddof=0)
        σ2_intraday   = rs.rolling(N).mean()

        σ2_yz = σ2_overnight + k * σ2_open_close + (1 - k) * σ2_intraday
        return np.sqrt(σ2_yz * self.ann_factor)

    def _rolling_std(self, df: pd.DataFrame) -> pd.Series:
        """
        Paper's estimator (Alex Li, 2024) — kept for direct comparison.

        σ = √252 × std(log(Close_t / Close_{t-1}), window=21)

        Only uses Close prices — ignores overnight gaps and intraday swings.
        """
        log_ret = np.log(df["Close"] / df["Close"].shift(1))
        return log_ret.rolling(self.window).std() * np.sqrt(self.ann_factor)


class FeatureEngineer:
    """
    Computes exogenous features for TimeMixer models (Phase 2).
    Takes in raw OHLCV DataFrame, merges it with VolDataFrame,
    and computes lag / technical / regime features.
    """
    
    def __init__(self):
        pass
        
    def generate_features(self, raw_df: pd.DataFrame, vol_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge volatility data with OHLCV data to compute technicals, and lags.
        """
        import ta
        log.info("Generating exogenous features (Technical + Lags + Regimes)")
        
        # Replace string dates with datetime if not already
        raw_df['Date'] = pd.to_datetime(raw_df['Date'])
        vol_df['ds'] = pd.to_datetime(vol_df['ds'])
        
        merged = pd.merge(
            raw_df, 
            vol_df, 
            left_on=['Ticker', 'Date'], 
            right_on=['unique_id', 'ds'], 
            how='inner'
        )
        
        results = []
        for ticker, group in merged.groupby('unique_id'):
            group = group.sort_values('ds').copy()
            
            # --- 1. LAGGED VOLATILITY ---
            group['vol_lag_1']  = group['y'].shift(1)
            group['vol_lag_5']  = group['y'].shift(5)
            group['vol_lag_21'] = group['y'].shift(21)
            group['vol_lag_63'] = group['y'].shift(63)
            
            # --- 2. TECHNICAL INDICATORS ---
            # ATR (Average True Range) mapped to volatility
            try:
                group['atr'] = ta.volatility.average_true_range(group['High'], group['Low'], group['Close'], window=14)
            except Exception:
                group['atr'] = 0.0
                
            # Bollinger Bands Width
            try:
                group['bb_width'] = ta.volatility.bollinger_wband(group['Close'], window=20, window_dev=2)
            except Exception:
                group['bb_width'] = 0.0
                
            # RSI 
            try:
                group['rsi'] = ta.momentum.rsi(group['Close'], window=14)
            except Exception:
                group['rsi'] = 50.0
            
            # MACD
            try:
                group['macd'] = ta.trend.macd(group['Close'])
            except Exception:
                group['macd'] = 0.0
                
            # Volume Z-score
            group['volume_zscore'] = (group['Volume'] - group['Volume'].rolling(21).mean()) / (group['Volume'].rolling(21).std() + 1e-8)
            
            # --- 3. REGIME INDICATORS ---
            # Distance from 200-day moving average (Trend regime)
            ma_200 = group['Close'].rolling(200).mean()
            group['dist_to_ma200'] = (group['Close'] - ma_200) / (ma_200 + 1e-8)
            
            results.append(group)
            
        final_df = pd.concat(results, ignore_index=True)
        final_df = final_df.dropna().reset_index(drop=True)
        
        # Keep only NeuralForecast required plus the newly engineered exogenous features
        cols_to_keep = ['unique_id', 'ds', 'y', 'vol_lag_1', 'vol_lag_5', 'vol_lag_21', 'vol_lag_63',
                        'atr', 'bb_width', 'rsi', 'macd', 'volume_zscore', 'dist_to_ma200']
                        
        out = final_df[cols_to_keep]
        log.info(f"  ✓ Engineered {len(cols_to_keep)-3} exogenous features for {len(out):,} rows")
        return out


if __name__ == "__main__":
    from src.data.loader import DataLoader
    loader = DataLoader()
    stock_df = loader.load("stock")

    proc = VolatilityProcessor()
    vol_df = proc.compute_volatility(stock_df)
    
    eng = FeatureEngineer()
    feat_df = eng.generate_features(stock_df, vol_df)
    print(feat_df.head())
