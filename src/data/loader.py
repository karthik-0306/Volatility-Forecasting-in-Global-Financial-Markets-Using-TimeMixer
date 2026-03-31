"""
src/data/loader.py
──────────────────
Loads raw OHLCV CSV data for all 4 asset classes.
Handles parsing, type casting, sorting, and basic validation.

Usage:
    from src.data.loader import DataLoader
    loader = DataLoader()
    stock_df = loader.load("stock")
    all_dfs  = loader.load_all()
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Asset class → file path mapping ───────────────────────────
ASSET_MAP = {
    "stock":  cfg.paths.data.stock,
    "etf":    cfg.paths.data.etf,
    "forex":  cfg.paths.data.forex,
    "crypto": cfg.paths.data.crypto,
}

# Project root (2 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class DataLoader:
    """
    Centralized data loader for all asset classes.

    Attributes:
        date_col   (str): Name of date column in raw CSV.
        ticker_col (str): Name of ticker/symbol column.
    """

    def __init__(self):
        self.date_col   = cfg.data.date_column
        self.ticker_col = cfg.data.ticker_column
        self.ohlcv_cols = cfg.data.ohlcv_columns

    # ── Public API ─────────────────────────────────────────────

    def load(self, asset_type: str) -> pd.DataFrame:
        """
        Load and clean raw OHLCV data for a given asset class.

        Args:
            asset_type: One of 'stock', 'etf', 'forex', 'crypto'.

        Returns:
            Cleaned DataFrame with columns:
            [Ticker, Date, Open, High, Low, Close, Volume]
        """
        if asset_type not in ASSET_MAP:
            raise ValueError(
                f"Unknown asset_type '{asset_type}'. "
                f"Choose from: {list(ASSET_MAP.keys())}"
            )

        path = PROJECT_ROOT / ASSET_MAP[asset_type]
        log.info(f"Loading {asset_type} data from: {path}")

        df = pd.read_csv(path, parse_dates=[self.date_col])
        df = self._clean(df, asset_type)

        log.info(
            f"  ✓ {asset_type}: {df.shape[0]:,} rows | "
            f"{df[self.ticker_col].nunique()} tickers | "
            f"{df[self.date_col].min().date()} → {df[self.date_col].max().date()}"
        )
        return df

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all 4 asset classes.

        Returns:
            Dict with keys: 'stock', 'etf', 'forex', 'crypto'
        """
        log.info("Loading all asset classes...")
        all_data = {}
        for asset_type in ASSET_MAP:
            all_data[asset_type] = self.load(asset_type)
        log.info("✓ All asset classes loaded.")
        return all_data

    def get_tickers(self, asset_type: str) -> list:
        """Return list of tickers for a given asset class."""
        df = self.load(asset_type)
        return sorted(df[self.ticker_col].unique().tolist())

    def summary(self, all_data: Optional[Dict] = None) -> pd.DataFrame:
        """
        Print and return a summary DataFrame of loaded data.

        Args:
            all_data: Dict from load_all(). Loads fresh if None.

        Returns:
            Summary DataFrame.
        """
        if all_data is None:
            all_data = self.load_all()

        rows = []
        for asset_type, df in all_data.items():
            rows.append({
                "Asset Class":  asset_type.upper(),
                "Tickers":      df[self.ticker_col].nunique(),
                "Total Rows":   f"{len(df):,}",
                "Date Start":   str(df[self.date_col].min().date()),
                "Date End":     str(df[self.date_col].max().date()),
                "Null %":       f"{df.isnull().mean().mean() * 100:.2f}%",
            })
        summary = pd.DataFrame(rows)
        return summary

    # ── Private Helpers ────────────────────────────────────────

    def _clean(self, df: pd.DataFrame, asset_type: str) -> pd.DataFrame:
        """Apply common cleaning steps to a raw OHLCV DataFrame."""

        # 1. Standardize column names (strip whitespace)
        df.columns = df.columns.str.strip()

        # 2. Ensure date column is datetime
        df[self.date_col] = pd.to_datetime(df[self.date_col])

        # 3. Sort by ticker and date
        df = df.sort_values([self.ticker_col, self.date_col]).reset_index(drop=True)

        # 4. Drop rows where Close is null/zero (data errors)
        before = len(df)
        df = df[df["Close"].notna() & (df["Close"] > 0)]
        dropped = before - len(df)
        if dropped > 0:
            log.warning(f"  Dropped {dropped} rows with null/zero Close in {asset_type}")

        # 5. Cast numeric columns
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 6. Forward-fill small gaps within each ticker (max 3 days)
        df = df.groupby(self.ticker_col, group_keys=False).apply(
            lambda g: g.set_index(self.date_col)
                       .resample("D")
                       .first()
                       .ffill(limit=3)
                       .reset_index()
        )
        df = df.sort_values([self.ticker_col, self.date_col]).reset_index(drop=True)

        return df


if __name__ == "__main__":
    loader = DataLoader()
    all_data = loader.load_all()
    print(loader.summary(all_data).to_string(index=False))
