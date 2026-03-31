"""
src/models/baselines.py
────────────────────────
Baseline volatility forecasting model: GARCH(1,1)

The sole baseline is GARCH(1,1) — the industry-standard model for
conditional volatility forecasting. It runs on ALL 40 tickers × all
5 horizons, giving a perfectly symmetric comparison against TimeMixer.

Usage:
    from src.models.baselines import BaselineForecaster
    bf = BaselineForecaster()
    pred_vol = bf.fit_predict_garch(raw_df, horizon=96, ticker="AAPL")
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

GARCH_DIR = Path(__file__).resolve().parents[2] / "models" / "garch"
GARCH_DIR.mkdir(parents=True, exist_ok=True)


class BaselineForecaster:
    """
    GARCH(1,1) baseline forecaster.

    Fits on log-returns up to the 85/15 train/test split and produces
    an h-step-ahead conditional volatility forecast, annualized to match
    the Yang-Zhang target used by TimeMixer.
    """

    def __init__(self):
        self.p   = cfg.baselines.garch.p    # 1
        self.q   = cfg.baselines.garch.q    # 1
        self.vol = cfg.baselines.garch.vol  # "Garch"

    # ═══════════════════════════════════════════════════════════
    # GARCH(1,1)
    # ═══════════════════════════════════════════════════════════

    def fit_predict_garch(
        self,
        raw_df: pd.DataFrame,
        horizon: int,
        ticker: str,
    ) -> np.ndarray:
        """
        Fit GARCH(1,1) on log-returns and forecast conditional volatility.

        Args:
            raw_df  : Raw OHLCV DataFrame for a single ticker (all dates).
            horizon : Forecast horizon in days (e.g. 12, 96, 192, 336, 720).
            ticker  : Ticker symbol — used for saving and logging.

        Returns:
            pred_vol : np.ndarray of shape (horizon,) — annualized
                       conditional volatility forecasts.
        """
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError("Install 'arch' package: pip install arch")

        df = raw_df.sort_values("Date").copy()

        df['Date'] = pd.to_datetime(df['Date'])

        # Log-returns scaled to % (arch library convention)
        log_ret_series = np.log(df["Close"] / df["Close"].shift(1)) * 100
        df['log_ret'] = log_ret_series
        df = df.dropna(subset=['log_ret'])

        # Absolute Train/Test split via config threshold
        split_dt = pd.to_datetime(cfg.data.split_date)
        train_df = df[df['Date'] < split_dt]
        train_ret = train_df['log_ret']

        if len(train_ret) < 100:
            log.warning(f"Insufficient training data for {ticker} (N={len(train_ret)}). Returning fallback volatility.")
            fallback_val = train_ret.std() * np.sqrt(252) / 100 if not train_ret.empty else 0.2
            return np.full(horizon, fallback_val)

        # Fit GARCH(p, q) with aggressive max iterations bounds
        am  = arch_model(train_ret, vol=self.vol, p=self.p, q=self.q, dist=cfg.baselines.garch.dist)
        res = am.fit(disp="off", options={'maxiter': 50})

        # h-step-ahead forecast — returns conditional variance
        forecasts = res.forecast(horizon=horizon, reindex=False)
        pred_var  = forecasts.variance.values[-1]          # shape: (horizon,)
        pred_vol  = np.sqrt(pred_var) * np.sqrt(252) / 100 # annualize from %

        log.info(f"GARCH({self.p},{self.q}) | {ticker} | h={horizon} → "
                 f"mean_vol={pred_vol.mean():.4f}")

        # Persist fitted model
        path = GARCH_DIR / f"{ticker}_garch_h{horizon}.pkl"
        with open(path, "wb") as f:
            pickle.dump(res, f)
        log.debug(f"Saved GARCH model → {path}")

        return pred_vol
