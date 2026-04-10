"""
src/models/timemixer.py
────────────────────────
TimeMixer training, inference, and MLflow integration wrapper.

Usage:
    from src.models.timemixer import TimeMixerTrainer
    trainer = TimeMixerTrainer()
    nf = trainer.train(train_df, asset_type="stock", horizon=12)
    preds = trainer.predict(nf, train_df)
"""

import os
os.environ["LIT_LOG_LEVEL"] = "error" # Lightning log level
os.environ["PYTORCH_LIGHTNING_SUPPRESS_WARNINGS"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Disabled to allow GPU training as requestedm noise

import logging
import warnings
# Suppress PyTorch Lightning and system noise
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.core").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.accelerators").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities.seed").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric.utilities.seed").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
warnings.filterwarnings("ignore", category=UserWarning, module="lightning_fabric")
warnings.filterwarnings("ignore", ".*Tip: For seamless cloud logging*")
warnings.filterwarnings("ignore", ".*You are using a CUDA device*")
warnings.filterwarnings("ignore", ".*Seed set to*")

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

import mlflow
import mlflow.sklearn

from neuralforecast import NeuralForecast
from neuralforecast.models import TimeMixer

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "timemixer"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class TimeMixerTrainer:
    """
    Wraps NeuralForecast TimeMixer with:
      - MLflow experiment tracking
      - Automatic model saving / loading
      - Clean train/predict interface
    """

    def __init__(self):
        self.horizons   = cfg.timemixer.horizons
        self.input_size = cfg.timemixer.input_size
        self.max_steps  = cfg.timemixer.max_steps
        self.lr         = cfg.timemixer.learning_rate
        self.dropout    = cfg.timemixer.dropout
        self.seed       = cfg.timemixer.random_seed
        self.freq       = cfg.timemixer.freq

        # Setup MLflow
        mlflow.set_tracking_uri(
            str(Path(__file__).resolve().parents[2] / cfg.mlflow.tracking_uri)
        )
        mlflow.set_experiment(cfg.mlflow.experiment_name)

    # ═══════════════════════════════════════════════════════════
    # Training
    # ═══════════════════════════════════════════════════════════

    def train(
        self,
        train_df: pd.DataFrame,
        asset_type: str,
        horizon: int,
        input_size: Optional[int] = None,
        max_steps: Optional[int] = None,
        learning_rate: Optional[float] = None,
        dropout: Optional[float] = None,
        run_name: Optional[str] = None,
        ticker: Optional[str] = None,
    ) -> NeuralForecast:
        """
        Train a TimeMixer model and log everything to MLflow.

        Args:
            train_df      : NeuralForecast-format DataFrame [unique_id, ds, y].
            asset_type    : 'stock', 'etf', 'forex', or 'crypto'.
            horizon       : Forecast horizon in days.
            input_size    : Lookback window (default from config).
            max_steps     : Training steps (default from config).
            learning_rate : LR (default from config).
            dropout       : Dropout rate (default from config).
            run_name      : MLflow run name (auto-generated if None).

        Returns:
            Trained NeuralForecast instance.
        """
        # Use config defaults unless overridden
        input_size    = input_size    or self.input_size
        max_steps_val = max_steps     or self.max_steps
        lr            = learning_rate or self.lr
        drop          = dropout       or self.dropout
        n_series      = train_df["unique_id"].nunique()

        if run_name is None:
            prefix = f"TimeMixer_{ticker}" if ticker else f"TimeMixer_{asset_type}"
            run_name = f"{prefix}_h{horizon}"

        # We will suppress this to keep notebook output clean and structured
        # log.info(f"Training TimeMixer | {run_name} | n_series={n_series}")

        params = {
            "asset_type":    asset_type,
            "horizon":       horizon,
            "input_size":    input_size,
            "max_steps":     max_steps_val,
            "learning_rate": lr,
            "dropout":       drop,
            "n_series":      n_series,
            "random_seed":   self.seed,
        }

        with mlflow.start_run(run_name=run_name):
            # Log hyperparameters
            mlflow.log_params(params)

            # Build model
            model = TimeMixer(
                h=horizon,
                input_size=input_size,
                n_series=n_series,
                max_steps=max_steps_val,
                learning_rate=lr,
                dropout=drop,
                scaler_type=cfg.timemixer.scaler_type,
                random_seed=self.seed,
                start_padding_enabled=cfg.timemixer.start_padding_enabled,
                batch_size=8,
                enable_progress_bar=False,
                enable_model_summary=False,
            )

            nf = NeuralForecast(models=[model], freq=self.freq)
            nf.fit(df=train_df)

            # Save model artifact
            model_path = self._model_path(asset_type, horizon, ticker=ticker)
            self._save(nf, model_path)
            mlflow.log_artifact(str(model_path))

            # log.info(f"  ✓ Training complete | saved → {model_path}")

        return nf

    def train_all(
        self,
        train_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, Dict[int, NeuralForecast]]:
        """
        Train TimeMixer for all asset classes × all horizons.

        Args:
            train_data: Dict of { asset_type: train_df }

        Returns:
            Dict of { asset_type: { horizon: NeuralForecast } }
        """
        trained = {asset: {} for asset in train_data}

        for asset_type, train_df in train_data.items():
            log.info(f"\n{'='*50}\nTraining {asset_type.upper()} models\n{'='*50}")
            for h in self.horizons:
                trained[asset_type][h] = self.train(train_df, asset_type, h)

        log.info("\n✅ All models trained successfully!")
        return trained

    # ═══════════════════════════════════════════════════════════
    # Prediction
    # ═══════════════════════════════════════════════════════════

    def predict(
        self,
        nf: NeuralForecast,
        train_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate predictions using a trained NeuralForecast model.

        Args:
            nf       : Trained NeuralForecast instance.
            train_df : Training context DataFrame [unique_id, ds, y].

        Returns:
            DataFrame with [unique_id, ds, TimeMixer] columns.
        """
        forecast = nf.predict(df=train_df)
        return forecast

    # ═══════════════════════════════════════════════════════════
    # Save / Load
    # ═══════════════════════════════════════════════════════════

    def _model_path(self, asset_type: str, horizon: int, ticker: Optional[str] = None) -> Path:
        if ticker:
            path = MODELS_DIR / asset_type / ticker / f"h{horizon}.pkl"
        else:
            path = MODELS_DIR / f"{asset_type}_h{horizon}.pkl"
            
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _save(self, nf: NeuralForecast, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(nf, f)
        # log.info(f"  Model saved: {path}")

    def load(self, asset_type: str, horizon: int) -> NeuralForecast:
        """Load a previously trained model."""
        path = self._model_path(asset_type, horizon)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}. Train first.")
        with open(path, "rb") as f:
            nf = pickle.load(f)
        log.info(f"  Model loaded: {path}")
        return nf

    def load_all(self) -> Dict[str, Dict[int, NeuralForecast]]:
        """Load all saved models."""
        asset_types = ["stock", "etf", "forex", "crypto"]
        loaded = {a: {} for a in asset_types}
        for asset in asset_types:
            for h in self.horizons:
                try:
                    loaded[asset][h] = self.load(asset, h)
                except FileNotFoundError:
                    log.warning(f"  Model not found: {asset} h={h} — skipping")
        return loaded
