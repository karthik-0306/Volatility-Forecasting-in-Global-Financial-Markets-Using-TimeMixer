"""
src/models/evaluation.py
─────────────────────────
All evaluation metrics, statistical tests, and result visualizations.

Metrics implemented:
  MAE, MSE, RMSE, MAPE, sMAPE, QLIKE, R², Hit Rate

Statistical tests:
  - Diebold-Mariano Test (is TimeMixer significantly better than baselines?)

Usage:
    from src.models.evaluation import Evaluator
    ev = Evaluator()
    metrics = ev.compute_metrics(y_true, y_pred)
    ev.diebold_mariano(y_true, pred_model1, pred_model2)
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

from src.utils.config import cfg
from src.utils.logger import get_logger

log = get_logger(__name__)

FIGURES_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"
METRICS_DIR = Path(__file__).resolve().parents[2] / "results" / "metrics"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


class Evaluator:
    """
    Computes all evaluation metrics and generates comparison visuals.
    """

    def __init__(self):
        self.alpha = cfg.evaluation.significance_level  # 0.05

    # ═══════════════════════════════════════════════════════════
    # Metrics
    # ═══════════════════════════════════════════════════════════

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.

        Args:
            y_true     : Array of actual volatility values.
            y_pred     : Array of predicted volatility values.
            model_name : Name for logging.

        Returns:
            Dict of metric names → values.
        """
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)

        # Remove NaN pairs
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true, y_pred = y_true[mask], y_pred[mask]

        if len(y_true) == 0:
            log.warning(f"{model_name}: No valid predictions to evaluate.")
            return {}

        err  = y_true - y_pred
        abs_err = np.abs(err)

        mae   = float(np.mean(abs_err))
        mse   = float(np.mean(err**2))
        rmse  = float(np.sqrt(mse))
        mape  = float(np.mean(abs_err / (np.abs(y_true) + 1e-8)) * 100)
        smape = float(np.mean(2 * abs_err / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100)
        qlike = float(np.mean(np.log(y_pred**2 + 1e-8) + (y_true**2) / (y_pred**2 + 1e-8)))
        r2    = float(1 - (np.sum(err**2) / (np.sum((y_true - y_true.mean())**2) + 1e-8)))

        # Hit Rate: did we predict the direction of vol change correctly?
        if len(y_true) > 1:
            true_dir = np.sign(np.diff(y_true))
            pred_dir = np.sign(np.diff(y_pred))
            hit_rate = float(np.mean(true_dir == pred_dir) * 100)
        else:
            hit_rate = np.nan

        metrics = {
            "MAE":      mae,
            "MSE":      mse,
            "RMSE":     rmse,
            "MAPE":     mape,
            "sMAPE":    smape,
            "QLIKE":    qlike,
            "R2":       r2,
            "HitRate":  hit_rate,
            "N":        int(len(y_true)),
        }

        log.info(
            f"{model_name} | MAE={mae:.4f} RMSE={rmse:.4f} "
            f"sMAPE={smape:.2f}% R²={r2:.4f} HitRate={hit_rate:.1f}%"
        )
        return metrics

    def compute_metrics_df(
        self,
        results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> pd.DataFrame:
        """
        Compute metrics for multiple models and return a comparison DataFrame.

        Args:
            results: Dict of { model_name: (y_true, y_pred) }

        Returns:
            DataFrame with models as rows, metrics as columns.
        """
        rows = []
        for model_name, (y_true, y_pred) in results.items():
            m = self.compute_metrics(y_true, y_pred, model_name)
            m["Model"] = model_name
            rows.append(m)
        df = pd.DataFrame(rows).set_index("Model")
        return df

    # ═══════════════════════════════════════════════════════════
    # Diebold-Mariano Test
    # ═══════════════════════════════════════════════════════════

    def diebold_mariano(
        self,
        y_true: np.ndarray,
        pred_1: np.ndarray,
        pred_2: np.ndarray,
        h: int = 1,
        loss: str = "squared",
        model1_name: str = "Model1",
        model2_name: str = "Model2",
    ) -> Dict:
        """
        Diebold-Mariano test: is model1's forecast significantly better than model2?

        H0: Equal predictive accuracy
        H1: Model1 is better (one-sided)

        Args:
            y_true     : Actual values.
            pred_1     : Predictions from model 1 (e.g., TimeMixer).
            pred_2     : Predictions from model 2 (e.g., GARCH).
            h          : Forecast horizon.
            loss       : Loss function — 'squared' or 'absolute'.
            model1_name: Name of model 1.
            model2_name: Name of model 2.

        Returns:
            Dict with test statistic, p-value, and verdict.
        """
        y_true = np.array(y_true, dtype=float)
        pred_1 = np.array(pred_1, dtype=float)
        pred_2 = np.array(pred_2, dtype=float)

        if loss == "squared":
            d = (y_true - pred_1)**2 - (y_true - pred_2)**2
        else:
            d = np.abs(y_true - pred_1) - np.abs(y_true - pred_2)

        n = len(d)
        d_mean = d.mean()

        # Harvey et al. (1997) small-sample correction
        gamma = np.array([np.mean(d[h:] * d[:-h]) for h in range(1, h + 1)])
        var_d = (1 / n) * (np.var(d, ddof=0) + 2 * gamma.sum())
        var_d = max(var_d, 1e-10)

        dm_stat = d_mean / np.sqrt(var_d)
        p_value = stats.t.sf(np.abs(dm_stat), df=n - 1) * 2  # two-sided

        # Is model1 significantly BETTER?
        model1_better = dm_stat < 0 and p_value < self.alpha

        result = {
            "model1":        model1_name,
            "model2":        model2_name,
            "DM_statistic":  round(float(dm_stat), 4),
            "p_value":       round(float(p_value), 4),
            "significant":   bool(p_value < self.alpha),
            "model1_better": bool(model1_better),
            "verdict": (
                f"✅ {model1_name} is significantly better (p={p_value:.4f})"
                if model1_better
                else f"⚠️  No significant difference (p={p_value:.4f})"
            ),
        }

        log.info(f"DM Test [{model1_name} vs {model2_name}]: {result['verdict']}")
        return result

    # ═══════════════════════════════════════════════════════════
    # Visualizations
    # ═══════════════════════════════════════════════════════════

    def plot_forecast(
        self,
        ds: pd.Series,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        ticker: str,
        horizon: int,
        model_name: str = "TimeMixer",
        save: bool = True,
    ) -> None:
        """Plot actual vs predicted volatility for a single ticker."""
        fig, ax = plt.subplots(figsize=tuple(cfg.viz.figsize_wide))
        ax.plot(ds, y_true, label="Actual Volatility",    color=cfg.viz.color_palette.actual, linewidth=1.5)
        ax.plot(ds, y_pred, label=f"{model_name} Forecast", color=cfg.viz.color_palette.timemixer,
                linewidth=1.5, linestyle="--")
        ax.fill_between(
            ds,
            y_pred * 0.85, y_pred * 1.15,
            alpha=0.15, color=cfg.viz.color_palette.confidence,
            label="±15% band",
        )
        ax.set_title(f"{ticker} Volatility Forecast — {model_name} (h={horizon}d)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Annualized Volatility", fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=30)
        plt.tight_layout()

        if save:
            fname = FIGURES_DIR / f"{ticker}_{model_name}_h{horizon}.png"
            fig.savefig(fname, dpi=cfg.viz.dpi, bbox_inches="tight")
            log.info(f"Saved: {fname}")
        plt.show()
        plt.close()

    def plot_model_comparison(
        self,
        metrics_df: pd.DataFrame,
        metric: str = "MAE",
        title: str = "Model Comparison",
        save: bool = True,
    ) -> None:
        """Horizontal bar chart comparing models on a given metric."""
        fig, ax = plt.subplots(figsize=(10, max(4, len(metrics_df) * 0.8)))
        colors = ["#2563EB" if i == 0 else "#94A3B8" for i in range(len(metrics_df))]
        metrics_df[metric].sort_values().plot(kind="barh", ax=ax, color=colors)
        ax.set_title(f"{title} — {metric}", fontsize=13, fontweight="bold")
        ax.set_xlabel(metric, fontsize=11)
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()

        if save:
            fname = FIGURES_DIR / f"comparison_{metric}_{title.replace(' ', '_')}.png"
            fig.savefig(fname, dpi=cfg.viz.dpi, bbox_inches="tight")
        plt.show()
        plt.close()

    def plot_heatmap(
        self,
        data: pd.DataFrame,
        title: str = "Performance Heatmap",
        metric: str = "MAE",
        save: bool = True,
    ) -> None:
        """Heatmap of metric scores — asset class × horizon."""
        fig, ax = plt.subplots(figsize=tuple(cfg.viz.figsize_square))
        sns.heatmap(
            data, annot=True, fmt=".4f", cmap="RdYlGn_r",
            ax=ax, linewidths=0.5, linecolor="white",
        )
        ax.set_title(f"{title} ({metric})", fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save:
            fname = FIGURES_DIR / f"heatmap_{metric}.png"
            fig.savefig(fname, dpi=cfg.viz.dpi, bbox_inches="tight")
        plt.show()
        plt.close()
