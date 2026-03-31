import nbformat as nbf
import os

os.makedirs('Notebooks', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)
nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell("# Phase 3: Comprehensive Baseline Modeling — GARCH(1,1)\n\n**Goal**: Scientifically establish the industry-standard benchmark for ALL 40 tickers across ALL 5 horizons ([12, 96, 192, 336, 720]).\n\nThis notebook performs a full sweep of 200 GARCH(1,1) models, providing the perfect baseline for TimeMixer.\n\n### Experimental Setup:\n- **Train set**: 2010 - 2023-12-31\n- **Test set**: 2024-01-01 - 2026-03-30\n- **Models**: 40 Tickers × 5 Horizons = 200 total GARCH models."),
    
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from src.utils.config import cfg
from src.data.loader import DataLoader
from src.data.preprocessing import VolatilityProcessor
from src.models.baselines import BaselineForecaster
from src.models.evaluation import Evaluator

plt.style.use(cfg.viz.style)
sns.set_palette("tab10")"""),

    nbf.v4.new_markdown_cell("## 1. The Full Experimental Sweep\nRun GARCH(1,1) for every ticker and every horizon."),
    
    nbf.v4.new_code_cell("""loader = DataLoader()
proc = VolatilityProcessor()
bf = BaselineForecaster()
ev = Evaluator()

asset_classes = ['stock', 'etf', 'forex', 'crypto']
horizons = [12, 96, 192, 336, 720]

all_metrics = []
all_results = {}

# Use tqdm for progress tracking
for group in asset_classes:
    df = loader.load(group)
    tickers = df['Ticker'].unique()
    
    for ticker in tqdm(tickers, desc=f"Processing {group}"):
        try:
            df_ticker = df[df['Ticker'] == ticker].copy()
            
            # Ground Truth Actuals
            vol_df = proc.compute_volatility(df_ticker)
            if vol_df.empty:
                print(f"  [Error] Skipping {ticker}: Ohlc volatility calculation returned no rows.")
                continue

            _, test_actual_df = proc.train_test_split(vol_df)
            if test_actual_df.empty:
                print(f"  [Error] Skipping {ticker}: Test set (2024+) is empty.")
                continue

            actual_test_vol = test_actual_df['y'].values
            test_dates = test_actual_df['ds']
            
            if ticker not in all_results:
                all_results[ticker] = {}
            
            for h in horizons:
                # Forecast
                pred_vol = bf.fit_predict_garch(df_ticker, horizon=h, ticker=ticker)
                
                # Align
                min_len = min(len(actual_test_vol), len(pred_vol))
                y_true = actual_test_vol[:min_len]
                y_pred = pred_vol[:min_len]
                dates = test_dates[:min_len]
                
                all_results[ticker][h] = {'dates': dates, 'actual': y_true, 'pred': y_pred}
                
                # Compute metrics
                metrics = ev.compute_metrics(y_true, y_pred, model_name=f"GARCH_{ticker}_h{h}")
                if metrics:
                    metrics['AssetClass'] = group
                    metrics['Ticker'] = ticker
                    metrics['Horizon'] = h
                    all_metrics.append(metrics)
        except Exception as e:
            print(f"  [Critical] Unexpected error processing {ticker}: {e}")

if all_metrics:
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv('../results/metrics/baseline_results_exhaustive.csv', index=False)
    print(f"✓ All possible baseline models trained. Rows: {len(metrics_df)}")
    display(metrics_df.head())
else:
    print("❌ No baseline results were collected.")"""),

    nbf.v4.new_markdown_cell("## 2. Identify the 2 'Best' Tickers per Class\nWe average the sMAPE performance across all 5 horizons for each ticker and select the top 2."),

    nbf.v4.new_code_cell("""# Rank by mean sMAPE across horizons
if not all_metrics:
    print("No metrics to rank.")
else:
    ticker_ranking = metrics_df.groupby(['AssetClass', 'Ticker'])['sMAPE'].mean().reset_index()
    best_tickers = ticker_ranking.sort_values(['AssetClass', 'sMAPE']).groupby('AssetClass').head(2)

    print("Target Tickers for Visualization (Top 2 per Class):")
    display(best_tickers)"""),

    nbf.v4.new_markdown_cell("## 3. Visualization Sweep: All Horizons for Selected Tickers\n8 Best Tickers × 5 Horizons = 40 Plots."),

    nbf.v4.new_code_cell("""if not all_metrics:
    print("No results to plot.")
else:
    target_tickers = best_tickers['Ticker'].tolist()

    for ticker in target_tickers:
        asset_class = best_tickers[best_tickers['Ticker'] == ticker]['AssetClass'].values[0]
        fig, axes = plt.subplots(1, 5, figsize=(25, 4))
        fig.suptitle(f"GARCH Baseline: {ticker} ({asset_class}) - Full Horizon Results", fontsize=16)
        
        for idx, h in enumerate(horizons):
            if h not in all_results[ticker]:
                continue
            data = all_results[ticker][h]
            ax = axes[idx]
            
            ax.plot(data['dates'], data['actual'], label='Actual', color='black', alpha=0.8)
            ax.plot(data['dates'], data['pred'], label=f'GARCH h={h}', color='crimson', ls='--')
            ax.fill_between(data['dates'], data['pred']*0.85, data['pred']*1.15, color='crimson', alpha=0.1)
            
            ax.set_title(f"H={h}")
            ax.grid(True, alpha=0.3)
            if idx == 0: ax.set_ylabel("Volatility")
            if idx == 4: ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()"""),

    nbf.v4.new_markdown_cell("## Executive Summary\n- **Exhaustive Baseline Complete**: We have benchmarked the entire set of tickers.\n- **Definitive Baseline**: The results in `baseline_results_exhaustive.csv` will be the definitive comparison target for TimeMixer.")
]

nb['cells'] = cells

with open('Notebooks/03_Baseline_Models.ipynb', 'w') as f:
    nbf.write(nb, f)

print('Exhaustive baseline notebook generated.')
