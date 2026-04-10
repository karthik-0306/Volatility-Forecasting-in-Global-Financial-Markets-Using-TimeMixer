import nbformat as nbf
import os

os.makedirs('Notebooks', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)

nb = nbf.v4.new_notebook()

# ─────────────────────────────────────────────────────────────────────
# Cell 1: Title
# ─────────────────────────────────────────────────────────────────────
cells = [
    nbf.v4.new_markdown_cell("""\
# Phase 4: Exhaustive TimeMixer Training
### Volatility Forecasting Across Global Financial Markets

Training **200 independent univariate TimeMixer models** (40 Tickers x 5 Horizons).

| Configuration | Value |
|---|---|
| Model | TimeMixer (NeuralForecast) |
| Horizons | 12 · 96 · 192 · 336 · 720 days |
| Asset Classes | Stock · ETF · Forex · Crypto |
| Tickers per class | 10 |
| Total Models | 200 |
| Accelerator | CPU (H100 is in shared MIG mode) |
"""),

# ─────────────────────────────────────────────────────────────────────
# Cell 2: Setup — CUDA must be killed BEFORE torch import
# ─────────────────────────────────────────────────────────────────────
    nbf.v4.new_code_cell(
r"""import os, sys, logging, warnings

# ── Enable GPU — using MIG device 2 which has 19GB free memory ──
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["LIT_LOG_LEVEL"] = "error"
os.environ["PYTORCH_LIGHTNING_SUPPRESS_WARNINGS"] = "1"
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

sys.path.append('..')

import torch
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm as tqdm_nb   # plain text — works in VS Code Remote SSH

from src.utils.config import cfg
from src.data.loader import DataLoader
from src.data.preprocessing import VolatilityProcessor
from src.models.timemixer import TimeMixerTrainer
from src.models.evaluation import Evaluator

print(f"CUDA visible to torch : {torch.cuda.is_available()} (Targeting Free MIG Slice)")
print(f"CPU cores             : {os.cpu_count()}")

loader  = DataLoader()
proc    = VolatilityProcessor()
trainer = TimeMixerTrainer()
ev      = Evaluator()

HORIZONS      = [12, 96, 192, 336, 720]
all_tm_metrics = []
all_preds      = {}

print("Setup complete. Ready to train 200 models on targeted H100 MIG slice.")
"""),
]

# ─────────────────────────────────────────────────────────────────────
# Per-asset-class cells — one per class
# ─────────────────────────────────────────────────────────────────────
for group in ['stock', 'etf', 'forex', 'crypto']:
    label = group.upper()

    cells.append(nbf.v4.new_markdown_cell(
        f"---\n## Training Asset Class: {label}\n"
    ))

    code = """\
# ═══════════════════════════════════════════════════════
# {label} — TimeMixer GPU Training
# ═══════════════════════════════════════════════════════
_group  = '{group}'
df_raw  = loader.load(_group)
tickers = df_raw['Ticker'].unique()

print("=" * 60)
print(f"  ASSET CLASS : {{_group.upper()}}   ({{len(tickers)}} tickers x {{len(HORIZONS)}} horizons)")
print("=" * 60)

group_metrics = []

for ticker in tickers:
    print(f"\\n  [ {{ticker}} ]")

    df_t = df_raw[df_raw['Ticker'] == ticker].copy()
    vol  = proc.compute_volatility(df_t)
    if vol.empty:
        print("    => skipped (no volatility data)")
        continue

    tr_df, te_df = proc.train_test_split(vol)
    if tr_df.empty or te_df.empty:
        print("    => skipped (insufficient data)")
        continue

    tr_nf = proc.to_neuralforecast_format(tr_df)
    te_y  = te_df['y'].values
    all_preds.setdefault(ticker, {{}})

    for h in HORIZONS:
        # One bar per horizon — fills to 100%, stays visible with sMAPE on right
        hbar = tqdm_nb(
            total=1,
            desc=f"    h={{h:>3}}d",
            bar_format="{{desc}} : {{bar:40}} {{n_fmt}}/{{total_fmt}}  {{postfix}}",
            ncols=100,
            leave=True,
        )
        hbar.set_postfix_str("training ...")

        try:
            nf   = trainer.train(tr_nf, asset_type=_group, horizon=h, ticker=ticker, max_steps=1000)
            fdf  = trainer.predict(nf, tr_nf)
            yhat = fdf['TimeMixer'].values

            yt = te_y[:min(h, len(te_y))]
            yp = yhat[:len(yt)]
            yt = yt[:len(yp)]
            ds = te_df['ds'].values[:len(yp)]
            all_preds[ticker][h] = {{'dates': ds, 'actual': yt, 'pred': yp}}

            m = ev.compute_metrics(yt, yp, model_name=f"TM_{{ticker}}_h{{h}}")
            if m:
                m.update({{'AssetClass': _group, 'Ticker': ticker, 'Horizon': h}})
                all_tm_metrics.append(m)
                group_metrics.append(m)
                r2_val = m.get('R2', m.get('R2', float('nan')))
                hbar.set_postfix_str(
                    f"sMAPE = {{m['sMAPE']:.2f}}%  |  "
                    f"HitRate = {{m['HitRate']:.1f}}%  |  "
                    f"R2 = {{r2_val:.3f}}"
                )
            else:
                hbar.set_postfix_str("done — no metrics")

        except Exception as e:
            hbar.set_postfix_str(f"FAILED: {{str(e)[:60]}}")

        hbar.update(1)
        hbar.close()

if group_metrics:
    gdf = pd.DataFrame(group_metrics)
    print(f"\\n  [DONE] {{_group.upper()}} — {{len(gdf)}} models  |  mean sMAPE = {{gdf['sMAPE'].mean():.2f}}%")
""".format(group=group, label=label)

    cells.append(nbf.v4.new_code_cell(code))

# ─────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("---\n## Save & Summarize Results"))
cells.append(nbf.v4.new_code_cell(
r"""METRICS_DIR = Path('../results/metrics')
PREDS_DIR   = Path('../results/predictions')
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PREDS_DIR.mkdir(parents=True, exist_ok=True)

tm_df = pd.DataFrame(all_tm_metrics)
tm_df.to_csv(METRICS_DIR / 'timemixer_results_exhaustive.csv', index=False)

with open(PREDS_DIR / 'timemixer_all_preds.pkl', 'wb') as f:
    pickle.dump(all_preds, f)

print(f"Saved {len(tm_df)} model records to results/metrics/timemixer_results_exhaustive.csv")
print(f"Saved predictions to results/predictions/timemixer_all_preds.pkl")
print(f"Mean sMAPE (all): {tm_df['sMAPE'].mean():.2f}%")
"""))

cells.append(nbf.v4.new_markdown_cell("## Summary — Average sMAPE (%) by Asset Class x Horizon"))
cells.append(nbf.v4.new_code_cell(
r"""from IPython.display import display

summary = tm_df.groupby(['AssetClass', 'Horizon'])['sMAPE'].mean().unstack()
summary.columns = [f'h={c}' for c in summary.columns]

styled = (
    summary.style
    .format("{:.2f}%")
    .background_gradient(cmap='RdYlGn_r', axis=None)
    .set_caption("Mean sMAPE (%) — Lower is Better")
    .set_properties(**{'font-size': '13px', 'text-align': 'center'})
)
display(styled)
"""))

nb['cells'] = cells

with open('Notebooks/04_TimeMixer_Training.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Generated: Notebooks/04_TimeMixer_Training.ipynb")
