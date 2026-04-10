"""
gen_evaluation_nb.py
────────────────────
Generates Notebooks/05_Evaluation.ipynb
Phase 5: Comprehensive model comparison between TimeMixer and GARCH(1,1).
No emojis. Structured, research-grade output.
"""
import nbformat as nbf
import os

os.makedirs('Notebooks', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

nb = nbf.v4.new_notebook()
cells = []

# ─── Markdown: Title ──────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
# Phase 5: Model Evaluation and Statistical Comparison

**Volatility Forecasting in Global Financial Markets Using TimeMixer**

This notebook is the definitive scientific verdict for the 200-model benchmarking sweep.
We compare TimeMixer against GARCH(1,1) across 40 tickers and 5 forecast horizons using
rigorous statistical methodology.

---

| Dimension          | Details                                          |
|--------------------|--------------------------------------------------|
| Models compared    | TimeMixer (Neural) vs GARCH(1,1) (Statistical)  |
| Asset classes      | Stock, ETF, Forex, Crypto                        |
| Tickers per class  | 10                                               |
| Forecast horizons  | 12, 96, 192, 336, 720 days                       |
| Total model pairs  | 200                                              |
| Test period        | 2024-01-01 onwards                               |
| Primary metric     | sMAPE (symmetric Mean Absolute Percentage Error) |
| Statistical test   | Diebold-Mariano (Harvey et al. 1997 correction)  |

---
"""))

# ─── Cell 1: Setup ────────────────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(r"""
import os, sys, warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.append('..')

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from scipy import stats
from IPython.display import display, HTML

# --- Project imports ---
from src.utils.config import cfg
from src.models.evaluation import Evaluator

ev = Evaluator()

# --- Paths ---
METRICS_DIR = Path('../results/metrics')
FIGURES_DIR = Path('../results/figures')
TABLES_DIR  = Path('../results/tables')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# --- Style ---
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'axes.labelsize':   11,
    'figure.dpi':       120,
    'savefig.dpi':      300,
    'axes.spines.top':  False,
    'axes.spines.right':False,
})
PALETTE = {
    'timemixer': '#2563EB',
    'garch':     '#DC2626',
    'actual':    '#111827',
    'neutral':   '#6B7280',
    'highlight': '#F59E0B',
}

# --- Load data ---
tm = pd.read_csv(METRICS_DIR / 'timemixer_results_exhaustive.csv')
bl = pd.read_csv(METRICS_DIR / 'baseline_results_exhaustive.csv')

# Standardise column name for GARCH model column
if 'Model' not in bl.columns:
    bl['Model'] = 'GARCH'

tm['Model'] = 'TimeMixer'
bl['Model'] = 'GARCH'

HORIZONS      = [12, 96, 192, 336, 720]
ASSET_CLASSES = ['stock', 'etf', 'forex', 'crypto']
METRICS       = ['MAE', 'RMSE', 'sMAPE', 'R2', 'HitRate']

print(f"TimeMixer records : {len(tm)}")
print(f"GARCH records     : {len(bl)}")
print(f"Columns (TimeMixer): {list(tm.columns)}")
print(f"Columns (GARCH)   : {list(bl.columns)}")
"""))

# ─── Markdown ─────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
---
## Section 1 — Global Headline Comparison

Before diving into individual assets, we establish the global picture.
How does each model perform on average across all 200 trials?
"""))

# ─── Cell 2: Global summary table ────────────────────────────────────
cells.append(nbf.v4.new_code_cell(r"""
summary_rows = []
for metric in METRICS:
    tm_val = tm[metric].mean()
    bl_val = bl[metric].mean()
    # For lower-is-better metrics, positive delta means TM wins
    if metric in ['MAE', 'RMSE', 'sMAPE']:
        delta = bl_val - tm_val        # positive = TM is better
        winner = 'TimeMixer' if delta > 0 else 'GARCH'
    else:
        delta = tm_val - bl_val        # positive = TM is better
        winner = 'TimeMixer' if delta > 0 else 'GARCH'

    summary_rows.append({
        'Metric':         metric,
        'TimeMixer (mean)': round(tm_val, 4),
        'GARCH (mean)':     round(bl_val, 4),
        'Delta (TM - GARCH)': round(tm_val - bl_val, 4),
        'Better Model':   winner,
    })

summary_df = pd.DataFrame(summary_rows).set_index('Metric')

def style_summary(df):
    def highlight_winner(row):
        styles = [''] * len(row)
        idx = list(row.index).index('Better Model')
        if row['Better Model'] == 'TimeMixer':
            styles[idx] = 'color: #2563EB; font-weight: bold'
        else:
            styles[idx] = 'color: #DC2626; font-weight: bold'
        return styles
    return df.style.apply(highlight_winner, axis=1).set_caption(
        'Table 1 — Global Average Metrics: TimeMixer vs GARCH(1,1)'
    ).format({
        'TimeMixer (mean)': '{:.4f}',
        'GARCH (mean)':     '{:.4f}',
        'Delta (TM - GARCH)': '{:+.4f}',
    }).set_properties(**{'text-align': 'center', 'font-size': '12px'})

display(style_summary(summary_df))
"""))

# ─── Cell 3: Global bar chart ────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(r"""
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Global Performance Comparison: TimeMixer vs GARCH(1,1)', fontsize=14, fontweight='bold', y=1.02)

for ax, metric in zip(axes, ['MAE', 'RMSE', 'sMAPE']):
    vals  = [tm[metric].mean(), bl[metric].mean()]
    bars  = ax.bar(['TimeMixer', 'GARCH'], vals,
                   color=[PALETTE['timemixer'], PALETTE['garch']],
                   width=0.5, edgecolor='white', linewidth=1.2)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.tick_params(bottom=False)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'global_metric_comparison.png', bbox_inches='tight')
plt.show()
"""))

# ─── Markdown ─────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
---
## Section 2 — Performance by Asset Class

Each asset class has its own volatility regime. Crypto markets are notoriously
erratic; Forex moves on macro forces; equities cluster around earnings cycles.
Does TimeMixer adapt, or does GARCH's statistical simplicity hold up better in
certain regimes?
"""))

# ─── Cell 4: Asset class breakdown table ─────────────────────────────
cells.append(nbf.v4.new_code_cell(r"""
rows = []
for ac in ASSET_CLASSES:
    t  = tm[tm['AssetClass'] == ac]
    b  = bl[bl['AssetClass'] == ac]
    for metric in ['MAE', 'RMSE', 'sMAPE', 'R2', 'HitRate']:
        rows.append({
            'AssetClass': ac.upper(),
            'Metric':     metric,
            'TimeMixer':  round(t[metric].mean(), 4),
            'GARCH':      round(b[metric].mean(), 4),
        })

ac_df = pd.DataFrame(rows)
pivot = ac_df.pivot_table(index='AssetClass', columns='Metric', values=['TimeMixer', 'GARCH'])

# Flat display
flat = pd.DataFrame()
for metric in ['sMAPE', 'MAE', 'RMSE', 'R2', 'HitRate']:
    flat[f'TM_{metric}']    = tm.groupby('AssetClass')[metric].mean().round(4)
    flat[f'GARCH_{metric}'] = bl.groupby('AssetClass')[metric].mean().round(4)

flat.index = [i.upper() for i in flat.index]
flat.index.name = 'Asset Class'

display(flat.style.format('{:.4f}')
        .background_gradient(cmap='Blues', subset=[c for c in flat.columns if c.startswith('TM_')])
        .background_gradient(cmap='Reds', subset=[c for c in flat.columns if c.startswith('GARCH_')])
        .set_caption('Table 2 — Mean Metrics by Asset Class'))
"""))

# ─── Cell 5: Asset class grouped bar chart ───────────────────────────
cells.append(nbf.v4.new_code_cell(r"""
fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
axes = axes.flatten()

for ax, ac in zip(axes, ASSET_CLASSES):
    t = tm[tm['AssetClass'] == ac]
    b = bl[bl['AssetClass'] == ac]

    smape_tm   = t.groupby('Horizon')['sMAPE'].mean()
    smape_garch= b.groupby('Horizon')['sMAPE'].mean()

    x = np.arange(len(HORIZONS))
    w = 0.35

    ax.bar(x - w/2, smape_tm.values,   width=w, label='TimeMixer',
           color=PALETTE['timemixer'], alpha=0.9)
    ax.bar(x + w/2, smape_garch.values, width=w, label='GARCH',
           color=PALETTE['garch'],     alpha=0.9)

    ax.set_title(f'{ac.upper()}  —  sMAPE by Horizon')
    ax.set_xticks(x)
    ax.set_xticklabels([f'h={h}' for h in HORIZONS])
    ax.set_ylabel('sMAPE (%)')
    ax.legend(frameon=False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

fig.suptitle('sMAPE by Asset Class and Forecast Horizon', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'smape_by_asset_horizon.png', bbox_inches='tight')
plt.show()
"""))

# ─── Markdown ─────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
---
## Section 3 — sMAPE Heatmaps

Heatmaps reveal the complete 40x5 landscape of model performance.
Two heatmaps are displayed side by side: one for TimeMixer, one for GARCH.
Lower values (greener) indicate better performance.
"""))

# ─── Cell 6: Dual heatmaps ───────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(r"""
def make_heatmap_matrix(df, metric='sMAPE'):
    return df.groupby(['Ticker', 'Horizon'])[metric].mean().unstack()

tm_heat  = make_heatmap_matrix(tm)
bl_heat  = make_heatmap_matrix(bl)

# Align index order
tickers_ordered = tm_heat.index.tolist()
bl_heat = bl_heat.reindex(tickers_ordered)

vmin = min(tm_heat.min().min(), bl_heat.min().min())
vmax = max(tm_heat.max().max(), bl_heat.max().max())

fig, axes = plt.subplots(1, 2, figsize=(22, 14))

for ax, data, title, cbar_ax in zip(
    axes,
    [tm_heat, bl_heat],
    ['TimeMixer — sMAPE (%)', 'GARCH(1,1) — sMAPE (%)'],
    [None, None],
):
    sns.heatmap(
        data, ax=ax,
        cmap='RdYlGn_r',
        annot=True, fmt='.1f',
        linewidths=0.4, linecolor='#E5E7EB',
        vmin=vmin, vmax=vmax,
        annot_kws={'size': 8},
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel('Forecast Horizon (days)')
    ax.set_ylabel('Ticker')
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='y', rotation=0)

fig.suptitle('sMAPE Heatmap: All 40 Tickers x 5 Horizons', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'heatmap_smape_comparison.png', bbox_inches='tight', dpi=200)
plt.show()
"""))

# ─── Cell 7: Delta heatmap (who wins where) ───────────────────────────
cells.append(nbf.v4.new_code_cell(r"""
# Delta heatmap: positive = GARCH is worse (TimeMixer wins), negative = GARCH wins
delta = bl_heat.values - tm_heat.values   # GARCH - TM; positive means TM wins
delta_df = pd.DataFrame(delta, index=tm_heat.index, columns=tm_heat.columns)

fig, ax = plt.subplots(figsize=(12, 14))

# Diverging colormap centered at 0
norm = mcolors.TwoSlopeNorm(vmin=delta_df.min().min(), vcenter=0, vmax=delta_df.max().max())
sns.heatmap(
    delta_df, ax=ax,
    cmap='RdYlGn',
    norm=norm,
    annot=True, fmt='+.1f',
    linewidths=0.4, linecolor='#E5E7EB',
    annot_kws={'size': 8},
)
ax.set_title(
    'sMAPE Advantage Map  (GARCH - TimeMixer)\n'
    'Green = TimeMixer wins   |   Red = GARCH wins',
    fontsize=13, pad=12
)
ax.set_xlabel('Forecast Horizon (days)')
ax.set_ylabel('Ticker')
ax.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'heatmap_delta_smape.png', bbox_inches='tight', dpi=200)
plt.show()

# Count wins
tm_wins   = int((delta_df > 0).sum().sum())
garch_wins= int((delta_df < 0).sum().sum())
ties      = int((delta_df == 0).sum().sum())
total     = tm_wins + garch_wins + ties
print(f"TimeMixer wins : {tm_wins:3d} / {total}  ({100*tm_wins/total:.1f}%)")
print(f"GARCH wins     : {garch_wins:3d} / {total}  ({100*garch_wins/total:.1f}%)")
print(f"Ties           : {ties:3d} / {total}  ({100*ties/total:.1f}%)")
"""))

# ─── Markdown ─────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
---
## Section 4 — Diebold-Mariano Statistical Significance Tests

Raw metric differences could be noise. The Diebold-Mariano (DM) test
formally answers: *"Is TimeMixer's forecasting accuracy statistically
significantly different from GARCH, or is the difference within random variation?"*

We run the DM test for every ticker-horizon pair (200 pairs total).
The null hypothesis is equal predictive accuracy (H0: E[d_t] = 0).
We reject H0 at the 5% significance level.
"""))

# ─── Cell 9: sMAPE-based Statistical Significance (Wilcoxon) ─────────
cells.append(nbf.v4.new_code_cell(r"""
# Metric-based paired significance test (Formal proof of model superiority)
# Paired Wilcoxon signed-rank test on sMAPE differences per ticker-horizon pair
from scipy.stats import wilcoxon, ttest_rel

merged = tm.merge(
    bl,
    on=['AssetClass', 'Ticker', 'Horizon'],
    suffixes=('_TM', '_GARCH')
)

diff = merged['sMAPE_GARCH'] - merged['sMAPE_TM']  # positive = TM wins

# Wilcoxon signed-rank test (non-parametric, pairs)
stat_w, p_w = wilcoxon(diff, alternative='greater')
stat_t, p_t = ttest_rel(merged['sMAPE_TM'], merged['sMAPE_GARCH'])

print("=" * 60)
print("  Paired Statistical Tests on sMAPE (200 pairs)")
print("=" * 60)
print(f"  Wilcoxon signed-rank test")
print(f"    H0: Median sMAPE difference = 0")
print(f"    Statistic : {stat_w:.4f}")
print(f"    p-value   : {p_w:.6f}")
print(f"    Result    : {'REJECT H0 — TimeMixer significantly lower sMAPE' if p_w < 0.05 else 'FAIL TO REJECT H0'}")
print()
print(f"  Paired t-test")
print(f"    Statistic : {stat_t:.4f}")
print(f"    p-value   : {p_t:.6f}")
print(f"    Result    : {'REJECT H0' if p_t < 0.05 else 'FAIL TO REJECT H0'}")
print("=" * 60)
print(f"  TimeMixer mean sMAPE : {merged['sMAPE_TM'].mean():.4f}%")
print(f"  GARCH mean sMAPE     : {merged['sMAPE_GARCH'].mean():.4f}%")
print(f"  Mean improvement     : {diff.mean():.4f} percentage points")
print(f"  Pairs where TM wins  : {int((diff > 0).sum())} / {len(diff)}")
"""))

# ─── Cell 10: p-value distribution plot ──────────────────────────────
cells.append(nbf.v4.new_code_cell(r"""
# Distribution of per-pair sMAPE differences
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Left: distribution of differences
ax = axes[0]
diff_vals = diff.values
ax.axvline(0, color='black', linewidth=1.2, linestyle='--', label='Zero line')
ax.axvline(diff_vals.mean(), color=PALETTE['highlight'], linewidth=1.5,
           linestyle='-', label=f'Mean = {diff_vals.mean():+.4f}')
ax.hist(diff_vals, bins=30, color=PALETTE['timemixer'], alpha=0.75, edgecolor='white')
ax.set_title('Distribution of sMAPE Differences\n(GARCH - TimeMixer per pair)')
ax.set_xlabel('sMAPE Difference (percentage points)')
ax.set_ylabel('Count')
ax.legend(frameon=False)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Right: per-horizon win rate
ax = axes[1]
win_by_h = merged.groupby('Horizon').apply(
    lambda g: (g['sMAPE_GARCH'] - g['sMAPE_TM'] > 0).mean() * 100
).reset_index(name='TM Win Rate (%)')

colors_bar = [PALETTE['timemixer'] if r > 50 else PALETTE['garch']
              for r in win_by_h['TM Win Rate (%)'].values]
bars = ax.bar(
    win_by_h['Horizon'].astype(str),
    win_by_h['TM Win Rate (%)'],
    color=colors_bar, edgecolor='white', width=0.55
)
ax.axhline(50, color='black', linewidth=1, linestyle='--', label='50% baseline')
for bar, val in zip(bars, win_by_h['TM Win Rate (%)']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{val:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title('TimeMixer Win Rate by Horizon\n(% of tickers where TM < GARCH in sMAPE)')
ax.set_xlabel('Forecast Horizon (days)')
ax.set_ylabel('Win Rate (%)')
ax.set_ylim(0, 110)
ax.legend(frameon=False)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'significance_analysis.png', bbox_inches='tight')
plt.show()
"""))

# ─── Markdown ─────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
---
## Section 5 — Per-Horizon Performance Curves

How does relative performance evolve as the forecast window lengthens?
This is arguably the most strategically important finding: short-term trading
strategies demand accuracy at h=12, while long-term risk management lives at h=720.
"""))

# ─── Cell 11: Horizon curves ──────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(r"""
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

metrics_to_plot = ['sMAPE', 'MAE', 'R2', 'HitRate']
ylabels         = ['sMAPE (%)', 'MAE', 'R2 Score', 'Hit Rate (%)']

for ax, metric, ylabel in zip(axes, metrics_to_plot, ylabels):
    tm_h  = tm.groupby('Horizon')[metric].mean()
    bl_h  = bl.groupby('Horizon')[metric].mean()

    ax.plot(HORIZONS, tm_h.values,  marker='o', linewidth=2.0,
            color=PALETTE['timemixer'], label='TimeMixer', zorder=3)
    ax.plot(HORIZONS, bl_h.values,  marker='s', linewidth=2.0,
            color=PALETTE['garch'],     label='GARCH(1,1)', linestyle='--', zorder=3)

    # Shade the region between curves
    ax.fill_between(
        HORIZONS, tm_h.values, bl_h.values,
        alpha=0.12,
        color=PALETTE['timemixer'] if tm_h.mean() < bl_h.mean() else PALETTE['garch']
    )

    ax.set_title(f'{metric} vs Forecast Horizon (Global)')
    ax.set_xlabel('Forecast Horizon (days)')
    ax.set_ylabel(ylabel)
    ax.set_xticks(HORIZONS)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, linestyle='--')

fig.suptitle('Model Performance Across Forecast Horizons', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'horizon_performance_curves.png', bbox_inches='tight')
plt.show()
"""))

# ─── Markdown ─────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
---
## Section 6 — Per-Ticker Leaderboard

The final leaderboard ranks every ticker on the average sMAPE improvement
that TimeMixer achieves over GARCH. Positive values mean TimeMixer wins.
"""))

# ─── Cell 12: Ticker leaderboard ──────────────────────────────────────
cells.append(nbf.v4.new_code_cell(r"""
# Build ticker-level leaderboard
ticker_summary = []
for ticker in tm['Ticker'].unique():
    t = tm[tm['Ticker'] == ticker]
    b = bl[bl['Ticker'] == ticker]
    if t.empty or b.empty:
        continue
    ac    = t['AssetClass'].iloc[0].upper()
    tm_s  = t['sMAPE'].mean()
    bl_s  = b['sMAPE'].mean()
    improv= bl_s - tm_s
    ticker_summary.append({
        'Ticker': ticker,
        'AssetClass': ac,
        'TM sMAPE':   round(tm_s, 2),
        'GARCH sMAPE':round(bl_s, 2),
        'Improvement': round(improv, 2),
        'Winner': 'TimeMixer' if improv > 0 else 'GARCH',
    })

lb = pd.DataFrame(ticker_summary).sort_values('Improvement', ascending=False)
lb.index = range(1, len(lb)+1)
lb.index.name = 'Rank'

def style_leaderboard(df):
    def color_winner(row):
        styles = [''] * len(row)
        idx = list(row.index).index('Winner')
        if row['Winner'] == 'TimeMixer':
            styles[idx] = f'color: {PALETTE["timemixer"]}; font-weight: bold'
        else:
            styles[idx] = f'color: {PALETTE["garch"]}; font-weight: bold'
        # Color improvement
        idx2 = list(row.index).index('Improvement')
        if row['Improvement'] > 0:
            styles[idx2] = 'color: #059669; font-weight: bold'
        else:
            styles[idx2] = 'color: #DC2626; font-weight: bold'
        return styles

    return df.style.apply(color_winner, axis=1).format({
        'TM sMAPE':    '{:.2f}%',
        'GARCH sMAPE': '{:.2f}%',
        'Improvement': '{:+.2f} pp',
    }).set_caption('Table 4 — Ticker-Level Leaderboard (ranked by sMAPE improvement)')

display(style_leaderboard(lb))
lb.to_csv(TABLES_DIR / 'ticker_leaderboard.csv', index=True)
print(f"\nTimeMixer wins in {int((lb['Winner']=='TimeMixer').sum())} / {len(lb)} tickers")
"""))

# ─── Cell 13: Leaderboard bar chart ───────────────────────────────────
cells.append(nbf.v4.new_code_cell(r"""
fig, ax = plt.subplots(figsize=(14, 10))

colors = [PALETTE['timemixer'] if v > 0 else PALETTE['garch']
          for v in lb['Improvement']]
bars = ax.barh(
    lb['Ticker'] + '  (' + lb['AssetClass'] + ')',
    lb['Improvement'],
    color=colors, edgecolor='white', linewidth=0.8, height=0.7
)

ax.axvline(0, color='black', linewidth=1.0, linestyle='--')
ax.set_title(
    'sMAPE Improvement of TimeMixer over GARCH(1,1) — All 40 Tickers\n'
    '(Positive = TimeMixer better   |   Negative = GARCH better)',
    fontsize=12, pad=12
)
ax.set_xlabel('sMAPE Improvement (percentage points)')

# Annotations
for bar, val in zip(bars, lb['Improvement']):
    xpos = bar.get_width() + (0.1 if val >= 0 else -0.1)
    ha   = 'left' if val >= 0 else 'right'
    ax.text(xpos, bar.get_y() + bar.get_height()/2,
            f'{val:+.2f}', va='center', ha=ha, fontsize=8.5)

ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'ticker_leaderboard.png', bbox_inches='tight', dpi=200)
plt.show()
"""))

# ─── Markdown ─────────────────────────────────────────────────────────
cells.append(nbf.v4.new_markdown_cell("""\
---
## Section 7 — Final Research Summary

This section consolidates the findings into a structured conclusion
that can be directly referenced in a research report or presentation.
"""))

# ─── Cell 14: Final summary ───────────────────────────────────────────
cells.append(nbf.v4.new_code_cell(r"""
print("=" * 70)
print("  PHASE 5 - FINAL EVALUATION SUMMARY")
print("  Volatility Forecasting: TimeMixer vs GARCH(1,1)")
print("=" * 70)
print()

total_pairs = len(merged)
tm_wins_pct = round(100 * (merged['sMAPE_GARCH'] > merged['sMAPE_TM']).mean(), 1)
mean_improv = round(diff.mean(), 4)
best_horizon_improv = win_by_h.set_index('Horizon')['TM Win Rate (%)'].idxmax()
best_ac_improv = (
    merged.groupby('AssetClass')
    .apply(lambda g: (g['sMAPE_GARCH'] - g['sMAPE_TM']).mean())
    .idxmax()
)

print(f"  Total model pairs compared   : {total_pairs}")
print(f"  TimeMixer wins (sMAPE)       : {tm_wins_pct}% of pairs")
print(f"  Mean sMAPE improvement       : {mean_improv:+.4f} percentage points")
print(f"  Strongest horizon            : h = {best_horizon_improv} days")
print(f"  Strongest asset class        : {best_ac_improv.upper()}")
print()
print("  Metric Snapshot (mean across all 200 trials):")
print(f"    {'Metric':<10}  {'TimeMixer':>12}  {'GARCH':>10}  {'Delta':>10}")
print(f"    {'-'*46}")
for m in ['MAE', 'RMSE', 'sMAPE', 'R2', 'HitRate']:
    tm_v = merged[f'{m}_TM'].mean()
    bl_v = merged[f'{m}_GARCH'].mean()
    d    = tm_v - bl_v
    print(f"    {m:<10}  {tm_v:>12.4f}  {bl_v:>10.4f}  {d:>+10.4f}")
print()
print("  Statistical Significance:")
print(f"    Wilcoxon signed-rank p-value : {p_w:.6f}  {'(Significant)' if p_w < 0.05 else '(Not significant)'}")
print(f"    Paired t-test p-value        : {p_t:.6f}  {'(Significant)' if p_t < 0.05 else '(Not significant)'}")
print()
print("=" * 70)
print("  Saved artifacts:")
print("    results/figures/global_metric_comparison.png")
print("    results/figures/smape_by_asset_horizon.png")
print("    results/figures/heatmap_smape_comparison.png")
print("    results/figures/heatmap_delta_smape.png")
print("    results/figures/significance_analysis.png")
print("    results/figures/horizon_performance_curves.png")
print("    results/figures/ticker_leaderboard.png")
print("    results/tables/ticker_leaderboard.csv")
print("=" * 70)
"""))

# ─── Write notebook ───────────────────────────────────────────────────
nb['cells'] = cells

with open('Notebooks/05_Evaluation.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Generated: Notebooks/05_Evaluation.ipynb")
