"""Quick smoke test of all src modules."""
import sys
sys.path.insert(0, '/home/thota23/Volatility-Forecasting-in-Global-Financial-Markets-Using-TimeMixer')

from src.utils.config import cfg
print(f'[1/5] Config OK  — estimator={cfg.data.volatility.primary_estimator}, horizons={cfg.timemixer.horizons}')

from src.utils.logger import get_logger
log = get_logger('smoke_test')
print('[2/5] Logger OK')

from src.data.loader import DataLoader
loader = DataLoader()
stock_df = loader.load('stock')
print(f'[3/5] DataLoader OK — stock {stock_df.shape}')

from src.data.preprocessing import VolatilityProcessor
proc = VolatilityProcessor()
vol_df = proc.compute_volatility(stock_df.head(3000), estimator='yang_zhang', add_all_estimators=True)
print(f'[4/5] VolatilityProcessor OK — {vol_df.shape}, cols={list(vol_df.columns)}')

from src.models.evaluation import Evaluator
import numpy as np
ev = Evaluator()
y = np.random.rand(100) * 0.3
m = ev.compute_metrics(y, y + np.random.randn(100)*0.01, model_name='SmokeTest')
print(f'[5/5] Evaluator OK — MAE={m["MAE"]:.5f}')

print()
print('ALL MODULES VERIFIED OK')
