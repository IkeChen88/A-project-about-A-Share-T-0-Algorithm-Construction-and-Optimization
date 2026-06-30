"""
Price prediction visualization using saved model predictions.
Logic: binary signal with optimal threshold → cumulative price curve.
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from sklearn.metrics import f1_score
import warnings; warnings.filterwarnings('ignore')

# ============================================================
# 1. Load predictions & find optimal threshold
# ============================================================
PREDS_PATH = 'results/test_predictions.csv'
PRICE_PATH = 'output/processed_data.parquet'
OUTPUT_DIR = Path('visualization_results')
OUTPUT_DIR.mkdir(exist_ok=True)

preds = pd.read_csv(PREDS_PATH)
y_true = preds['y_true'].values
y_prob = preds['y_pred'].values

# Fixed threshold = 0.04
best_t = 0.04

# Binary signal
y_class = (y_prob > best_t).astype(int)
acc = (y_true == y_class).mean()
best_f1 = f1_score(y_true, y_class)
print(f'Threshold: {best_t:.2f}')
print(f'True up: {y_true.mean():.1%}, Pred up: {y_class.mean():.1%}, Acc: {acc:.2%}, F1: {best_f1:.4f}')

# ============================================================
# 2. Build predicted prices
# ============================================================
df = pd.read_parquet(PRICE_PATH)
close_all = df['close']
n = len(preds)
true_prices = close_all.values[-n:]
dates = close_all.index[-n:]

# Actual returns
actual_returns = np.diff(true_prices, prepend=true_prices[0]) / np.where(
    np.roll(true_prices, 1) != 0, np.roll(true_prices, 1), 1.0)

# Strategy: long if signal=1, cash if signal=0
strategy_returns = actual_returns * y_class
predicted_prices = true_prices[0] * np.cumprod(1 + strategy_returns)

# ============================================================
# 3. Filter to last week
# ============================================================
mask = dates >= dates[-1] - pd.Timedelta(days=7)
d = dates[mask]
t = true_prices[mask]
p = predicted_prices[mask]

# ============================================================
# 4. Price difference scatter plot
# ============================================================
price_diff = p - t
price_diff_pct = (p - t) / t * 100

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# --- Subplot 1: Price curve ---
ax = axes[0, 0]
ax.plot(d, t, linewidth=1.2, color='#1a73e8', label='True Price')
ax.plot(d, p, linewidth=1.0, color='#e84c3d', linestyle='--',
        label=f'Predicted (thresh=0.04, F1={best_f1:.3f})')
ax.fill_between(d, t, alpha=0.06, color='#1a73e8')
ax.set_title(f'SMIC (688981) — Price Prediction (Last Week)', fontsize=13, fontweight='bold')
ax.set_ylabel('Price')
ax.legend(fontsize=9, loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
ax.grid(True, alpha=0.3)

# --- Subplot 2: Price difference over time ---
ax = axes[0, 1]
colors = ['#e84c3d' if v < 0 else '#2e7d32' for v in price_diff]
ax.bar(range(len(d)), price_diff, color=colors, alpha=0.7, width=0.8)
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_title('Price Difference (Predicted - True)', fontsize=13, fontweight='bold')
ax.set_xlabel('Time Index')
ax.set_ylabel('Price Difference')
ax.grid(True, alpha=0.3, axis='y')

# --- Subplot 3: Scatter: True vs Predicted price ---
ax = axes[1, 0]
ax.scatter(t, p, alpha=0.5, s=15, c=price_diff, cmap='RdYlGn', edgecolors='none')
min_v, max_v = min(t.min(), p.min()), max(t.max(), p.max())
ax.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=1, label='Perfect Prediction')
ax.set_xlabel('True Price')
ax.set_ylabel('Predicted Price')
ax.set_title('True vs Predicted Price Scatter', fontsize=13, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
# Add correlation text
corr = np.corrcoef(t, p)[0, 1]
ax.text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=ax.transAxes,
        fontsize=11, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# --- Subplot 4: Difference distribution ---
ax = axes[1, 1]
ax.hist(price_diff_pct, bins=40, alpha=0.7, color='steelblue', edgecolor='white')
ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5)
ax.set_xlabel('Price Difference (%)')
ax.set_ylabel('Frequency')
ax.set_title('Price Difference Distribution', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
# Stats
mean_diff = price_diff_pct.mean()
std_diff = price_diff_pct.std()
ax.text(0.95, 0.95, f'Mean: {mean_diff:+.3f}%\nStd: {std_diff:.3f}%',
        transform=ax.transAxes, fontsize=10, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'price_curve.png', dpi=200, bbox_inches='tight')
plt.close()

# Save data
pd.DataFrame({
    'date': d, 'true_price': t, 'predicted_price': p,
    'price_diff': price_diff, 'price_diff_pct': price_diff_pct
}).to_csv(OUTPUT_DIR / 'price_comparison.csv', index=False)

print(f'\nPrice diff stats: mean={mean_diff:+.3f}%, std={std_diff:.3f}%')
print(f'Correlation true vs pred: {corr:.4f}')
print(f'Saved: {OUTPUT_DIR}/price_curve.png')
print(f'Saved: {OUTPUT_DIR}/price_comparison.csv')
