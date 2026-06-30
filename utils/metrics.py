"""
Key performance metrics & risk management for quantitative trading.
- Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio
- Dynamic position sizing
- Stop-loss / trailing stop mechanism
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict


# ============================================================
# Performance Metrics
# ============================================================

def sharpe_ratio(returns, risk_free=0.0, periods_per_year=252):
    """Annualized Sharpe ratio."""
    excess = returns - risk_free / periods_per_year
    if np.std(excess) < 1e-10:
        return 0.0
    return np.mean(excess) / np.std(excess) * np.sqrt(periods_per_year)


def sortino_ratio(returns, risk_free=0.0, periods_per_year=252, target_return=0.0):
    """Annualized Sortino ratio - uses only downside deviation."""
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < target_return]
    if len(downside) < 2 or np.std(downside) < 1e-10:
        return 0.0
    downside_std = np.std(downside, ddof=1)
    return np.mean(excess) / downside_std * np.sqrt(periods_per_year)


def max_drawdown(returns):
    """Maximum drawdown from peak. Returns (max_dd_pct, peak_idx, trough_idx)."""
    cum = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cum)
    drawdowns = (cum - running_max) / np.where(running_max > 0, running_max, 1.0)
    trough_idx = int(np.argmin(drawdowns))
    peak_idx = int(np.argmax(cum[:trough_idx + 1])) if trough_idx > 0 else 0
    return float(drawdowns[trough_idx]), peak_idx, trough_idx


def calmar_ratio(returns, periods_per_year=252):
    """Annualized Calmar ratio = annualized return / |max drawdown|."""
    ann_return = np.mean(returns) * periods_per_year
    mdd, _, _ = max_drawdown(returns)
    if abs(mdd) < 1e-10:
        return 0.0
    return ann_return / abs(mdd)


def win_rate(returns):
    """Percentage of periods with positive return."""
    return float(np.mean(returns > 0))


def profit_factor(returns):
    """Gross profit / gross loss. >1 means profitable."""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses < 1e-10:
        return float('inf')
    return float(gains / losses)


def compute_all_metrics(returns, risk_free=0.0, periods_per_year=252):
    """Compute all key metrics at once. Returns dict."""
    mdd, peak_i, trough_i = max_drawdown(returns)
    return {
        'Annualized Return (%)': np.mean(returns) * periods_per_year * 100,
        'Annualized Volatility (%)': np.std(returns, ddof=1) * np.sqrt(periods_per_year) * 100,
        'Sharpe Ratio': sharpe_ratio(returns, risk_free, periods_per_year),
        'Sortino Ratio': sortino_ratio(returns, risk_free, periods_per_year),
        'Max Drawdown (%)': mdd * 100,
        'Calmar Ratio': calmar_ratio(returns, periods_per_year),
        'Win Rate (%)': win_rate(returns) * 100,
        'Profit Factor': profit_factor(returns),
        'Total Return (%)': (np.cumprod(1 + returns)[-1] - 1) * 100,
    }


# ============================================================
# Dynamic Position Sizing
# ============================================================

def position_kelly(win_prob, win_loss_ratio, max_position=1.0):
    """Kelly criterion position sizing."""
    if win_loss_ratio <= 0:
        return 0.0
    kelly = win_prob - (1 - win_prob) / win_loss_ratio
    return np.clip(kelly, 0.0, max_position)


def position_probability_weighted(prob, threshold=0.5, max_position=1.0):
    """Map prediction probability to position size."""
    return np.clip(2 * (prob - threshold), -max_position, max_position)


def position_volatility_scaled(returns, lookback=20, target_vol=0.15, max_position=1.5):
    """Volatility-targeted position sizing."""
    n = len(returns)
    positions = np.ones(n)
    for i in range(lookback, n):
        recent_vol = np.std(returns[i - lookback:i], ddof=1)
        if recent_vol > 1e-10:
            positions[i] = min(target_vol / (recent_vol * np.sqrt(252)), max_position)
        else:
            positions[i] = max_position
    return positions


def position_combined(prob, returns_history, threshold=0.5, lookback=20, max_position=1.5):
    """Combined: probability-weighted * volatility-scaled."""
    base = position_probability_weighted(prob, threshold)
    if lookback < len(returns_history):
        recent_vol = np.std(returns_history[-lookback:], ddof=1)
        vol_scale = 0.15 / (recent_vol * np.sqrt(252) + 1e-10)
        vol_scale = np.clip(vol_scale, 0.3, 2.0)
        return np.clip(base * vol_scale, -max_position, max_position)
    return np.clip(base, -max_position, max_position)


# ============================================================
# Stop-Loss Mechanisms
# ============================================================

def apply_stop_loss(prices, signals, stop_loss_pct=0.05):
    """Fixed percentage stop-loss. Returns modified signals."""
    signals_mod = signals.copy().astype(float)
    entry_price = None
    in_position = False
    for i in range(len(prices)):
        if signals_mod[i] > 0 and not in_position:
            entry_price = prices[i]
            in_position = True
        elif in_position:
            if entry_price and prices[i] < entry_price * (1 - stop_loss_pct):
                signals_mod[i] = 0
                in_position = False
                entry_price = None
            elif signals_mod[i] == 0:
                in_position = False
                entry_price = None
    return signals_mod


def apply_trailing_stop(prices, signals, trail_pct=0.05):
    """Trailing stop-loss. Returns modified signals."""
    signals_mod = signals.copy().astype(float)
    in_position = False
    highest_since_entry = 0.0
    for i in range(len(prices)):
        if signals_mod[i] > 0:
            if not in_position:
                highest_since_entry = prices[i]
                in_position = True
            else:
                highest_since_entry = max(highest_since_entry, prices[i])
                if prices[i] < highest_since_entry * (1 - trail_pct):
                    signals_mod[i] = 0
                    in_position = False
        else:
            in_position = False
    return signals_mod


def apply_time_stop(signals, max_hold_bars=20):
    """Time-based stop: force exit after max_hold_bars periods."""
    signals_mod = signals.copy().astype(float)
    hold_count = 0
    for i in range(len(signals_mod)):
        if signals_mod[i] > 0:
            hold_count += 1
            if hold_count > max_hold_bars:
                signals_mod[i] = 0
                hold_count = 0
        else:
            hold_count = 0
    return signals_mod


# ============================================================
# Utility: apply signals to returns with risk management
# ============================================================

def strategy_returns_with_risk_mgmt(
    returns, signals, prices, stop_loss_pct=0.03, trailing_stop_pct=0.05,
    use_probability_weighting=True, probs=None, threshold=0.04,
):
    """Generate strategy returns with full risk management pipeline.

    Args:
        returns: actual asset returns
        signals: raw binary signals (0/1)
        prices: asset prices for stop-loss reference
        stop_loss_pct: fixed stop-loss threshold
        trailing_stop_pct: trailing stop threshold
        use_probability_weighting: use probability-weighted positions
        probs: model probability outputs (for position sizing)
        threshold: classification threshold

    Returns:
        dict with keys: raw, stop_loss, trailing_stop, combined
    """
    results = {}

    # Raw strategy returns (binary signal)
    results['raw'] = returns * signals

    # Fixed stop-loss
    sl_signals = apply_stop_loss(prices, signals, stop_loss_pct)
    results['stop_loss'] = returns * sl_signals

    # Trailing stop
    ts_signals = apply_trailing_stop(prices, signals, trailing_stop_pct)
    results['trailing_stop'] = returns * ts_signals

    # Combined: probability-weighted + trailing stop
    if use_probability_weighting and probs is not None:
        weighted_positions = np.array([
            position_probability_weighted(p, threshold) for p in probs
        ])
        weighted_positions[weighted_positions < 0] = 0
        combined_signals = apply_trailing_stop(prices, weighted_positions, trailing_stop_pct)
        results['combined'] = returns * combined_signals

    return results
