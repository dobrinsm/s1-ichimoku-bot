"""
Walk-Forward Validation + Multi-Asset Test for Cantonese Cat Strategies
=======================================================================
1. Walk-forward S2 (BB Squeeze), S1 (Ichimoku), S4 (Fib Breakout) on XLM/USD
2. Test S2 on SOL, ADA, DOGE, AVAX, LINK
"""

import os
os.environ.setdefault("LSE_API_KEY", os.environ.get("LSE_API_KEY", ""))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lse import LSE
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

client = LSE()

# ============================================================
# DATA FETCHING
# ============================================================

def fetch_candles(symbol, timeframe, start_date, end_date):
    all_candles = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    chunk_days = 60 if timeframe in ["1h", "4h"] else 365
    
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        try:
            candles = client.candles(symbol, timeframe,
                                      start=current.strftime("%Y-%m-%d"),
                                      end=chunk_end.strftime("%Y-%m-%d"))
            all_candles.extend(candles)
        except Exception as e:
            print(f"  ERROR {symbol} {timeframe} {current}: {e}")
        current = chunk_end
    
    if not all_candles:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_candles)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df[['open', 'high', 'low', 'close', 'volume']].drop_duplicates().sort_index()
    return df

# ============================================================
# INDICATORS
# ============================================================

def ichimoku(df, tenkan=9, kijun=26, senkou_b=52):
    high, low = df['high'], df['low']
    t = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    k = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    sa = ((t + k) / 2).shift(kijun)
    sb = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    return pd.DataFrame({'tenkan': t, 'kijun': k, 'senkou_a': sa, 'senkou_b': sb}, index=df.index)

def bollinger_bands(df, period=20, std_mult=2):
    sma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    return pd.DataFrame({
        'bb_mid': sma,
        'bb_upper': sma + std_mult * std,
        'bb_lower': sma - std_mult * std,
        'bb_bw': (sma + std_mult * std - (sma - std_mult * std)) / sma
    }, index=df.index)

# ============================================================
# STRATEGIES
# ============================================================

def strategy_s2_bb_squeeze(df, bb_period=20, bb_std=2, squeeze_lookback=50, squeeze_pctile=20):
    """BB Squeeze Breakout"""
    bb = bollinger_bands(df, bb_period, bb_std)
    pos = pd.Series(0.0, index=df.index)
    
    bw_pctile = bb['bb_bw'].rolling(squeeze_lookback).rank(pct=True)
    in_squeeze = bw_pctile < (squeeze_pctile / 100)
    was_squeeze = in_squeeze.rolling(10).max() > 0
    breakout = (df['close'] > bb['bb_upper']) & was_squeeze
    
    pos[breakout] = 1.0
    pos[(df['close'] > bb['bb_mid']) & (pos.shift(1) == 1)] = 1.0
    pos[df['close'] < bb['bb_mid']] = 0.0
    
    return pos.shift(1).fillna(0)

def strategy_s1_ichimoku_tenkan_ride(df):
    """Ichimoku Tenkan Ride"""
    ichi = ichimoku(df)
    pos = pd.Series(0.0, index=df.index)
    
    bullish = (df['close'] > ichi['tenkan']) & (ichi['tenkan'] > ichi['kijun'])
    pos[bullish] = 1.0
    above_cloud = df['close'] > ichi['senkou_a']
    pos[~above_cloud] = 0.0
    
    return pos.shift(1).fillna(0)

def strategy_s4_fib_breakout(df, lookback=100, entry_fib=0.786, exit_fib=0.618):
    """Fibonacci 0.786 Breakout"""
    pos = pd.Series(0.0, index=df.index)
    
    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i]
        swing_high = window['high'].max()
        swing_low = window['low'].min()
        fib_entry = swing_low + entry_fib * (swing_high - swing_low)
        fib_exit = swing_low + exit_fib * (swing_high - swing_low)
        
        if df['close'].iloc[i] > fib_entry:
            pos.iloc[i] = 1.0
        elif df['close'].iloc[i] < fib_exit:
            pos.iloc[i] = 0.0
        else:
            pos.iloc[i] = pos.iloc[i-1]
    
    return pos.shift(1).fillna(0)

# ============================================================
# BACKTEST ENGINE
# ============================================================

def compute_metrics(returns_series, ann_factor=365):
    clean = returns_series.dropna()
    if len(clean) == 0:
        return {}
    total_ret = (1 + clean).prod() - 1
    pa_ret = (1 + total_ret) ** (ann_factor / len(clean)) - 1 if total_ret > -1 else -1
    vol = clean.std() * np.sqrt(ann_factor)
    sharpe = pa_ret / vol if vol > 0 else 0
    downside = clean[clean < 0]
    dvol = downside.std() * np.sqrt(ann_factor) if len(downside) > 0 else 0
    sortino = pa_ret / dvol if dvol > 0 else 0
    cum = (1 + clean).cumprod()
    dd = (cum / cum.cummax() - 1)
    max_dd = dd.min()
    calmar = pa_ret / abs(max_dd) if max_dd < 0 else 0
    return {
        'total_ret': total_ret * 100, 'pa_ret': pa_ret * 100,
        'vol': vol * 100, 'sharpe': sharpe, 'sortino': sortino,
        'max_dd': max_dd * 100, 'calmar': calmar,
    }

def backtest(df, positions, cost_bps=0.0005, ann_factor=365):
    ret = df['close'].pct_change()
    trades = positions.diff().abs()
    strat_ret = positions * ret - trades * cost_bps
    n_trades = int((trades > 0).sum())
    time_in = (positions > 0).sum() / len(positions) * 100
    metrics = compute_metrics(strat_ret, ann_factor)
    metrics['n_trades'] = n_trades
    metrics['time_in'] = time_in
    equity = (1 + strat_ret).cumprod()
    return metrics, equity, strat_ret

# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

def walk_forward(df, strategy_fn, strategy_name, is_months=12, oos_months=6,
                 ann_factor=365, cost_bps=0.0005, strategy_kwargs=None):
    """
    Walk-forward: optimize nothing (fixed params), just measure OOS performance.
    IS window = is_months, OOS window = oos_months, rolling.
    """
    if strategy_kwargs is None:
        strategy_kwargs = {}
    
    total_days = (df.index[-1] - df.index[0]).days
    total_months = total_days / 30.44
    is_days = is_months * 30
    oos_days = oos_months * 30
    
    results = []
    start_idx = 0
    
    while True:
        is_start_idx = start_idx
        is_end_idx = start_idx + int(is_days)
        oos_start_idx = is_end_idx
        oos_end_idx = oos_start_idx + int(oos_days)
        
        if oos_end_idx >= len(df):
            break
        
        is_data = df.iloc[is_start_idx:is_end_idx]
        oos_data = df.iloc[oos_start_idx:oos_end_idx]
        
        if len(is_data) < 100 or len(oos_data) < 50:
            break
        
        # Run strategy on IS (for reference) and OOS (for validation)
        is_pos = strategy_fn(is_data, **strategy_kwargs)
        oos_pos = strategy_fn(oos_data, **strategy_kwargs)
        
        is_m, _, _ = backtest(is_data, is_pos, cost_bps, ann_factor)
        oos_m, _, oos_ret = backtest(oos_data, oos_pos, cost_bps, ann_factor)
        
        results.append({
            'window': len(results),
            'is_start': is_data.index[0],
            'is_end': is_data.index[-1],
            'oos_start': oos_data.index[0],
            'oos_end': oos_data.index[-1],
            'is_sharpe': is_m.get('sharpe', 0),
            'oos_sharpe': oos_m.get('sharpe', 0),
            'oos_ret': oos_m.get('total_ret', 0),
            'oos_maxdd': oos_m.get('max_dd', 0),
            'oos_trades': oos_m.get('n_trades', 0),
            'oos_ret_series': oos_ret,
        })
        
        start_idx = oos_start_idx  # roll forward
    
    return results

def walk_forward_summary(results, strategy_name):
    if not results:
        print(f"  {strategy_name}: No valid windows")
        return None
    
    oos_sharpes = [r['oos_sharpe'] for r in results]
    oos_rets = [r['oos_ret'] for r in results]
    oos_dds = [r['oos_maxdd'] for r in results]
    oos_trades = [r['oos_trades'] for r in results]
    
    # Compound OOS returns
    compound = 1.0
    for r in results:
        ret = r['oos_ret'] / 100
        compound *= (1 + ret)
    compound_ret = (compound - 1) * 100
    
    positive_windows = sum(1 for r in oos_rets if r > 0)
    
    summary = {
        'n_windows': len(results),
        'avg_oos_sharpe': np.mean(oos_sharpes),
        'med_oos_sharpe': np.median(oos_sharpes),
        'avg_oos_ret': np.mean(oos_rets),
        'compound_oos_ret': compound_ret,
        'avg_oos_maxdd': np.mean(oos_dds),
        'avg_oos_trades': np.mean(oos_trades),
        'positive_windows': positive_windows,
        'win_rate': positive_windows / len(results) * 100,
        'sharpe_gt_0.5': sum(1 for s in oos_sharpes if s > 0.5),
    }
    
    return summary

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("WALK-FORWARD VALIDATION — Cantonese Cat Strategies on XLM/USD")
    print("=" * 80)
    
    # Fetch XLM daily data (need enough history for walk-forward)
    print("\nFetching XLM/USD daily data (full history)...")
    xlm = fetch_candles("XLM/USD", "1d", "2019-01-01", "2026-08-12")
    print(f"XLM/USD 1d: {len(xlm)} candles, {xlm.index[0]} → {xlm.index[-1]}")
    
    # Also fetch with 1h for more granular walk-forward
    print("Fetching XLM/USD 4h data...")
    xlm_4h = fetch_candles("XLM/USD", "4h", "2020-01-01", "2026-08-12")
    print(f"XLM/USD 4h: {len(xlm_4h)} candles, {xlm_4h.index[0]} → {xlm_4h.index[-1]}")
    
    # ---- WALK-FORWARD ON DAILY ----
    print("\n" + "=" * 80)
    print("WALK-FORWARD — DAILY (12mo IS → 6mo OOS, rolling)")
    print("=" * 80)
    
    strategies_wf = {
        'S2: BB Squeeze Breakout': (strategy_s2_bb_squeeze, {}),
        'S1: Ichimoku Tenkan Ride': (strategy_s1_ichimoku_tenkan_ride, {}),
        'S4: Fib 0.786 Breakout': (strategy_s4_fib_breakout, {}),
    }
    
    all_wf_results = {}
    
    for name, (fn, kwargs) in strategies_wf.items():
        print(f"\n--- {name} ---")
        results = walk_forward(xlm, fn, name, is_months=12, oos_months=6,
                              ann_factor=365, strategy_kwargs=kwargs)
        
        # Print window details
        print(f"  {'W':<4} {'OOS Start':>12} {'OOS End':>12} {'IS Shp':>8} {'OOS Shp':>8} {'OOS Ret%':>10} {'OOS DD%':>10} {'Trades':>7}")
        print("  " + "-" * 75)
        for r in results:
            print(f"  W{r['window']:<3} {r['oos_start'].strftime('%Y-%m-%d'):>12} {r['oos_end'].strftime('%Y-%m-%d'):>12} {r['is_sharpe']:>8.2f} {r['oos_sharpe']:>8.2f} {r['oos_ret']:>10.1f} {r['oos_maxdd']:>10.1f} {r['oos_trades']:>7}")
        
        summary = walk_forward_summary(results, name)
        if summary:
            all_wf_results[name] = summary
            print(f"\n  SUMMARY: {summary['n_windows']} windows")
            print(f"    Avg OOS Sharpe:    {summary['avg_oos_sharpe']:.2f}")
            print(f"    Med OOS Sharpe:    {summary['med_oos_sharpe']:.2f}")
            print(f"    Avg OOS Return:    {summary['avg_oos_ret']:.1f}%")
            print(f"    Compound OOS Ret:  {summary['compound_oos_ret']:.1f}%")
            print(f"    Avg OOS MaxDD:     {summary['avg_oos_maxdd']:.1f}%")
            print(f"    Avg OOS Trades:    {summary['avg_oos_trades']:.1f}")
            print(f"    Win rate:          {summary['win_rate']:.0f}% ({summary['positive_windows']}/{summary['n_windows']})")
            print(f"    Sharpe > 0.5:      {summary['sharpe_gt_0.5']}/{summary['n_windows']}")
    
    # ---- WALK-FORWARD ON 4H (shorter windows for more data points) ----
    print("\n" + "=" * 80)
    print("WALK-FORWARD — 4H (6mo IS → 3mo OOS, rolling)")
    print("=" * 80)
    
    for name, (fn, kwargs) in strategies_wf.items():
        print(f"\n--- {name} (4H) ---")
        ann_4h = 365 * 6  # 4h candles: 6 per day
        results = walk_forward(xlm_4h, fn, name, is_months=6, oos_months=3,
                              ann_factor=ann_4h, strategy_kwargs=kwargs)
        
        print(f"  {'W':<4} {'OOS Start':>12} {'OOS End':>12} {'IS Shp':>8} {'OOS Shp':>8} {'OOS Ret%':>10} {'OOS DD%':>10} {'Trades':>7}")
        print("  " + "-" * 75)
        for r in results:
            print(f"  W{r['window']:<3} {r['oos_start'].strftime('%Y-%m-%d'):>12} {r['oos_end'].strftime('%Y-%m-%d'):>12} {r['is_sharpe']:>8.2f} {r['oos_sharpe']:>8.2f} {r['oos_ret']:>10.1f} {r['oos_maxdd']:>10.1f} {r['oos_trades']:>7}")
        
        summary = walk_forward_summary(results, name)
        if summary:
            key = f"{name} (4H)"
            all_wf_results[key] = summary
            print(f"\n  SUMMARY: {summary['n_windows']} windows")
            print(f"    Avg OOS Sharpe:    {summary['avg_oos_sharpe']:.2f}")
            print(f"    Med OOS Sharpe:    {summary['med_oos_sharpe']:.2f}")
            print(f"    Compound OOS Ret:  {summary['compound_oos_ret']:.1f}%")
            print(f"    Win rate:          {summary['win_rate']:.0f}% ({summary['positive_windows']}/{summary['n_windows']})")
    
    # ============================================================
    # PART 2: MULTI-ASSET S2 BB SQUEEZE TEST
    # ============================================================
    
    print("\n" + "=" * 80)
    print("MULTI-ASSET TEST — S2 BB Squeeze Breakout")
    print("=" * 80)
    
    alts = {
        'XLM/USD': '2019-01-01',
        'SOL/USD': '2020-08-11',
        'ADA/USD': '2018-04-17',
        'DOGE/USD': '2019-07-05',
        'AVAX/USD': '2020-09-22',
        'LINK/USD': '2019-02-01',
    }
    
    multi_results = {}
    
    for symbol, start_date in alts.items():
        print(f"\n--- {symbol} ---")
        df = fetch_candles(symbol, "1d", start_date, "2026-08-12")
        if len(df) < 200:
            print(f"  Not enough data: {len(df)} candles")
            continue
        
        print(f"  Data: {len(df)} candles, {df.index[0]} → {df.index[-1]}")
        
        # Full-sample backtest
        pos = strategy_s2_bb_squeeze(df)
        m, eq, _ = backtest(df, pos)
        
        # Walk-forward
        wf = walk_forward(df, strategy_s2_bb_squeeze, 'S2', is_months=12, oos_months=6)
        wf_sum = walk_forward_summary(wf, 'S2')
        
        print(f"  Full-sample: Ret={m['total_ret']:.1f}% | Sharpe={m['sharpe']:.2f} | MaxDD={m['max_dd']:.1f}% | Trades={m['n_trades']} | InMkt={m['time_in']:.1f}%")
        if wf_sum:
            print(f"  Walk-fwd:   AvgOOS_Shp={wf_sum['avg_oos_sharpe']:.2f} | MedOOS_Shp={wf_sum['med_oos_sharpe']:.2f} | CompoundOOS={wf_sum['compound_oos_ret']:.1f}% | WinRate={wf_sum['win_rate']:.0f}% | Sharpe>0.5: {wf_sum['sharpe_gt_0.5']}/{wf_sum['n_windows']}")
        
        multi_results[symbol] = {
            'full': m,
            'wf': wf_sum,
            'equity': eq,
        }
    
    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY — Walk-Forward Validation (XLM/USD Daily)")
    print("=" * 80)
    
    print(f"\n{'Strategy':<30} {'AvgOOS_Sh':>10} {'MedOOS_Sh':>10} {'CompOOS%':>10} {'AvgOOS_DD':>10} {'WinRate':>10} {'Shp>0.5':>10}")
    print("-" * 95)
    for name, s in all_wf_results.items():
        if '4H' not in name:  # Daily only for main summary
            print(f"{name:<30} {s['avg_oos_sharpe']:>10.2f} {s['med_oos_sharpe']:>10.2f} {s['compound_oos_ret']:>10.1f} {s['avg_oos_maxdd']:>10.1f} {s['win_rate']:>9.0f}% {s['sharpe_gt_0.5']:>5}/{s['n_windows']}")
    
    print(f"\n{'Strategy':<30} {'AvgOOS_Sh':>10} {'MedOOS_Sh':>10} {'CompOOS%':>10} {'WinRate':>10} {'Shp>0.5':>10}")
    print("-" * 85)
    for name, s in all_wf_results.items():
        if '4H' in name:
            print(f"{name:<30} {s['avg_oos_sharpe']:>10.2f} {s['med_oos_sharpe']:>10.2f} {s['compound_oos_ret']:>10.1f} {s['win_rate']:>9.0f}% {s['sharpe_gt_0.5']:>5}/{s['n_windows']}")
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY — S2 BB Squeeze Multi-Asset")
    print("=" * 80)
    
    print(f"\n{'Asset':<12} {'FS_Ret%':>8} {'FS_Shp':>8} {'FS_DD%':>8} {'FS_Tr':>6} {'WF_AvgSh':>10} {'WF_MedSh':>10} {'WF_Comp%':>10} {'WF_WR':>7} {'WF_>0.5':>8}")
    print("-" * 100)
    for symbol, r in multi_results.items():
        f = r['full']
        w = r['wf']
        if w:
            print(f"{symbol:<12} {f['total_ret']:>8.1f} {f['sharpe']:>8.2f} {f['max_dd']:>8.1f} {f['n_trades']:>6} {w['avg_oos_sharpe']:>10.2f} {w['med_oos_sharpe']:>10.2f} {w['compound_oos_ret']:>10.1f} {w['win_rate']:>6.0f}% {w['sharpe_gt_0.5']:>4}/{w['n_windows']}")
        else:
            print(f"{symbol:<12} {f['total_ret']:>8.1f} {f['sharpe']:>8.2f} {f['max_dd']:>8.1f} {f['n_trades']:>6} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>7} {'N/A':>8}")
    
    # ============================================================
    # PLOTTING
    # ============================================================
    
    print("\n" + "=" * 80)
    print("PLOTTING...")
    print("=" * 80)
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 16), gridspec_kw={'height_ratios': [2, 1, 2]})
    
    # Plot 1: XLM walk-forward OOS Sharpe by window
    ax = axes[0]
    for name, (fn, kwargs) in strategies_wf.items():
        results = walk_forward(xlm, fn, name, is_months=12, oos_months=6, strategy_kwargs=kwargs)
        if results:
            windows = [f"W{r['window']}" for r in results]
            sharpes = [r['oos_sharpe'] for r in results]
            dates = [r['oos_start'].strftime('%Y-%m') for r in results]
            ax.bar([f"{w}\n{d}" for w, d in zip(windows, dates)], sharpes, 
                   label=name, alpha=0.7)
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axhline(0.5, color='green', linewidth=0.5, linestyle='--', label='Sharpe=0.5 threshold')
    ax.set_title('Walk-Forward OOS Sharpe by Window — XLM/USD Daily', fontsize=13)
    ax.set_ylabel('OOS Sharpe')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Walk-forward compound returns
    ax = axes[1]
    for name, (fn, kwargs) in strategies_wf.items():
        results = walk_forward(xlm, fn, name, is_months=12, oos_months=6, strategy_kwargs=kwargs)
        if results:
            windows = [f"W{r['window']}" for r in results]
            rets = [r['oos_ret'] for r in results]
            colors = ['green' if r > 0 else 'red' for r in rets]
            ax.bar(windows, rets, color=colors, alpha=0.6, label=name)
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title('Walk-Forward OOS Returns by Window — XLM/USD Daily', fontsize=12)
    ax.set_ylabel('OOS Return %')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Multi-asset equity curves
    ax = axes[2]
    for symbol, r in multi_results.items():
        ax.plot(r['equity'].index, r['equity'].values, label=f"{symbol} (Shp={r['full']['sharpe']:.2f})", linewidth=1.5)
    
    ax.set_yscale('log')
    ax.set_title('S2 BB Squeeze Breakout — Multi-Asset Equity Curves', fontsize=13)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Cumulative Return (log scale)')
    
    plt.tight_layout()
    plt.savefig('/tmp/cantonese_cat_walkforward.png', dpi=150, bbox_inches='tight')
    print("Chart saved: /tmp/cantonese_cat_walkforward.png")
    
    print("\nDone.")
