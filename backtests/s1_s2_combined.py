"""
S1+S2 Combined Strategy: Ichimoku Tenkan Ride + BB Squeeze Filter
=================================================================
Logic:
- S1 (Ichimoku Tenkan Ride) is the primary trend-following signal
- S2 (BB Squeeze) acts as a filter/overlay:
  - Only enter S1 longs when BB is NOT in a squeeze (we want expansion phase)
  - OR: Enter S1 longs when BB squeeze breaks out (squeeze + ICH confirm = high conviction)
  - Exit when S1 says exit OR S2 says exit (below mid BB)

We test 3 combination variants:
  A) S1 AND NOT squeeze: trade trend, but skip during tight consolidation
  B) S1 AND squeeze_breakout: only enter on squeeze breakout + ICH confirm (high conviction)
  C) S1 entry, S2 exit: S1 for entries, exit early if price < lower BB (tighter risk)
  D) S1 OR S2: union of both signals (more trades)
  E) S1 with S2 position sizing: full size normally, half size during squeeze
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
# BASE SIGNALS
# ============================================================

def s1_signal(df):
    """S1 Ichimoku Tenkan Ride — raw boolean signal (not shifted)."""
    ichi = ichimoku(df)
    bullish = (df['close'] > ichi['tenkan']) & (ichi['tenkan'] > ichi['kijun'])
    above_cloud = df['close'] > ichi['senkou_a']
    return bullish & above_cloud

def s2_squeeze_state(df, bb_period=20, bb_std=2, squeeze_lookback=50, squeeze_pctile=20):
    """Returns squeeze state: 'squeeze', 'expanding', or 'normal'."""
    bb = bollinger_bands(df, bb_period, bb_std)
    bw_pctile = bb['bb_bw'].rolling(squeeze_lookback).rank(pct=True)
    
    in_squeeze = bw_pctile < (squeeze_pctile / 100)
    was_squeeze = in_squeeze.rolling(10).max() > 0
    breakout = (df['close'] > bb['bb_upper']) & was_squeeze
    
    # 'expanding' = just broke out of squeeze
    # 'squeeze' = currently in squeeze
    # 'normal' = neither
    state = pd.Series('normal', index=df.index)
    state[in_squeeze] = 'squeeze'
    state[breakout] = 'expanding'
    return state, bb

def s2_signal(df):
    """S2 BB Squeeze Breakout — raw boolean signal."""
    state, bb = s2_squeeze_state(df)
    # Long when expanding (just broke out of squeeze)
    # Hold while above mid BB
    # Exit below mid BB
    expanding = state == 'expanding'
    above_mid = df['close'] > bb['bb_mid']
    
    # Build position from expanding signals, holding while above mid
    pos = pd.Series(False, index=df.index)
    pos[expanding] = True
    # Hold while above mid and was previously long
    for i in range(1, len(df)):
        if pos.iloc[i-1] and above_mid.iloc[i]:
            pos.iloc[i] = True
    return pos

# ============================================================
# COMBINED STRATEGIES
# ============================================================

def combo_A_s1_not_squeeze(df):
    """Variant A: S1 signal but skip when BB is in squeeze.
    Rationale: During squeeze, trend is unclear — wait for expansion."""
    s1 = s1_signal(df)
    state, _ = s2_squeeze_state(df)
    not_squeeze = state != 'squeeze'
    return (s1 & not_squeeze).astype(float)

def combo_B_s1_squeeze_breakout(df):
    """Variant B: Only enter when S1 bullish AND BB just broke out of squeeze.
    High conviction: trend + volatility expansion aligned."""
    s1 = s1_signal(df)
    state, _ = s2_squeeze_state(df)
    expanding = state == 'expanding'
    
    # Enter on expansion + S1
    entry = s1 & expanding
    
    # Hold while S1 stays bullish (don't need to stay in expansion)
    pos = pd.Series(0.0, index=df.index)
    pos[entry] = 1.0
    for i in range(1, len(df)):
        if pos.iloc[i-1] == 1.0 and s1.iloc[i]:
            pos.iloc[i] = 1.0
    return pos

def combo_C_s1_entry_s2_exit(df):
    """Variant C: S1 for entries, exit early if price < lower BB.
    Tighter risk management — cut losses faster than S1 alone."""
    s1 = s1_signal(df)
    _, bb = s2_squeeze_state(df)
    
    pos = pd.Series(0.0, index=df.index)
    
    for i in range(1, len(df)):
        # S1 says long
        if s1.iloc[i]:
            # But exit if below lower BB (fast stop)
            if df['close'].iloc[i] < bb['bb_lower'].iloc[i]:
                pos.iloc[i] = 0.0
            else:
                pos.iloc[i] = 1.0
        else:
            pos.iloc[i] = 0.0
    
    return pos

def combo_D_s1_or_s2(df):
    """Variant D: Union of S1 and S2 — more trades, more opportunities."""
    s1 = s1_signal(df)
    s2 = s2_signal(df)
    return (s1 | s2).astype(float)

def combo_E_s1_sized_by_s2(df):
    """Variant E: S1 full size normally, half size during squeeze.
    Graduated exposure — reduce risk during uncertainty."""
    s1 = s1_signal(df)
    state, _ = s2_squeeze_state(df)
    
    pos = pd.Series(0.0, index=df.index)
    pos[s1] = 1.0
    # Half size during squeeze
    squeeze_mask = state == 'squeeze'
    pos[s1 & squeeze_mask] = 0.5
    return pos

def combo_F_s1_and_s2_confirm(df):
    """Variant F: S1 entry only when S2 also bullish (above mid BB after squeeze).
    Both must agree — strictest filter."""
    s1 = s1_signal(df)
    s2 = s2_signal(df)
    return (s1 & s2).astype(float)

# ============================================================
# BACKTEST
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
# WALK-FORWARD
# ============================================================

def walk_forward(df, strategy_fn, is_months=12, oos_months=6,
                 ann_factor=365, cost_bps=0.0005):
    results = []
    start_idx = 0
    is_days = is_months * 30
    oos_days = oos_months * 30
    
    while True:
        is_end_idx = start_idx + int(is_days)
        oos_start = is_end_idx
        oos_end = oos_start + int(oos_days)
        
        if oos_end >= len(df):
            break
        
        is_data = df.iloc[start_idx:is_end_idx]
        oos_data = df.iloc[oos_start:oos_end]
        
        if len(is_data) < 100 or len(oos_data) < 50:
            break
        
        is_pos = strategy_fn(is_data)
        oos_pos = strategy_fn(oos_data)
        
        is_m, _, _ = backtest(is_data, is_pos, cost_bps, ann_factor)
        oos_m, _, oos_ret = backtest(oos_data, oos_pos, cost_bps, ann_factor)
        
        results.append({
            'window': len(results),
            'oos_start': oos_data.index[0],
            'oos_end': oos_data.index[-1],
            'is_sharpe': is_m.get('sharpe', 0),
            'oos_sharpe': oos_m.get('sharpe', 0),
            'oos_ret': oos_m.get('total_ret', 0),
            'oos_maxdd': oos_m.get('max_dd', 0),
            'oos_trades': oos_m.get('n_trades', 0),
        })
        
        start_idx = oos_start
    
    return results

def wf_summary(results):
    if not results:
        return None
    sharpes = [r['oos_sharpe'] for r in results]
    rets = [r['oos_ret'] for r in results]
    dds = [r['oos_maxdd'] for r in results]
    
    compound = 1.0
    for r in rets:
        compound *= (1 + r / 100)
    
    positive = sum(1 for r in rets if r > 0)
    
    return {
        'n_windows': len(results),
        'avg_sharpe': np.mean(sharpes),
        'med_sharpe': np.median(sharpes),
        'compound_ret': (compound - 1) * 100,
        'avg_maxdd': np.mean(dds),
        'win_rate': positive / len(results) * 100,
        'sharpe_gt_05': sum(1 for s in sharpes if s > 0.5),
    }

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("S1+S2 COMBINED STRATEGY BACKTEST")
    print("=" * 80)
    
    # Fetch data
    print("\nFetching XLM/USD daily (2019-2026)...")
    xlm = fetch_candles("XLM/USD", "1d", "2019-01-01", "2026-08-12")
    print(f"  {len(xlm)} candles, {xlm.index[0]} → {xlm.index[-1]}")
    
    # Baselines
    s1_pos = s1_signal(xlm).astype(float)
    s2_pos_raw = s2_signal(xlm)
    s2_pos = s2_pos_raw.astype(float)
    
    baselines = {
        'S1 alone (Ichimoku)': backtest(xlm, s1_pos),
        'S2 alone (BB Squeeze)': backtest(xlm, s2_pos),
    }
    
    # Combinations
    combos = {
        'A: S1 NOT squeeze': (combo_A_s1_not_squeeze, backtest(xlm, combo_A_s1_not_squeeze(xlm))),
        'B: S1 + squeeze breakout': (combo_B_s1_squeeze_breakout, backtest(xlm, combo_B_s1_squeeze_breakout(xlm))),
        'C: S1 entry, S2 exit': (combo_C_s1_entry_s2_exit, backtest(xlm, combo_C_s1_entry_s2_exit(xlm))),
        'D: S1 OR S2': (combo_D_s1_or_s2, backtest(xlm, combo_D_s1_or_s2(xlm))),
        'E: S1 sized by S2': (combo_E_s1_sized_by_s2, backtest(xlm, combo_E_s1_sized_by_s2(xlm))),
        'F: S1 AND S2': (combo_F_s1_and_s2_confirm, backtest(xlm, combo_F_s1_and_s2_confirm(xlm))),
    }
    
    # Print full-sample results
    print("\n" + "=" * 80)
    print("FULL-SAMPLE RESULTS — XLM/USD Daily (2019-2026)")
    print("=" * 80)
    
    print(f"\n{'Strategy':<30} {'TotRet%':>8} {'PaRet%':>8} {'Vol%':>8} {'Sharpe':>7} {'Sortino':>7} {'MaxDD%':>8} {'Calmar':>7} {'Trades':>7} {'InMkt%':>7}")
    print("-" * 110)
    
    all_results = {}
    
    for name, (m, eq, _) in baselines.items():
        all_results[name] = (m, eq)
        print(f"{name:<30} {m['total_ret']:>8.1f} {m['pa_ret']:>8.1f} {m['vol']:>8.1f} {m['sharpe']:>7.2f} {m['sortino']:>7.2f} {m['max_dd']:>8.1f} {m['calmar']:>7.2f} {m['n_trades']:>7} {m['time_in']:>7.1f}")
    
    for name, (fn, (m, eq, _)) in combos.items():
        all_results[name] = (m, eq)
        print(f"{name:<30} {m['total_ret']:>8.1f} {m['pa_ret']:>8.1f} {m['vol']:>8.1f} {m['sharpe']:>7.2f} {m['sortino']:>7.2f} {m['max_dd']:>8.1f} {m['calmar']:>7.2f} {m['n_trades']:>7} {m['time_in']:>7.1f}")
    
    # Walk-forward on top combos + baselines
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION (12mo IS → 6mo OOS)")
    print("=" * 80)
    
    wf_strategies = {
        'S1 alone (Ichimoku)': s1_pos.to_frame('pos')['pos'].apply(lambda x: float(x)).pipe(lambda s: s),
        'S2 alone (BB Squeeze)': s2_pos,
    }
    
    # Need to pass functions, not pre-computed positions
    wf_fns = {
        'S1 alone (Ichimoku)': lambda df: s1_signal(df).astype(float),
        'S2 alone (BB Squeeze)': lambda df: s2_signal(df).astype(float),
        'A: S1 NOT squeeze': combo_A_s1_not_squeeze,
        'B: S1 + squeeze breakout': combo_B_s1_squeeze_breakout,
        'C: S1 entry, S2 exit': combo_C_s1_entry_s2_exit,
        'D: S1 OR S2': combo_D_s1_or_s2,
        'E: S1 sized by S2': combo_E_s1_sized_by_s2,
        'F: S1 AND S2': combo_F_s1_and_s2_confirm,
    }
    
    print(f"\n{'Strategy':<30} {'AvgShp':>8} {'MedShp':>8} {'CompOOS%':>10} {'AvgDD%':>8} {'WinRate':>8} {'Shp>0.5':>8}")
    print("-" * 90)
    
    wf_all = {}
    
    for name, fn in wf_fns.items():
        results = walk_forward(xlm, fn)
        s = wf_summary(results)
        if s:
            wf_all[name] = (s, results)
            print(f"{name:<30} {s['avg_sharpe']:>8.2f} {s['med_sharpe']:>8.2f} {s['compound_ret']:>10.1f} {s['avg_maxdd']:>8.1f} {s['win_rate']:>7.0f}% {s['sharpe_gt_05']:>4}/{s['n_windows']}")
    
    # Detailed window-by-window for top 3
    print("\n" + "=" * 80)
    print("WINDOW-BY-WINDOW — Top 3 Combos + Baselines")
    print("=" * 80)
    
    # Sort by avg OOS sharpe
    sorted_wf = sorted(wf_all.items(), key=lambda x: x[1][0]['avg_sharpe'], reverse=True)
    
    for name, (s, results) in sorted_wf[:4]:
        print(f"\n--- {name} ---")
        print(f"  {'W':<4} {'OOS Start':>12} {'OOS End':>12} {'OOS Shp':>8} {'OOS Ret%':>10} {'OOS DD%':>8} {'Trades':>7}")
        print("  " + "-" * 70)
        for r in results:
            print(f"  W{r['window']:<3} {r['oos_start'].strftime('%Y-%m-%d'):>12} {r['oos_end'].strftime('%Y-%m-%d'):>12} {r['oos_sharpe']:>8.2f} {r['oos_ret']:>10.1f} {r['oos_maxdd']:>8.1f} {r['oos_trades']:>7}")
        print(f"\n  SUMMARY: AvgShp={s['avg_sharpe']:.2f} | MedShp={s['med_sharpe']:.2f} | CompOOS={s['compound_ret']:.1f}% | AvgDD={s['avg_maxdd']:.1f}% | WinRate={s['win_rate']:.0f}% | Shp>0.5: {s['sharpe_gt_05']}/{s['n_windows']}")
    
    # Test on DOGE and SOL too (best alt candidates)
    print("\n" + "=" * 80)
    print("MULTI-ASSET TEST — Top Combo")
    print("=" * 80)
    
    # Find best combo
    best_name = sorted_wf[0][0]
    best_fn = wf_fns[best_name]
    print(f"\nBest combo: {best_name}")
    
    alt_data = {
        'XLM/USD': xlm,
    }
    
    for sym, start in [('DOGE/USD', '2019-07-05'), ('SOL/USD', '2020-08-11')]:
        df = fetch_candles(sym, "1d", start, "2026-08-12")
        if len(df) > 200:
            alt_data[sym] = df
            print(f"  {sym}: {len(df)} candles")
    
    print(f"\n{'Asset':<12} {'FS_Ret%':>8} {'FS_Shp':>8} {'FS_DD%':>8} {'FS_Tr':>6} {'WF_AvgSh':>10} {'WF_Comp%':>10} {'WF_WR':>7} {'WF_>0.5':>8}")
    print("-" * 85)
    
    for sym, df in alt_data.items():
        pos = best_fn(df)
        m, eq, _ = backtest(df, pos)
        wf = walk_forward(df, best_fn)
        s = wf_summary(wf)
        if s:
            print(f"{sym:<12} {m['total_ret']:>8.1f} {m['sharpe']:>8.2f} {m['max_dd']:>8.1f} {m['n_trades']:>6} {s['avg_sharpe']:>10.2f} {s['compound_ret']:>10.1f} {s['win_rate']:>6.0f}% {s['sharpe_gt_05']:>4}/{s['n_windows']}")
        else:
            print(f"{sym:<12} {m['total_ret']:>8.1f} {m['sharpe']:>8.2f} {m['max_dd']:>8.1f} {m['n_trades']:>6} {'N/A':>10} {'N/A':>10} {'N/A':>7} {'N/A':>8}")
    
    # Plot
    print("\n" + "=" * 80)
    print("PLOTTING...")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Equity curves
    ax = axes[0]
    for name, (m, eq) in sorted(all_results.items(), key=lambda x: x[1][0]['sharpe'], reverse=True)[:6]:
        ax.plot(eq.index, eq.values, label=f"{name} (Shp={m['sharpe']:.2f})", linewidth=1.5)
    ax.set_yscale('log')
    ax.set_title('S1+S2 Combined Strategies — XLM/USD Full Sample (2019-2026)', fontsize=13)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Cumulative Return (log)')
    
    # Walk-forward OOS Sharpe comparison
    ax = axes[1]
    x = np.arange(len(wf_all))
    width = 0.12
    
    for i, (name, (s, results)) in enumerate(sorted_wf[:8]):
        sharpes = [r['oos_sharpe'] for r in results]
        ax.bar(x + i * width - 3 * width, sharpes, width, label=name[:15], alpha=0.7)
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axhline(0.5, color='green', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_title('Walk-Forward OOS Sharpe by Strategy', fontsize=12)
    ax.set_ylabel('OOS Sharpe')
    ax.set_xticks(x)
    ax.set_xticklabels([f"W{i}" for i in range(len(list(wf_all.values())[0][1]))])
    ax.legend(fontsize=7, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/s1_s2_combined.png', dpi=150, bbox_inches='tight')
    print("Chart saved: /tmp/s1_s2_combined.png")
    
    print("\nDone.")
