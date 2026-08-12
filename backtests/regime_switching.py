"""Regime-switching complementary strategies — fixed version with CSV data loading."""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# DATA LOADING (from pre-saved CSVs)
# ============================================================
def load_data(path):
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df[['open','high','low','close','volume']].sort_index()

# ============================================================
# INDICATORS
# ============================================================
def ichimoku(df, tenkan=9, kijun=26, senkou_b=52):
    high, low = df['high'], df['low']
    t = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    k = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    sa = ((t + k) / 2).shift(kijun)
    return pd.DataFrame({'tenkan': t, 'kijun': k, 'senkou_a': sa}, index=df.index)

def bollinger_bands(df, period=20, std_mult=2):
    sma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    return pd.DataFrame({'bb_mid': sma, 'bb_upper': sma + std_mult * std,
        'bb_lower': sma - std_mult * std,
        'bb_bw': (sma + std_mult * std - (sma - std_mult * std)) / sma}, index=df.index)

def keltner_channel(df, period=20, atr_mult=2):
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr_val = tr.rolling(period).mean()
    ema = df['close'].ewm(span=period, adjust=False).mean()
    return pd.DataFrame({'kc_mid': ema, 'kc_upper': ema + atr_mult*atr_val,
        'kc_lower': ema - atr_mult*atr_val}, index=df.index)

def rsi(df, period=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    ag = gain.ewm(alpha=1/period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, adjust=False).mean()
    return 100 - (100 / (1 + ag / al))

def obv(df):
    return (np.sign(df['close'].diff()) * df['volume']).cumsum()

def sma(df, period):
    return df['close'].rolling(period).mean()

# ============================================================
# STRATEGIES
# ============================================================
def s1_ichimoku(df):
    ichi = ichimoku(df)
    bull = (df['close'] > ichi['tenkan']) & (ichi['tenkan'] > ichi['kijun']) & (df['close'] > ichi['senkou_a'])
    return bull.astype(float).shift(1).fillna(0)

def s11_keltner_mean_reversion(df):
    kc = keltner_channel(df)
    pos = pd.Series(0.0, index=df.index)
    dipped = df['low'] < kc['kc_lower']
    recovered = df['close'] > kc['kc_lower']
    entry = dipped & recovered
    pos[entry] = 1.0
    pos[(df['close'] < kc['kc_mid']) & (pos.shift(1) == 1)] = 1.0
    pos[df['close'] >= kc['kc_mid']] = 0.0
    return pos.shift(1).fillna(0)

def s12_squeeze_rsi_div(df):
    bb = bollinger_bands(df)
    r = rsi(df)
    pos = pd.Series(0.0, index=df.index)
    bw_pct = bb['bb_bw'].rolling(50).rank(pct=True)
    was_sq = (bw_pct < 0.20).rolling(10).max() > 0
    p_lows = df['low'].rolling(50, min_periods=10).min()
    p_ll = df['low'] < p_lows.shift(5)
    r_lows = r.rolling(50, min_periods=10).min()
    r_hl = r > r_lows.shift(5)
    r_turn = (r > r.shift(1)) & (r.shift(1) < 40)
    entry = was_sq & p_ll & r_hl & r_turn
    pos[entry] = 1.0
    pos[(df['close'] > bb['bb_lower']) & (pos.shift(1) == 1)] = 1.0
    pos[df['close'] < bb['bb_lower'] * 0.98] = 0.0
    pos[df['close'] > bb['bb_upper']] = 0.0
    return pos.shift(1).fillna(0)

def s13_wyckoff_spring(df):
    pos = pd.Series(0.0, index=df.index)
    r_low = df['low'].rolling(50).min()
    r_high = df['high'].rolling(50).max()
    spring = (df['low'] < r_low.shift(1)) & (df['close'] > r_low.shift(1))
    avg_vol = df['volume'].rolling(50).mean()
    vol_dec = df['volume'] < avg_vol * 0.8
    entry = spring & vol_dec
    pos[entry] = 1.0
    pos[(df['close'] > r_low * 0.98) & (pos.shift(1) == 1)] = 1.0
    pos[df['close'] > r_high] = 0.0
    pos[df['close'] < r_low * 0.95] = 0.0
    return pos.shift(1).fillna(0)

def s14_overextended_fade(df):
    """SHORT when overextended above upper BB."""
    bb = bollinger_bands(df)
    pos = pd.Series(0.0, index=df.index)
    bb_std_val = (bb['bb_upper'] - bb['bb_mid']) / 2
    z_above = (df['close'] - bb['bb_upper']) / bb_std_val
    entry = z_above > 1.0
    pos[entry] = -1.0
    pos[(df['close'] > bb['bb_mid']) & (pos.shift(1) == -1)] = -1.0
    pos[df['close'] <= bb['bb_mid']] = 0.0
    return pos.shift(1).fillna(0)

def s15_obv_divergence(df):
    o = obv(df)
    pos = pd.Series(0.0, index=df.index)
    p_lows = df['low'].rolling(50, min_periods=10).min()
    p_ll = df['low'] < p_lows.shift(10)
    o_lows = o.rolling(50, min_periods=10).min()
    o_hl = o > o_lows.shift(10)
    entry = p_ll & o_hl
    pos[entry] = 1.0
    # Hold 20 bars
    for i in range(20, len(df)):
        if pos.iloc[i] == 0 and pos.iloc[i-1] == 1:
            for j in range(i-1, max(i-21, 0), -1):
                if pos.iloc[j] == 1 and (j == 0 or pos.iloc[j-1] == 0):
                    if i - j < 20: pos.iloc[i] = 1.0
                    break
    pos[df['close'] < p_lows.shift(50) * 0.97] = 0.0
    return pos.shift(1).fillna(0)

def s16_sma_reclaim(df, period=20):
    s = sma(df, period)
    pos = pd.Series(0.0, index=df.index)
    was_below = (df['close'].shift(5) < s.shift(5)).rolling(5).sum() >= 3
    reclaim = (df['close'] > s) & was_below
    slope = (s - s.shift(10)) / s.shift(10)
    flat = slope.abs() < 0.05
    entry = reclaim & flat
    pos[entry] = 1.0
    pos[(df['close'] > s) & (pos.shift(1) == 1)] = 1.0
    pos[df['close'] < s] = 0.0
    return pos.shift(1).fillna(0)

def s17_ichimoku_short(df):
    ichi = ichimoku(df)
    bear = (df['close'] < ichi['tenkan']) & (ichi['tenkan'] < ichi['kijun']) & (df['close'] < ichi['senkou_a'])
    return (-bear.astype(float)).shift(1).fillna(0)

def s18_keltner_squeeze(df):
    kc = keltner_channel(df)
    bb = bollinger_bands(df)
    pos = pd.Series(0.0, index=df.index)
    bw_pct = bb['bb_bw'].rolling(50).rank(pct=True)
    was_sq = (bw_pct < 0.25).rolling(15).max() > 0
    dipped = df['low'] < kc['kc_lower']
    recovered = df['close'] > kc['kc_lower']
    entry = was_sq & dipped & recovered
    pos[entry] = 1.0
    pos[(df['close'] < kc['kc_mid']) & (pos.shift(1) == 1)] = 1.0
    pos[df['close'] >= kc['kc_mid']] = 0.0
    pos[df['close'] < kc['kc_lower'] * 0.98] = 0.0
    return pos.shift(1).fillna(0)

def s1_plus_parallel_longonly(df):
    """S1 long + complementary long when S1 is flat."""
    s1 = s1_ichimoku(df)
    s11 = s11_keltner_mean_reversion(df)
    s12 = s12_squeeze_rsi_div(df)
    s18 = s18_keltner_squeeze(df)
    pos = pd.Series(0.0, index=df.index)
    pos[s1 > 0] = 1.0
    flat = s1 == 0
    pos[flat & (s18 > 0)] = 1.0
    pos[flat & (s11 > 0) & (s18 == 0)] = 1.0
    pos[flat & (s12 > 0) & (s18 == 0) & (s11 == 0)] = 1.0
    return pos

def s1_plus_s17_with_fade(df):
    """S1 long in trend up, S17 short in trend down, S14 fade when overextended."""
    s1 = s1_ichimoku(df)
    s17 = s17_ichimoku_short(df)
    s14 = s14_overextended_fade(df)
    pos = pd.Series(0.0, index=df.index)
    # S1 long when bullish
    pos[s1 > 0] = 1.0
    # S17 short when bearish
    pos[s17 < 0] = -1.0
    # S14 fade when overextended (overrides if conflicting)
    pos[(s14 < 0) & (s1 == 0)] = -1.0
    return pos

def regime_switch_master(df):
    """Master regime switcher — non-overlapping regimes."""
    ichi = ichimoku(df)
    bb = bollinger_bands(df)
    r = rsi(df)
    kc = keltner_channel(df)
    
    regime = pd.Series('NEUTRAL', index=df.index)
    
    trend_up = (df['close'] > ichi['tenkan']) & (ichi['tenkan'] > ichi['kijun']) & (df['close'] > ichi['senkou_a'])
    trend_dn = (df['close'] < ichi['tenkan']) & (ichi['tenkan'] < ichi['kijun']) & (df['close'] < ichi['senkou_a'])
    regime[trend_up] = 'TREND_UP'
    regime[trend_dn] = 'TREND_DOWN'
    
    bw_pct = bb['bb_bw'].rolling(50).rank(pct=True)
    chop = (bw_pct < 0.20) & ~trend_up & ~trend_dn
    regime[chop] = 'CHOP'
    
    bb_std_val = (bb['bb_upper'] - bb['bb_mid']) / 2
    over = (df['close'] > bb['bb_upper'] + 0.5 * bb_std_val) & ~trend_up
    regime[over] = 'OVEREXTENDED'
    
    oversold = (df['close'] < bb['bb_lower']) & (r < 35) & ~trend_dn
    regime[oversold] = 'OVERSOLD'
    
    # Build positions based on regime
    s1 = s1_ichimoku(df)
    s11 = s11_keltner_mean_reversion(df)
    s14 = s14_overextended_fade(df)
    s17 = s17_ichimoku_short(df)
    s18 = s18_keltner_squeeze(df)
    
    pos = pd.Series(0.0, index=df.index)
    pos[regime == 'TREND_UP'] = s1[regime == 'TREND_UP']
    pos[regime == 'TREND_DOWN'] = s17[regime == 'TREND_DOWN']
    pos[regime == 'CHOP'] = s18[regime == 'CHOP']
    pos[regime == 'OVEREXTENDED'] = s14[regime == 'OVEREXTENDED']
    pos[regime == 'OVERSOLD'] = s11[regime == 'OVERSOLD']
    
    return pos

# ============================================================
# BACKTEST + WALK-FORWARD
# ============================================================
def compute_metrics(rets, ann=365):
    c = rets.dropna()
    if len(c) == 0: return {}
    tr = (1 + c).prod() - 1
    par = (1 + tr) ** (ann / len(c)) - 1 if tr > -1 else -1
    v = c.std() * np.sqrt(ann)
    sh = par / v if v > 0 else 0
    ds = c[c < 0]
    dv = ds.std() * np.sqrt(ann) if len(ds) > 0 else 0
    so = par / dv if dv > 0 else 0
    cum = (1 + c).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    return {'total_ret': tr*100, 'pa_ret': par*100, 'vol': v*100, 'sharpe': sh,
            'sortino': so, 'max_dd': dd*100, 'calmar': par/abs(dd) if dd < 0 else 0}

def backtest(df, pos, cost=0.0005, ann=365):
    ret = df['close'].pct_change()
    tr = pos.diff().abs()
    sr = pos * ret - tr * cost
    nt = int((tr > 0).sum())
    ti = (pos != 0).sum() / len(pos) * 100
    m = compute_metrics(sr, ann)
    m['n_trades'] = nt; m['time_in'] = ti
    return m, (1 + sr).cumprod()

def walk_forward(df, fn, is_m=12, oos_m=6, ann=365, cost=0.0005):
    res = []
    si = 0
    while True:
        ie = si + is_m * 30
        os = ie; oe = os + oos_m * 30
        if oe >= len(df): break
        isd = df.iloc[si:ie]; oosd = df.iloc[os:oe]
        if len(isd) < 100 or len(oosd) < 50: break
        op = fn(oosd)
        om, _ = backtest(oosd, op, cost, ann)
        res.append({'w': len(res), 'start': oosd.index[0], 'end': oosd.index[-1],
                    'shp': om.get('sharpe',0), 'ret': om.get('total_ret',0),
                    'dd': om.get('max_dd',0), 'tr': om.get('n_trades',0)})
        si = os
    return res

def wf_sum(res):
    if not res: return None
    shp = [r['shp'] for r in res]
    rts = [r['ret'] for r in res]
    comp = 1.0
    for r in rts: comp *= (1 + r/100)
    pos = sum(1 for r in rts if r > 0)
    return {'n': len(res), 'avg_shp': np.mean(shp), 'med_shp': np.median(shp),
            'comp': (comp-1)*100, 'avg_dd': np.mean([r['dd'] for r in res]),
            'wr': pos/len(res)*100, 'shp_gt': sum(1 for s in shp if s > 0.5)}

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 80)
    print("REGIME-SWITCHING COMPLEMENTARY STRATEGIES — DAILY DATA")
    print("=" * 80)
    
    data = {}
    for sym, path in [("XLM/USD","/tmp/XLM_USD_1d.csv"),("DOGE/USD","/tmp/DOGE_USD_1d.csv"),("SOL/USD","/tmp/SOL_USD_1d.csv")]:
        if os.path.exists(path):
            df = load_data(path)
            data[sym] = df
            print(f"  {sym}: {len(df)} candles, {df.index[0]} → {df.index[-1]}")
    
    strategies = {
        'S1: Ichimoku (baseline)': s1_ichimoku,
        'S11: Keltner Mean Reversion': s11_keltner_mean_reversion,
        'S12: Squeeze+RSI Divergence': s12_squeeze_rsi_div,
        'S13: Wyckoff Spring': s13_wyckoff_spring,
        'S14: Overextended Fade (SHORT)': s14_overextended_fade,
        'S15: OBV Divergence': s15_obv_divergence,
        'S16: 20-SMA Reclaim': s16_sma_reclaim,
        'S17: Ichimoku Short': s17_ichimoku_short,
        'S18: Keltner+Squeeze': s18_keltner_squeeze,
        'REGIME SWITCH (master)': regime_switch_master,
        'S1 + PARALLEL (long-only)': s1_plus_parallel_longonly,
        'S1+S17+FADE (long/short)': s1_plus_s17_with_fade,
    }
    
    # FULL-SAMPLE
    for aname, df in data.items():
        print(f"\n{'=' * 80}")
        print(f"FULL-SAMPLE — {aname} ({len(df)} candles)")
        print(f"{'=' * 80}")
        print(f"\n{'Strategy':<35} {'TotRet%':>10} {'Sharpe':>7} {'Sortino':>7} {'MaxDD%':>8} {'Calmar':>7} {'Trades':>7} {'InMkt%':>7}")
        print("-" * 95)
        
        results = {}
        for name, fn in strategies.items():
            try:
                pos = fn(df)
                m, eq = backtest(df, pos)
                results[name] = (m, eq)
                print(f"{name:<35} {m['total_ret']:>10.1f} {m['sharpe']:>7.2f} {m['sortino']:>7.2f} {m['max_dd']:>8.1f} {m['calmar']:>7.2f} {m['n_trades']:>7} {m['time_in']:>7.1f}")
            except Exception as e:
                print(f"{name:<35} ERROR: {str(e)[:40]}")
    
    # WALK-FORWARD
    print(f"\n{'=' * 80}")
    print("WALK-FORWARD VALIDATION (12mo IS → 6mo OOS)")
    print(f"{'=' * 80}")
    
    for aname, df in data.items():
        print(f"\n--- {aname} ---")
        print(f"{'Strategy':<35} {'AvgShp':>8} {'MedShp':>8} {'CompOOS%':>10} {'AvgDD%':>8} {'WinRate':>8} {'Shp>0.5':>8}")
        print("-" * 95)
        
        for name, fn in strategies.items():
            try:
                res = walk_forward(df, fn)
                s = wf_sum(res)
                if s:
                    print(f"{name:<35} {s['avg_shp']:>8.2f} {s['med_shp']:>8.2f} {s['comp']:>10.1f} {s['avg_dd']:>8.1f} {s['wr']:>7.0f}% {s['shp_gt']:>4}/{s['n']}")
            except Exception as e:
                print(f"{name:<35} ERROR: {str(e)[:40]}")
    
    # S1 vs COMBOS comparison
    print(f"\n{'=' * 80}")
    print("S1 vs COMBOS — WALK-FORWARD COMPARISON")
    print(f"{'=' * 80}")
    
    print(f"\n{'Asset':<12} | {'Strategy':<30} | {'CompOOS%':>10} {'WinRate':>8} {'AvgShp':>8}")
    print("-" * 75)
    
    key_strats = {
        'S1 alone': s1_ichimoku,
        'S1 + Parallel': s1_plus_parallel_longonly,
        'S1+S17+Fade': s1_plus_s17_with_fade,
        'Regime Switch': regime_switch_master,
    }
    
    for aname, df in data.items():
        for sname, fn in key_strats.items():
            try:
                res = walk_forward(df, fn)
                s = wf_sum(res)
                if s:
                    print(f"{aname:<12} | {sname:<30} | {s['comp']:>10.1f} {s['wr']:>7.0f}% {s['avg_shp']:>8.2f}")
            except:
                print(f"{aname:<12} | {sname:<30} | ERROR")
        print()
    
    # PLOT
    print("Plotting...")
    fig, axes = plt.subplots(len(data), 1, figsize=(16, 5*len(data)))
    if len(data) == 1: axes = [axes]
    
    for i, (aname, df) in enumerate(data.items()):
        ax = axes[i]
        for sname, fn in key_strats.items():
            try:
                pos = fn(df)
                m, eq = backtest(df, pos)
                ax.plot(eq.index, eq.values, label=f"{sname} (Shp={m['sharpe']:.2f})", linewidth=1.5)
            except: pass
        ax.set_yscale('log')
        ax.set_title(f'{aname} — Regime-Switching Comparison', fontsize=13)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Cumulative Return (log)')
    
    plt.tight_layout()
    plt.savefig('/tmp/regime_switching.png', dpi=150, bbox_inches='tight')
    print("Chart saved: /tmp/regime_switching.png")
    
    # Regime distribution
    print(f"\n{'=' * 80}")
    print("REGIME DISTRIBUTION")
    print(f"{'=' * 80}")
    for aname, df in data.items():
        ichi = ichimoku(df)
        bb = bollinger_bands(df)
        r = rsi(df)
        
        trend_up = (df['close'] > ichi['tenkan']) & (ichi['tenkan'] > ichi['kijun']) & (df['close'] > ichi['senkou_a'])
        trend_dn = (df['close'] < ichi['tenkan']) & (ichi['tenkan'] < ichi['kijun']) & (df['close'] < ichi['senkou_a'])
        bw_pct = bb['bb_bw'].rolling(50).rank(pct=True)
        chop = (bw_pct < 0.20) & ~trend_up & ~trend_dn
        bb_std_val = (bb['bb_upper'] - bb['bb_mid']) / 2
        over = (df['close'] > bb['bb_upper'] + 0.5 * bb_std_val) & ~trend_up
        oversold = (df['close'] < bb['bb_lower']) & (r < 35) & ~trend_dn
        neutral = ~trend_up & ~trend_dn & ~chop & ~over & ~oversold
        
        print(f"\n{aname}:")
        print(f"  TREND_UP:     {trend_up.sum():>5} ({trend_up.mean()*100:.1f}%)")
        print(f"  TREND_DOWN:   {trend_dn.sum():>5} ({trend_dn.mean()*100:.1f}%)")
        print(f"  CHOP:         {chop.sum():>5} ({chop.mean()*100:.1f}%)")
        print(f"  OVEREXTENDED: {over.sum():>5} ({over.mean()*100:.1f}%)")
        print(f"  OVERSOLD:     {oversold.sum():>5} ({oversold.mean()*100:.1f}%)")
        print(f"  NEUTRAL:      {neutral.sum():>5} ({neutral.mean()*100:.1f}%)")
    
    print("\nDone.")
