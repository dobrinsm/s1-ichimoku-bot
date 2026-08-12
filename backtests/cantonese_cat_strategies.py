"""
Cantonese Cat Trading Strategies — Backtest Suite
====================================================
Derived from @cantonmeow's chart analysis (100+ tweets analyzed, 10 charts vision-analyzed).

STRATEGIES EXTRACTED:
1. Ichimoku Tenkan Ride — Long when price > Tenkan-sen on monthly/daily
2. Bollinger Band Squeeze Breakout — Trade the squeeze expansion
3. RSI Bullish Divergence — RSI rounding bottom + price lower lows
4. Fibonacci 0.786 Breakout — Long when price reclaims 0.786 Fib
5. Volume Decline + Support Hold — Bullish when volume declining on pullback to support
6. Wyckoff Accumulation Detection — Spring + volume confirmation
7. Cross-Asset Bollinger Squeeze — BB squeeze on QQQ/SPY predicts crypto moves
8. Multi-Timeframe Ichimoku — 1h+4h Ichimoku cloud confirmation
9. Order Block Support — Price returns to buy order block zone
10. Supertrend/Ichimoku Cloud Flip — Resistance flips to support

Each strategy is coded as a signal function returning position (1=long, 0=flat).
We backtest on BTC/USD (primary) and XLM/USD (secondary) using LSE data.
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

# ============================================================
# DATA FETCHING
# ============================================================
client = LSE()

def fetch_candles(symbol, timeframe, start_date, end_date):
    """Fetch candles from LSE API with pagination."""
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

def fetch_multi_tf(symbol, start_date, end_date):
    """Fetch 1h, 4h, and 1d candles for multi-timeframe analysis."""
    print(f"Fetching {symbol} multi-timeframe data...")
    tfs = {}
    for tf_name, tf in [("1d", "1d"), ("4h", "4h"), ("1h", "1h")]:
        df = fetch_candles(symbol, tf, start_date, end_date)
        if len(df) > 0:
            tfs[tf_name] = df
            print(f"  {tf_name}: {len(df)} candles, {df.index[0]} → {df.index[-1]}")
    return tfs

# ============================================================
# INDICATOR FUNCTIONS
# ============================================================

def ichimoku(df, tenkan=9, kijun=26, senkou_b=52):
    """Compute Ichimoku Cloud components."""
    high = df['high']
    low = df['low']
    
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b_val = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    
    chikou = df['close'].shift(-kijun)
    
    return pd.DataFrame({
        'tenkan': tenkan_sen,
        'kijun': kijun_sen,
        'senkou_a': senkou_a,
        'senkou_b': senkou_b_val,
        'chikou': chikou
    }, index=df.index)

def bollinger_bands(df, period=20, std_mult=2):
    """Standard Bollinger Bands."""
    sma = df['close'].rolling(period).mean()
    std = df['close'].rolling(period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    return pd.DataFrame({
        'bb_mid': sma, 'bb_upper': upper, 'bb_lower': lower, 'bb_bw': bandwidth
    }, index=df.index)

def rsi(df, period=14):
    """Relative Strength Index."""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def fibonacci_levels(df, lookback=100):
    """Compute Fibonacci retracement levels from recent swing high/low."""
    window = df.tail(lookback)
    swing_high = window['high'].max()
    swing_low = window['low'].min()
    
    levels = {
        0.0: swing_low,
        0.382: swing_low + 0.382 * (swing_high - swing_low),
        0.5: swing_low + 0.5 * (swing_high - swing_low),
        0.618: swing_low + 0.618 * (swing_high - swing_low),
        0.786: swing_low + 0.786 * (swing_high - swing_low),
        0.886: swing_low + 0.886 * (swing_high - swing_low),
        1.0: swing_high,
        1.272: swing_high + 0.272 * (swing_high - swing_low),
        1.414: swing_high + 0.414 * (swing_high - swing_low),
    }
    return levels

def atr(df, period=14):
    """Average True Range."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def detect_order_blocks(df, lookback=20, threshold=0.02):
    """
    Detect buy/sell order blocks: zones where price had strong moves with high volume.
    Simplified: find candles with volume > 1.5x average and strong body.
    """
    avg_vol = df['volume'].rolling(lookback).mean()
    body = (df['close'] - df['open']).abs()
    avg_body = body.rolling(lookback).mean()
    
    # Buy order block: bullish candle with high volume
    buy_ob = (df['close'] > df['open']) & (df['volume'] > 1.5 * avg_vol) & (body > 1.5 * avg_body)
    
    # Sell order block: bearish candle with high volume
    sell_ob = (df['close'] < df['open']) & (df['volume'] > 1.5 * avg_vol) & (body > 1.5 * avg_body)
    
    return buy_ob, sell_ob

def supertrend(df, period=10, multiplier=3):
    """Supertrend indicator."""
    atr_val = atr(df, period)
    hl2 = (df['high'] + df['low']) / 2
    
    upper_band = hl2 + multiplier * atr_val
    lower_band = hl2 - multiplier * atr_val
    
    st = pd.Series(index=df.index, dtype=float)
    trend = pd.Series(index=df.index, dtype=int)
    
    st.iloc[0] = upper_band.iloc[0]
    trend.iloc[0] = 1
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > st.iloc[i-1]:
            trend.iloc[i] = 1
        elif df['close'].iloc[i] < st.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
        
        if trend.iloc[i] == 1:
            st.iloc[i] = min(lower_band.iloc[i], st.iloc[i-1]) if trend.iloc[i-1] == 1 else lower_band.iloc[i]
        else:
            st.iloc[i] = max(upper_band.iloc[i], st.iloc[i-1]) if trend.iloc[i-1] == -1 else upper_band.iloc[i]
    
    return st, trend

# ============================================================
# STRATEGY SIGNALS
# ============================================================

def strategy_1_ichimoku_tenkan_ride(df):
    """
    Strategy 1: Ichimoku Tenkan Ride
    - Long when price > Tenkan-sen AND Tenkan > Kijun (bullish alignment)
    - Exit when price < Tenkan-sen
    - Cantonese Cat: "Riding Tenkan up" pattern on monthly/daily
    """
    ichi = ichimoku(df)
    pos = pd.Series(0.0, index=df.index)
    
    # Entry: price above Tenkan AND Tenkan above Kijun
    bullish = (df['close'] > ichi['tenkan']) & (ichi['tenkan'] > ichi['kijun'])
    pos[bullish] = 1.0
    
    # Also require price above cloud for extra confirmation
    above_cloud = df['close'] > ichi['senkou_a']
    pos[~above_cloud] = 0.0
    
    return pos.shift(1).fillna(0)

def strategy_2_bb_squeeze_breakout(df, bb_period=20, bb_std=2, squeeze_lookback=50, squeeze_pctile=20):
    """
    Strategy 2: Bollinger Band Squeeze Breakout
    - Detect squeeze: BB bandwidth at lowest 20th percentile of last 50 periods
    - Enter long when price breaks above upper BB after squeeze
    - Exit when price returns to mid BB
    - Cantonese Cat: BB squeeze → expansion = directional move
    """
    bb = bollinger_bands(df, bb_period, bb_std)
    pos = pd.Series(0.0, index=df.index)
    
    # Squeeze detection
    bw_pctile = bb['bb_bw'].rolling(squeeze_lookback).rank(pct=True)
    in_squeeze = bw_pctile < (squeeze_pctile / 100)
    
    # Was in squeeze in last N periods
    was_squeeze = in_squeeze.rolling(10).max() > 0
    
    # Breakout: price above upper BB
    breakout = (df['close'] > bb['bb_upper']) & was_squeeze
    
    # Position: stay long after breakout until price returns to mid
    pos[breakout] = 1.0
    # Hold while above mid BB
    pos[(df['close'] > bb['bb_mid']) & (pos.shift(1) == 1)] = 1.0
    # Exit below mid
    pos[df['close'] < bb['bb_mid']] = 0.0
    
    return pos.shift(1).fillna(0)

def strategy_3_rsi_divergence(df, rsi_period=14, lookback=20, rsi_low=35):
    """
    Strategy 3: RSI Bullish Divergence
    - Detect: price making lower lows while RSI making higher lows (rounding bottom)
    - Entry: when RSI turns up from low zone
    - Exit: RSI > 70 or price makes new low
    - Cantonese Cat: RSI rounding bottom = bullish divergence, selling momentum weakening
    """
    rsi_val = rsi(df, rsi_period)
    pos = pd.Series(0.0, index=df.index)
    
    # Find local minima in price and RSI
    price_lows = df['low'].rolling(lookback, min_periods=5).min()
    rsi_lows = rsi_val.rolling(lookback, min_periods=5).min()
    
    # Price making lower low
    price_lower_low = df['low'] < price_lows.shift(5)
    
    # RSI making higher low (divergence)
    rsi_higher_low = rsi_val > rsi_lows.shift(5)
    
    # RSI in low zone and turning up
    rsi_turning_up = (rsi_val > rsi_val.shift(1)) & (rsi_val.shift(1) < rsi_low)
    
    # Entry signal
    entry = price_lower_low & rsi_higher_low & rsi_turning_up
    pos[entry] = 1.0
    
    # Hold while RSI rising
    pos[(rsi_val > rsi_val.shift(1)) & (pos.shift(1) == 1)] = 1.0
    
    # Exit when RSI > 70
    pos[rsi_val > 70] = 0.0
    # Or exit on new low
    pos[df['low'] < price_lows.shift(lookback)] = 0.0
    
    return pos.shift(1).fillna(0)

def strategy_4_fib_breakout(df, lookback=100, fib_level=0.786):
    """
    Strategy 4: Fibonacci 0.786 Breakout
    - Compute Fib levels from last 100-period swing
    - Long when price breaks above 0.786 Fib level
    - Exit when price falls below 0.618 Fib
    - Cantonese Cat: 0.786 is key level, above = bullish continuation
    """
    pos = pd.Series(0.0, index=df.index)
    
    # Rolling Fibonacci levels
    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i]
        swing_high = window['high'].max()
        swing_low = window['low'].min()
        fib_786 = swing_low + fib_level * (swing_high - swing_low)
        fib_618 = swing_low + 0.618 * (swing_high - swing_low)
        
        if df['close'].iloc[i] > fib_786:
            pos.iloc[i] = 1.0
        elif df['close'].iloc[i] < fib_618:
            pos.iloc[i] = 0.0
        else:
            pos.iloc[i] = pos.iloc[i-1]
    
    return pos.shift(1).fillna(0)

def strategy_5_volume_decline_support(df, lookback=20, vol_lookback=50):
    """
    Strategy 5: Volume Decline + Support Hold
    - Detect: price at support (lower BB or recent lows) with declining volume
    - Entry: price bouncing off support on declining volume
    - Exit: price breaks support
    - Cantonese Cat: "declining volume on sells = bullish, time correction"
    """
    bb = bollinger_bands(df, 20, 2)
    pos = pd.Series(0.0, index=df.index)
    
    avg_vol = df['volume'].rolling(vol_lookback).mean()
    vol_declining = df['volume'] < avg_vol * 0.7  # Volume below 70% of average
    
    # Support: lower BB or recent swing low
    recent_low = df['low'].rolling(lookback).min()
    at_support = (df['low'] <= bb['bb_lower'] * 1.01) | (df['low'] <= recent_low * 1.01)
    
    # Bouncing: close above support
    bouncing = df['close'] > df['low'] * 1.005
    
    # Entry: at support + declining volume + bouncing
    entry = at_support & vol_declining & bouncing
    pos[entry] = 1.0
    
    # Hold while above lower BB
    pos[(df['close'] > bb['bb_lower']) & (pos.shift(1) == 1)] = 1.0
    
    # Exit below support
    pos[df['close'] < bb['bb_lower'] * 0.98] = 0.0
    
    return pos.shift(1).fillna(0)

def strategy_6_wyckoff_accumulation(df, bb_period=20, lookback=50):
    """
    Strategy 6: Wyckoff Accumulation Detection
    - Detect: prolonged consolidation with declining volume after downtrend
    - Entry: Spring (false breakdown that recovers) or SOS (sign of strength)
    - Exit: price fails to hold accumulation range
    - Cantonese Cat: BB squeeze + Wyckoff accumulation on BTC
    """
    bb = bollinger_bands(df, bb_period, 2)
    pos = pd.Series(0.0, index=df.index)
    
    # Downtrend detection: price below mid BB
    in_downtrend = df['close'] < bb['bb_mid']
    
    # Consolidation: BB bandwidth narrowing
    bw_pctile = bb['bb_bw'].rolling(lookback).rank(pct=True)
    consolidating = bw_pctile < 0.3  # Bottom 30% = tight
    
    # Volume declining
    avg_vol = df['volume'].rolling(lookback).mean()
    vol_declining = df['volume'] < avg_vol * 0.8
    
    # Spring: price dips below lower BB but closes back inside
    spring = (df['low'] < bb['bb_lower']) & (df['close'] > bb['bb_lower'])
    
    # Accumulation conditions
    accum = in_downtrend & consolidating & vol_declining
    entry = accum & spring
    pos[entry] = 1.0
    
    # Hold while above lower BB
    pos[(df['close'] > bb['bb_lower']) & (pos.shift(1) == 1)] = 1.0
    
    # Exit below lower BB (confirmed breakdown)
    pos[df['close'] < bb['bb_lower'] * 0.97] = 0.0
    # Exit after significant gain (take profit)
    pos[df['close'] > bb['bb_upper']] = 0.0
    
    return pos.shift(1).fillna(0)

def strategy_7_cross_asset_bb_squeeze(btc_df, spy_df, bb_period=20, squeeze_pctile=20):
    """
    Strategy 7: Cross-Asset Bollinger Squeeze
    - When SPY/QQQ BB squeezes, crypto tends to follow with a big move
    - Enter BTC long when SPY breaks out of squeeze + BTC confirms
    - Cantonese Cat: tracks QQQ/SPY for crypto direction
    """
    spy_bb = bollinger_bands(spy_df, bb_period, 2)
    btc_bb = bollinger_bands(btc_df, bb_period, 2)
    
    pos = pd.Series(0.0, index=btc_df.index)
    
    # SPY squeeze detection
    spy_bw_pctile = spy_bb['bb_bw'].rolling(50).rank(pct=True)
    spy_squeeze = spy_bw_pctile < (squeeze_pctile / 100)
    spy_was_squeeze = spy_squeeze.rolling(10).max() > 0
    spy_breakout = spy_df['close'] > spy_bb['bb_upper']
    
    # BTC confirmation: also breaking out
    btc_breakout = btc_df['close'] > btc_bb['bb_upper']
    
    # Align timestamps
    spy_aligned = spy_was_squeeze.reindex(btc_df.index, method='ffill')
    spy_bo_aligned = spy_breakout.reindex(btc_df.index, method='ffill')
    
    # Entry: SPY breaking out of squeeze + BTC also breaking out
    entry = spy_bo_aligned & spy_aligned & btc_breakout
    pos[entry] = 1.0
    
    # Hold while BTC above mid BB
    pos[(btc_df['close'] > btc_bb['bb_mid']) & (pos.shift(1) == 1)] = 1.0
    pos[btc_df['close'] < btc_bb['bb_mid']] = 0.0
    
    return pos.shift(1).fillna(0)

def strategy_8_mtf_ichimoku(dfs):
    """
    Strategy 8: Multi-Timeframe Ichimoku
    - 1h: Tenkan > Kijun + price above cloud
    - 4h: Tenkan > Kijun + price above cloud
    - Both must align to enter
    - Cantonese Cat: uses 4H for structure, 1h for timing
    """
    ichi_1h = ichimoku(dfs['1h'])
    ichi_4h = ichimoku(dfs['4h'])
    
    # 1h signal
    sig_1h = (dfs['1h']['close'] > ichi_1h['tenkan']) & \
             (ichi_1h['tenkan'] > ichi_1h['kijun']) & \
             (dfs['1h']['close'] > ichi_1h['senkou_a'])
    
    # 4h signal
    sig_4h = (dfs['4h']['close'] > ichi_4h['tenkan']) & \
             (ichi_4h['tenkan'] > ichi_4h['kijun']) & \
             (dfs['4h']['close'] > ichi_4h['senkou_a'])
    
    # Align 4h to 1h
    sig_4h_aligned = sig_4h.reindex(dfs['1h'].index, method='ffill')
    
    pos = pd.Series(0.0, index=dfs['1h'].index)
    pos[sig_1h & sig_4h_aligned] = 1.0
    
    return pos.shift(1).fillna(0)

def strategy_9_order_block(df, lookback=20):
    """
    Strategy 9: Order Block Support
    - Detect buy order blocks (high volume bullish candles)
    - Enter when price returns to order block zone and bounces
    - Exit when order block is broken
    - Cantonese Cat: order blocks as primary support/demand zones
    """
    buy_ob, sell_ob = detect_order_blocks(df, lookback)
    pos = pd.Series(0.0, index=df.index)
    
    # Track active order blocks
    ob_zones = []  # List of (start_idx, end_idx, price_low, price_high)
    
    for i in range(lookback, len(df)):
        # Update existing zones
        new_zones = []
        for start, end, pl, ph in ob_zones:
            if df['low'].iloc[i] < pl * 0.98:  # Broken
                continue
            new_zones.append((start, end, pl, ph))
        ob_zones = new_zones
        
        # Detect new buy OB
        if buy_ob.iloc[i]:
            ob_zones.append((i, i, df['low'].iloc[i], df['high'].iloc[i]))
        
        # Check if price is at any active OB zone
        at_ob = False
        for start, end, pl, ph in ob_zones:
            if pl * 0.99 <= df['low'].iloc[i] <= ph * 1.01:
                at_ob = True
                break
        
        if at_ob and df['close'].iloc[i] > df['open'].iloc[i]:  # Bouncing
            pos.iloc[i] = 1.0
        elif pos.iloc[i-1] == 1.0:
            # Hold while above lowest OB
            min_ob = min(z[2] for z in ob_zones) if ob_zones else 0
            if df['close'].iloc[i] > min_ob * 0.98:
                pos.iloc[i] = 1.0
    
    return pos.shift(1).fillna(0)

def strategy_10_supertrend_cloud_flip(df):
    """
    Strategy 10: Supertrend + Ichimoku Cloud Flip
    - Enter when Supertrend flips bullish AND price above Ichimoku cloud
    - Exit when Supertrend flips bearish
    - Cantonese Cat: "resistance flip to support" on Ichimoku cloud
    """
    st, trend = supertrend(df, 10, 3)
    ichi = ichimoku(df)
    
    pos = pd.Series(0.0, index=df.index)
    
    # Bullish: supertrend positive + price above cloud
    bullish = (trend == 1) & (df['close'] > ichi['senkou_a'])
    pos[bullish] = 1.0
    
    # Exit when supertrend flips
    pos[trend == -1] = 0.0
    
    return pos.shift(1).fillna(0)

# ============================================================
# BACKTEST ENGINE
# ============================================================

def compute_metrics(returns_series, ann_factor=365):
    """Compute risk-adjusted metrics."""
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
        'total_ret': total_ret * 100,
        'pa_ret': pa_ret * 100,
        'vol': vol * 100,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd * 100,
        'calmar': calmar,
    }

def backtest(df, positions, cost_bps=0.0005, ann_factor=365):
    """Run backtest with transaction costs."""
    ret = df['close'].pct_change()
    trades = positions.diff().abs()
    strat_ret = positions * ret - trades * cost_bps
    n_trades = int((trades > 0).sum())
    time_in = (positions > 0).sum() / len(positions) * 100
    metrics = compute_metrics(strat_ret, ann_factor)
    metrics['n_trades'] = n_trades
    metrics['time_in'] = time_in
    equity = (1 + strat_ret).cumprod()
    return metrics, equity

# ============================================================
# MAIN: RUN ALL STRATEGIES
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CANTONESE CAT STRATEGY BACKTEST SUITE")
    print("=" * 80)
    
    # Fetch data
    start = "2024-01-01"
    end = "2026-08-12"
    
    btc_1d = fetch_candles("BTC/USD", "1d", start, end)
    btc_4h = fetch_candles("BTC/USD", "4h", start, end)
    xlm_1d = fetch_candles("XLM/USD", "1d", start, end)
    spy_1d = fetch_candles("SPY", "1d", start, end)
    
    print(f"\nBTC/USD 1d: {len(btc_1d)} candles")
    print(f"BTC/USD 4h: {len(btc_4h)} candles")
    print(f"XLM/USD 1d: {len(xlm_1d)} candles")
    print(f"SPY 1d: {len(spy_1d)} candles")
    
    # Run strategies on BTC daily
    print("\n" + "=" * 80)
    print("STRATEGY RESULTS — BTC/USD Daily")
    print("=" * 80)
    
    strategies = {}
    
    # Strategy 1: Ichimoku Tenkan Ride
    pos1 = strategy_1_ichimoku_tenkan_ride(btc_1d)
    strategies['S1: Ichimoku Tenkan Ride'] = backtest(btc_1d, pos1, ann_factor=365)
    
    # Strategy 2: BB Squeeze Breakout
    pos2 = strategy_2_bb_squeeze_breakout(btc_1d)
    strategies['S2: BB Squeeze Breakout'] = backtest(btc_1d, pos2, ann_factor=365)
    
    # Strategy 3: RSI Divergence
    pos3 = strategy_3_rsi_divergence(btc_1d)
    strategies['S3: RSI Divergence'] = backtest(btc_1d, pos3, ann_factor=365)
    
    # Strategy 4: Fibonacci Breakout
    pos4 = strategy_4_fib_breakout(btc_1d)
    strategies['S4: Fib 0.786 Breakout'] = backtest(btc_1d, pos4, ann_factor=365)
    
    # Strategy 5: Volume Decline + Support
    pos5 = strategy_5_volume_decline_support(btc_1d)
    strategies['S5: Vol Decline + Support'] = backtest(btc_1d, pos5, ann_factor=365)
    
    # Strategy 6: Wyckoff Accumulation
    pos6 = strategy_6_wyckoff_accumulation(btc_1d)
    strategies['S6: Wyckoff Accumulation'] = backtest(btc_1d, pos6, ann_factor=365)
    
    # Strategy 7: Cross-Asset BB Squeeze
    pos7 = strategy_7_cross_asset_bb_squeeze(btc_1d, spy_1d)
    strategies['S7: Cross-Asset BB Squeeze'] = backtest(btc_1d, pos7, ann_factor=365)
    
    # Strategy 9: Order Block Support
    pos9 = strategy_9_order_block(btc_1d)
    strategies['S9: Order Block Support'] = backtest(btc_1d, pos9, ann_factor=365)
    
    # Strategy 10: Supertrend + Cloud Flip
    pos10 = strategy_10_supertrend_cloud_flip(btc_1d)
    strategies['S10: Supertrend + Cloud Flip'] = backtest(btc_1d, pos10, ann_factor=365)
    
    # Buy & Hold
    bh_pos = pd.Series(1.0, index=btc_1d.index)
    strategies['Buy & Hold'] = backtest(btc_1d, bh_pos, ann_factor=365)
    
    # Print results table
    print(f"\n{'Strategy':<30} {'TotRet%':>8} {'PaRet%':>8} {'Vol%':>8} {'Sharpe':>7} {'Sortino':>7} {'MaxDD%':>8} {'Calmar':>7} {'Trades':>7} {'InMkt%':>7}")
    print("-" * 110)
    
    for name, (m, eq) in sorted(strategies.items(), key=lambda x: x[1][0]['sharpe'], reverse=True):
        print(f"{name:<30} {m['total_ret']:>8.1f} {m['pa_ret']:>8.1f} {m['vol']:>8.1f} {m['sharpe']:>7.2f} {m['sortino']:>7.2f} {m['max_dd']:>8.1f} {m['calmar']:>7.2f} {m['n_trades']:>7} {m['time_in']:>7.1f}")
    
    # Strategy 8: MTF Ichimoku (needs multi-TF data)
    print("\n" + "=" * 80)
    print("STRATEGY 8: MULTI-TIMEFRAME ICHIMOKU (1h + 4h)")
    print("=" * 80)
    
    btc_mtf = fetch_multi_tf("BTC/USD", "2025-01-01", "2026-08-12")
    if '1h' in btc_mtf and '4h' in btc_mtf:
        pos8 = strategy_8_mtf_ichimoku(btc_mtf)
        m8, eq8 = backtest(btc_mtf['1h'], pos8, ann_factor=365*24)
        print(f"  TotRet%: {m8['total_ret']:.1f} | PaRet%: {m8['pa_ret']:.1f} | Sharpe: {m8['sharpe']:.2f} | MaxDD%: {m8['max_dd']:.1f} | Trades: {m8['n_trades']} | InMkt%: {m8['time_in']:.1f}")
    
    # Run on XLM too
    print("\n" + "=" * 80)
    print("STRATEGY RESULTS — XLM/USD Daily")
    print("=" * 80)
    
    xlm_strategies = {}
    
    pos1x = strategy_1_ichimoku_tenkan_ride(xlm_1d)
    xlm_strategies['S1: Ichimoku Tenkan Ride'] = backtest(xlm_1d, pos1x, ann_factor=365)
    
    pos2x = strategy_2_bb_squeeze_breakout(xlm_1d)
    xlm_strategies['S2: BB Squeeze Breakout'] = backtest(xlm_1d, pos2x, ann_factor=365)
    
    pos4x = strategy_4_fib_breakout(xlm_1d)
    xlm_strategies['S4: Fib 0.786 Breakout'] = backtest(xlm_1d, pos4x, ann_factor=365)
    
    pos5x = strategy_5_volume_decline_support(xlm_1d)
    xlm_strategies['S5: Vol Decline + Support'] = backtest(xlm_1d, pos5x, ann_factor=365)
    
    pos6x = strategy_6_wyckoff_accumulation(xlm_1d)
    xlm_strategies['S6: Wyckoff Accumulation'] = backtest(xlm_1d, pos6x, ann_factor=365)
    
    pos9x = strategy_9_order_block(xlm_1d)
    xlm_strategies['S9: Order Block Support'] = backtest(xlm_1d, pos9x, ann_factor=365)
    
    pos10x = strategy_10_supertrend_cloud_flip(xlm_1d)
    xlm_strategies['S10: Supertrend + Cloud Flip'] = backtest(xlm_1d, pos10x, ann_factor=365)
    
    bh_pos_x = pd.Series(1.0, index=xlm_1d.index)
    xlm_strategies['Buy & Hold'] = backtest(xlm_1d, bh_pos_x, ann_factor=365)
    
    print(f"\n{'Strategy':<30} {'TotRet%':>8} {'PaRet%':>8} {'Vol%':>8} {'Sharpe':>7} {'Sortino':>7} {'MaxDD%':>8} {'Calmar':>7} {'Trades':>7} {'InMkt%':>7}")
    print("-" * 110)
    
    for name, (m, eq) in sorted(xlm_strategies.items(), key=lambda x: x[1][0]['sharpe'], reverse=True):
        print(f"{name:<30} {m['total_ret']:>8.1f} {m['pa_ret']:>8.1f} {m['vol']:>8.1f} {m['sharpe']:>7.2f} {m['sortino']:>7.2f} {m['max_dd']:>8.1f} {m['calmar']:>7.2f} {m['n_trades']:>7} {m['time_in']:>7.1f}")
    
    # Plot
    print("\n" + "=" * 80)
    print("PLOTTING...")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # BTC equity curves
    ax = axes[0]
    for name, (m, eq) in sorted(strategies.items(), key=lambda x: x[1][0]['sharpe'], reverse=True)[:5]:
        ax.plot(eq.index, eq.values, label=f"{name} (Shp={m['sharpe']:.2f})", linewidth=1.5)
    ax.set_yscale('log')
    ax.set_title('BTC/USD — Top 5 Cantonese Cat Strategies (2024-2026)', fontsize=13)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Cumulative Return (log scale)')
    
    # XLM equity curves
    ax = axes[1]
    for name, (m, eq) in sorted(xlm_strategies.items(), key=lambda x: x[1][0]['sharpe'], reverse=True)[:5]:
        ax.plot(eq.index, eq.values, label=f"{name} (Shp={m['sharpe']:.2f})", linewidth=1.5)
    ax.set_yscale('log')
    ax.set_title('XLM/USD — Top 5 Cantonese Cat Strategies (2024-2026)', fontsize=13)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('Cumulative Return (log scale)')
    
    plt.tight_layout()
    plt.savefig('/tmp/cantonese_cat_strategies.png', dpi=150, bbox_inches='tight')
    print("Chart saved: /tmp/cantonese_cat_strategies.png")
    
    print("\nDone.")
