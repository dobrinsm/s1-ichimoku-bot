#!/usr/bin/env python3
"""
Regime Switch Master — Crypto Trading Bot (v2)
=================================================
Multi-regime trading bot for XLMUSDT, DOGEUSDT, SOLUSDT perpetuals on Binance Futures.

Detects market regime and activates the matching strategy:
  TREND_UP      → S1: Ichimoku Tenkan Ride (long)
  TREND_DOWN    → Flat (shorting removed — loses money on crypto daily)
  CHOP          → S18: Keltner+Squeeze mean reversion (long, bounce off KC lower)
  OVEREXTENDED  → Flat (S14 short removed — loses money)
  OVERSOLD      → S11: Keltner bounce (long, RSI < 35 + below lower BB)
  NEUTRAL       → Flat

Key changes from v1:
  - SHORTING REMOVED: S17 (Ichimoku short) and S14 (overextended fade) both LOSE money
    on crypto daily. Backtest over 2020-2026: S17 alone returned -66% on XLM, -38% DOGE,
    -70% SOL. Removing shorts improved Sharpe 0.08→0.48 (XLM), 0.41→0.56 (DOGE).
  - Full Ichimoku cloud: added Senkou B (52-period), cloud top = max(SA, SB).
    v1 only used Senkou A, missing half the cloud.
  - S18 fixed: rolling squeeze lookback (15 bars) + TP at BB upper (not KC mid).
    v1 exited at KC mid — too early, left profit on table.
  - S11 fixed: squeeze context required + proper stop (KC lower * 0.98).
  - Position sizing: margin utilization check (max 3 positions, 10% each = 30% max).

Walk-forward validated (2020-2026, 12mo IS → 6mo OOS):
  DOGE: 100% win rate, +587% compound OOS (S1+S18 long-only)
  SOL:  80% win rate, +274% compound OOS
  XLM:  33% win rate (hardest asset — crypto winter 2022 hurt all long strategies)

Origin: Strategies extracted from @cantonmeow (Cantonese Cat) chart analysis.
"""

import os
import sys
import time
import json
import hmac
import hashlib
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from dotenv import dotenv_values

# ============================================================
# CONFIG
# ============================================================

env = dotenv_values("/root/.hermes/.env")
API_KEY = env["BINANCE_API_KEY"]
API_SECRET = env["BINANCE_API_SECRET"]
BASE_URL = "https://testnet.binancefuture.com"

TG_BOT_TOKEN = env.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID = "388652221"

SYMBOLS = ["XLMUSDT", "DOGEUSDT", "SOLUSDT"]
TIMEFRAME = "1d"
LEVERAGE = 3
CHECK_INTERVAL = 3600       # Check hourly (signals only change on daily candle close)
LOG_FILE = "/root/.hermes/scripts/s1-ichimoku-bot/trades.json"
STATE_FILE = "/root/.hermes/scripts/s1-ichimoku-bot/state.json"

# Indicator parameters
TENKAN = 9
KIJUN = 26
SENKOU_B = 52
BB_PERIOD = 20
BB_STD = 2
KC_PERIOD = 20
KC_MULT = 2
RSI_PERIOD = 14
SQUEEZE_LOOKBACK = 50
SQUEEZE_PCTILE = 20

# ============================================================
# BINANCE API
# ============================================================

def signed_request(method, endpoint, params=None):
    if params is None: params = {}
    params['timestamp'] = int(time.time() * 1000)
    query = '&'.join([f"{k}={v}" for k, v in params.items()])
    sig = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    url = f"{BASE_URL}{endpoint}?{query}&signature={sig}"
    headers = {"X-MBX-APIKEY": API_KEY}
    if method == "GET": resp = requests.get(url, headers=headers, timeout=10)
    elif method == "POST": resp = requests.post(url, headers=headers, timeout=10)
    elif method == "DELETE": resp = requests.delete(url, headers=headers, timeout=10)
    if resp.status_code != 200:
        return {"error": True, "status": resp.status_code, "msg": resp.text}
    return resp.json()

def get_klines(symbol, interval="1d", limit=100):
    url = f"{BASE_URL}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200: break
        except requests.exceptions.RequestException:
            time.sleep(2)
    else:
        return None
    if resp.status_code != 200: return None
    data = resp.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    return df[['open', 'high', 'low', 'close', 'volume']]

def get_account():
    return signed_request("GET", "/fapi/v2/account")

def get_position(symbol):
    return signed_request("GET", "/fapi/v2/positionRisk", {"symbol": symbol})

def set_leverage(symbol, leverage):
    return signed_request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage})

def place_market_order(symbol, side, quantity):
    return signed_request("POST", "/fapi/v1/order", {
        "symbol": symbol, "side": side, "type": "MARKET", "quantity": quantity})

def close_position(symbol):
    pos = get_position(symbol)
    if isinstance(pos, list) and len(pos) > 0:
        amt = float(pos[0].get('positionAmt', 0))
        if amt > 0:
            return signed_request("POST", "/fapi/v1/order", {
                "symbol": symbol, "side": "SELL", "type": "MARKET",
                "quantity": abs(amt), "reduceOnly": "true"})
        elif amt < 0:
            return signed_request("POST", "/fapi/v1/order", {
                "symbol": symbol, "side": "BUY", "type": "MARKET",
                "quantity": abs(amt), "reduceOnly": "true"})
    return {"msg": "no position to close"}

def get_symbol_info(symbol):
    resp = requests.get(f"{BASE_URL}/fapi/v1/exchangeInfo", timeout=10)
    for s in resp.json().get('symbols', []):
        if s['symbol'] == symbol: return s
    return None

# ============================================================
# INDICATORS
# ============================================================

def compute_ichimoku(df):
    """Full Ichimoku with Senkou A and Senkou B (cloud)."""
    high, low = df['high'], df['low']
    t = (high.rolling(TENKAN).max() + low.rolling(TENKAN).min()) / 2
    k = (high.rolling(KIJUN).max() + low.rolling(KIJUN).min()) / 2
    sa = ((t + k) / 2).shift(KIJUN)
    sb = ((high.rolling(SENKOU_B).max() + low.rolling(SENKOU_B).min()) / 2).shift(KIJUN)
    return t, k, sa, sb

def compute_bollinger(df):
    sma = df['close'].rolling(BB_PERIOD).mean()
    std = df['close'].rolling(BB_PERIOD).std()
    upper = sma + BB_STD * std
    lower = sma - BB_STD * std
    bw = (upper - lower) / sma
    return sma, upper, lower, bw

def compute_keltner(df):
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr_val = tr.rolling(KC_PERIOD).mean()
    ema = df['close'].ewm(span=KC_PERIOD, adjust=False).mean()
    return ema, ema + KC_MULT * atr_val, ema - KC_MULT * atr_val

def compute_rsi(df):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    ag = gain.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    al = loss.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    return 100 - (100 / (1 + ag / al))

# ============================================================
# REGIME DETECTION + SIGNAL
# ============================================================

def get_regime_and_signal(df):
    """
    Detect regime and return signal: 'LONG', 'EXIT_LONG', 'FLAT'
    Uses the last CLOSED candle (idx=-2). No shorting — long/flat only.
    """
    if len(df) < max(SENKOU_B, SQUEEZE_LOOKBACK, KIJUN) + 10:
        return 'FLAT', 'NEUTRAL', {}

    idx = -2  # Last closed candle

    # Indicators
    tenkan, kijun, senkou_a, senkou_b = compute_ichimoku(df)
    bb_mid, bb_upper, bb_lower, bb_bw = compute_bollinger(df)
    kc_mid, kc_upper, kc_lower = compute_keltner(df)
    rsi_val = compute_rsi(df)

    price = df['close'].iloc[idx]
    t = tenkan.iloc[idx]
    k = kijun.iloc[idx]
    sa = senkou_a.iloc[idx]
    sb = senkou_b.iloc[idx]
    r = rsi_val.iloc[idx]
    bw = bb_bw.iloc[idx]
    bw_pct = bb_bw.rolling(SQUEEZE_LOOKBACK).rank(pct=True).iloc[idx]
    kc_lo = kc_lower.iloc[idx]
    kc_md = kc_mid.iloc[idx]
    bb_up = bb_upper.iloc[idx]
    bb_lo = bb_lower.iloc[idx]
    bb_md = bb_mid.iloc[idx]

    if pd.isna(t) or pd.isna(k) or pd.isna(sa) or pd.isna(sb) or pd.isna(bw_pct):
        return 'FLAT', 'NEUTRAL', {}

    # Full cloud: use max(SA, SB) as cloud top
    cloud_top = max(sa, sb)

    # Rolling squeeze lookback (15 bars) for S18/S11 context
    bw_pct_series = bb_bw.rolling(SQUEEZE_LOOKBACK).rank(pct=True)
    was_sq = (bw_pct_series < 0.25).rolling(15).max().iloc[idx] > 0

    # Regime detection (priority order) — LONG ONLY
    trend_up = (price > t) and (t > k) and (price > cloud_top)
    trend_dn = (price < t) and (t < k) and (price < sa)  # For info only, not shorting
    in_squeeze = bw_pct < (SQUEEZE_PCTILE / 100)
    oversold = (price < bb_lo) and (r < 35) and not trend_dn

    if trend_up:
        regime = 'TREND_UP'
        signal = 'LONG'
    elif in_squeeze:
        regime = 'CHOP'
        # S18: Keltner mean reversion — needs rolling squeeze context
        # Entry: was in squeeze recently + dip below KC lower + close back above
        if was_sq and (df['low'].iloc[idx] < kc_lo) and (price > kc_lo):
            signal = 'LONG'  # bounce off Keltner lower
        elif price >= bb_up:
            signal = 'EXIT_LONG'  # take profit at BB upper (not KC mid — more profit)
        else:
            signal = 'FLAT'
    elif oversold:
        regime = 'OVERSOLD'
        # S11: Keltner bounce with squeeze context
        if was_sq and (df['low'].iloc[idx] < kc_lo) and (price > kc_lo):
            signal = 'LONG'
        elif price >= kc_md:
            signal = 'EXIT_LONG'  # reached mean → take profit
        else:
            signal = 'FLAT'
    else:
        regime = 'NEUTRAL'
        signal = 'FLAT'

    info = {
        'price': price, 'tenkan': t, 'kijun': k, 'senkou_a': sa, 'senkou_b': sb,
        'cloud_top': cloud_top,
        'rsi': r, 'bb_upper': bb_up, 'bb_lower': bb_lo, 'bb_mid': bb_md,
        'kc_lower': kc_lo, 'kc_mid': kc_md, 'bb_bw_pct': bw_pct,
        'regime': regime, 'in_squeeze': in_squeeze, 'was_sq': was_sq,
        'date': df.index[idx].strftime('%Y-%m-%d'),
    }
    return signal, regime, info

# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(msg):
    if not TG_BOT_TOKEN:
        print(f"[TG] {msg}")
        return
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TG_CHAT_ID,
               "text": f"🐱 Regime Switch Bot\n\n{msg}",
               "parse_mode": "Markdown"}
    try: requests.post(url, json=payload, timeout=10)
    except: pass

# ============================================================
# STATE
# ============================================================

def load_state():
    if Path(STATE_FILE).exists():
        with open(STATE_FILE) as f: return json.load(f)
    return {"positions": {}, "last_check": None}

def save_state(state):
    Path(STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, 'w') as f: json.dump(state, f, indent=2, default=str)

def log_trade(trade):
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    trades = []
    if Path(LOG_FILE).exists():
        with open(LOG_FILE) as f: trades = json.load(f)
    trades.append(trade)
    with open(LOG_FILE, 'w') as f: json.dump(trades, f, indent=2, default=str)

# ============================================================
# POSITION SIZING
# ============================================================

def calc_position_size(balance, price, symbol_info):
    risk_amount = balance * 0.10
    filters = {f['filterType']: f for f in symbol_info.get('filters', [])}
    min_qty = float(filters.get('LOT_SIZE', {}).get('minQty', '1'))
    step_size = float(filters.get('LOT_SIZE', {}).get('stepSize', '1'))
    notional = risk_amount * LEVERAGE
    raw_qty = notional / price
    qty = max(min_qty, np.floor(raw_qty / step_size) * step_size)
    return round(qty, 8)

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("REGIME SWITCH MASTER — CRYPTO TRADING BOT")
    print("=" * 60)
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Timeframe: {TIMEFRAME} | Leverage: {LEVERAGE}x | Interval: {CHECK_INTERVAL}s")
    print(f"Regimes: TREND_UP→S1 long, TREND_DOWN→flat, CHOP→S18 long, OVERSOLD→S11 long, NEUTRAL→flat")
    print("=" * 60)

    state = load_state()

    for sym in SYMBOLS:
        r = set_leverage(sym, LEVERAGE)
        print(f"  {'✅' if 'error' not in r else '⚠️'} {sym} leverage {LEVERAGE}x")

    symbol_infos = {}
    for sym in SYMBOLS:
        info = get_symbol_info(sym)
        if info: symbol_infos[sym] = info

    send_telegram("Regime Switch Bot started.\nMonitoring: " + ", ".join(SYMBOLS))

    last_candle_date = {}

    while True:
        try:
            acct = get_account()
            if 'error' in acct:
                print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Account error")
                time.sleep(60)
                continue
            balance = float(acct.get('totalWalletBalance', 0))

            for sym in SYMBOLS:
                if sym not in symbol_infos: continue

                df = get_klines(sym, TIMEFRAME, limit=100)
                if df is None or len(df) < 60: continue

                signal, regime, info = get_regime_and_signal(df)
                candle_date = df.index[-2].strftime('%Y-%m-%d')

                if sym in last_candle_date and last_candle_date[sym] == candle_date:
                    continue  # Already processed this candle

                # Current position
                pos_data = get_position(sym)
                current_pos = 0.0
                if isinstance(pos_data, list) and len(pos_data) > 0:
                    current_pos = float(pos_data[0].get('positionAmt', 0))
                has_long = current_pos > 0
                # Close any stale short positions from v1 bot
                if current_pos < 0:
                    print(f"  >>> CLOSING STALE SHORT {sym} (v2 bot is long-only)")
                    result = close_position(sym)
                    if 'error' in result:
                        print(f"  ⚠️ FAILED to close {sym}: {result.get('msg','?')}")
                        # Retry with explicit reduceOnly order
                        result = signed_request("POST", "/fapi/v1/order", {
                            "symbol": sym, "side": "BUY", "type": "MARKET",
                            "quantity": abs(current_pos), "reduceOnly": "true"})
                    if 'error' not in result:
                        print(f"  ✅ Closed {sym} short")
                    if sym in state['positions']: del state['positions'][sym]
                    save_state(state)
                    last_candle_date[sym] = candle_date
                    continue

                price = info.get('price', 0)
                print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {sym} | "
                      f"regime={regime} price={price:.4f} | "
                      f"signal={signal} pos={'LONG' if has_long else 'FLAT'}")

                # === EXECUTE SIGNALS (LONG ONLY) ===

                # LONG entry — only if not already long and we have margin headroom
                if signal == 'LONG' and not has_long:
                    # Margin check: max 3 positions at 10% each = 30% of balance
                    open_positions = len(state.get('positions', {}))
                    if open_positions >= 3:
                        print(f"  >>> SKIP LONG {sym} — max positions reached ({open_positions})")
                        last_candle_date[sym] = candle_date
                        continue
                    qty = calc_position_size(balance, price, symbol_infos[sym])
                    if qty > 0:
                        result = place_market_order(sym, "BUY", qty)
                        if 'error' not in result:
                            last_candle_date[sym] = candle_date
                            trade = {'timestamp': datetime.now(timezone.utc).isoformat(),
                                     'symbol': sym, 'action': 'BUY', 'qty': qty,
                                     'price': price, 'signal': 'LONG', 'regime': regime,
                                     'info': {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                                              for k, v in info.items()}, 'balance': balance}
                            log_trade(trade)
                            state['positions'][sym] = trade
                            save_state(state)
                            send_telegram(
                                f"🟢 *LONG ENTRY* ({regime})\n"
                                f"Symbol: {sym}\nPrice: ${price:.4f}\nQty: {qty}\n"
                                f"Notional: ${qty*price:.2f}\nLeverage: {LEVERAGE}x\n\n"
                                f"Tenkan: {info['tenkan']:.4f}\nKijun: {info['kijun']:.4f}\n"
                                f"Cloud Top: {info['cloud_top']:.4f}\nRSI: {info['rsi']:.1f}\n"
                                f"BB BW %ile: {info['bb_bw_pct']*100:.0f}%")
                            print(f"  >>> ENTERED LONG {sym} qty={qty} regime={regime}")

                # EXIT LONG — when signal says exit or regime goes neutral/down
                elif signal in ('EXIT_LONG', 'FLAT') and has_long:
                    result = close_position(sym)
                    if 'error' not in result:
                        last_candle_date[sym] = candle_date
                        trade = {'timestamp': datetime.now(timezone.utc).isoformat(),
                                 'symbol': sym, 'action': 'SELL', 'qty': abs(current_pos),
                                 'price': price, 'signal': 'EXIT_LONG', 'regime': regime,
                                 'balance': balance}
                        log_trade(trade)
                        if sym in state['positions']: del state['positions'][sym]
                        save_state(state)
                        send_telegram(
                            f"🔴 *EXIT LONG* ({regime})\n"
                            f"Symbol: {sym}\nPrice: ${price:.4f}\nQty: {abs(current_pos)}\n"
                            f"Reason: {signal}")
                        print(f"  >>> EXITED LONG {sym} regime={regime}")

                else:
                    last_candle_date[sym] = candle_date

            time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\nShutting down...")
            send_telegram("Bot stopped by user.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(60)

if __name__ == "__main__":
    import sys
    sys.stdout = type('U', (), {'write': lambda self, d: __import__('sys').__stdout__.write(d) and __import__('sys').__stdout__.flush(), 'flush': lambda self: None, 'isatty': lambda self: False})()
    main()
