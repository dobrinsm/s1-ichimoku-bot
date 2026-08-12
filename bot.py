#!/usr/bin/env python3
"""
Regime Switch Master — Crypto Trading Bot
==========================================
Multi-regime trading bot for XLMUSDT, DOGEUSDT, SOLUSDT perpetuals on Binance Futures.

Detects market regime and activates the matching strategy:
  TREND_UP      → S1: Ichimoku Tenkan Ride (long)
  TREND_DOWN    → S17: Ichimoku Short
  CHOP          → S18: Keltner+Squeeze mean reversion
  OVEREXTENDED  → S14: Overextended fade (short)
  OVERSOLD      → S11: Keltner bounce (long)
  NEUTRAL       → Flat

Walk-forward validated: 75-100% OOS win rate across XLM, DOGE, SOL.

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
CHECK_INTERVAL = 300
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
        if amt > 0: return place_market_order(symbol, "SELL", abs(amt))
        elif amt < 0: return place_market_order(symbol, "BUY", abs(amt))
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
    high, low = df['high'], df['low']
    t = (high.rolling(TENKAN).max() + low.rolling(TENKAN).min()) / 2
    k = (high.rolling(KIJUN).max() + low.rolling(KIJUN).min()) / 2
    sa = ((t + k) / 2).shift(KIJUN)
    return t, k, sa

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
    Detect regime and return signal: 'LONG', 'SHORT', 'EXIT_LONG', 'EXIT_SHORT', 'FLAT'
    Uses the last CLOSED candle (idx=-2).
    """
    if len(df) < max(SENKOU_B, SQUEEZE_LOOKBACK, KIJUN) + 10:
        return 'FLAT', 'NEUTRAL', {}

    idx = -2  # Last closed candle

    # Indicators
    tenkan, kijun, senkou_a = compute_ichimoku(df)
    bb_mid, bb_upper, bb_lower, bb_bw = compute_bollinger(df)
    kc_mid, kc_upper, kc_lower = compute_keltner(df)
    rsi_val = compute_rsi(df)

    price = df['close'].iloc[idx]
    t = tenkan.iloc[idx]
    k = kijun.iloc[idx]
    sa = senkou_a.iloc[idx]
    r = rsi_val.iloc[idx]
    bw = bb_bw.iloc[idx]
    bw_pct = bb_bw.rolling(SQUEEZE_LOOKBACK).rank(pct=True).iloc[idx]
    kc_lo = kc_lower.iloc[idx]
    kc_md = kc_mid.iloc[idx]
    bb_up = bb_upper.iloc[idx]
    bb_lo = bb_lower.iloc[idx]
    bb_md = bb_mid.iloc[idx]

    if pd.isna(t) or pd.isna(k) or pd.isna(sa) or pd.isna(bw_pct):
        return 'FLAT', 'NEUTRAL', {}

    # Regime detection (priority order)
    trend_up = (price > t) and (t > k) and (price > sa)
    trend_dn = (price < t) and (t < k) and (price < sa)
    in_squeeze = bw_pct < (SQUEEZE_PCTILE / 100)
    bb_std_val = (bb_up - bb_md) / BB_STD
    overextended = (price > bb_up + 0.5 * bb_std_val) and not trend_up
    oversold = (price < bb_lo) and (r < 35) and not trend_dn

    if trend_up:
        regime = 'TREND_UP'
        signal = 'LONG'
    elif trend_dn:
        regime = 'TREND_DOWN'
        signal = 'SHORT'
    elif in_squeeze:
        regime = 'CHOP'
        # S18: Keltner mean reversion — dip below KC lower + close back above
        if (df['low'].iloc[idx] < kc_lo) and (price > kc_lo):
            signal = 'LONG'  # bounce off Keltner lower
        elif price >= kc_md:
            signal = 'EXIT_LONG'  # reached mean → take profit
        else:
            signal = 'FLAT'
    elif overextended:
        regime = 'OVEREXTENDED'
        signal = 'SHORT'  # fade the overextension
        if price <= bb_md:
            signal = 'EXIT_SHORT'  # reverted to mean
    elif oversold:
        regime = 'OVERSOLD'
        # S11: Keltner bounce
        if (df['low'].iloc[idx] < kc_lo) and (price > kc_lo):
            signal = 'LONG'
        elif price >= kc_md:
            signal = 'EXIT_LONG'
        else:
            signal = 'FLAT'
    else:
        regime = 'NEUTRAL'
        signal = 'FLAT'

    info = {
        'price': price, 'tenkan': t, 'kijun': k, 'senkou_a': sa,
        'rsi': r, 'bb_upper': bb_up, 'bb_lower': bb_lo, 'bb_mid': bb_md,
        'kc_lower': kc_lo, 'kc_mid': kc_md, 'bb_bw_pct': bw_pct,
        'regime': regime, 'in_squeeze': in_squeeze,
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
    print(f"Regimes: TREND_UP→S1, TREND_DOWN→S17, CHOP→S18, OVEREXTENDED→S14, OVERSOLD→S11")
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
                    continue  # Already processed

                # Current position
                pos_data = get_position(sym)
                current_pos = 0.0
                if isinstance(pos_data, list) and len(pos_data) > 0:
                    current_pos = float(pos_data[0].get('positionAmt', 0))
                has_long = current_pos > 0
                has_short = current_pos < 0

                price = info.get('price', 0)
                print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {sym} | "
                      f"regime={regime} price={price:.4f} | "
                      f"signal={signal} pos={'LONG' if has_long else 'SHORT' if has_short else 'FLAT'}")

                # === EXECUTE SIGNALS ===

                # LONG entry
                if signal == 'LONG' and not has_long and not has_short:
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
                                f"Cloud A: {info['senkou_a']:.4f}\nRSI: {info['rsi']:.1f}\n"
                                f"BB BW %ile: {info['bb_bw_pct']*100:.0f}%")
                            print(f"  >>> ENTERED LONG {sym} qty={qty} regime={regime}")

                # SHORT entry
                elif signal == 'SHORT' and not has_short and not has_long:
                    qty = calc_position_size(balance, price, symbol_infos[sym])
                    if qty > 0:
                        result = place_market_order(sym, "SELL", qty)
                        if 'error' not in result:
                            last_candle_date[sym] = candle_date
                            trade = {'timestamp': datetime.now(timezone.utc).isoformat(),
                                     'symbol': sym, 'action': 'SELL', 'qty': qty,
                                     'price': price, 'signal': 'SHORT', 'regime': regime,
                                     'info': {k: float(v) if isinstance(v, (int, float, np.floating)) else str(v)
                                              for k, v in info.items()}, 'balance': balance}
                            log_trade(trade)
                            state['positions'][sym] = trade
                            save_state(state)
                            send_telegram(
                                f"🔴 *SHORT ENTRY* ({regime})\n"
                                f"Symbol: {sym}\nPrice: ${price:.4f}\nQty: {qty}\n"
                                f"Notional: ${qty*price:.2f}\nLeverage: {LEVERAGE}x\n\n"
                                f"Tenkan: {info['tenkan']:.4f}\nKijun: {info['kijun']:.4f}\n"
                                f"Cloud A: {info['senkou_a']:.4f}\nRSI: {info['rsi']:.1f}\n"
                                f"BB Upper: {info['bb_upper']:.4f}")
                            print(f"  >>> ENTERED SHORT {sym} qty={qty} regime={regime}")

                # EXIT LONG
                elif signal in ('EXIT_LONG', 'SHORT') and has_long:
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
                        print(f"  >>> EXITED LONG {sym}")

                        # If signal is SHORT, immediately enter short
                        if signal == 'SHORT':
                            qty = calc_position_size(balance, price, symbol_infos[sym])
                            if qty > 0:
                                r2 = place_market_order(sym, "SELL", qty)
                                if 'error' not in r2:
                                    trade2 = {'timestamp': datetime.now(timezone.utc).isoformat(),
                                              'symbol': sym, 'action': 'SELL', 'qty': qty,
                                              'price': price, 'signal': 'SHORT', 'regime': regime,
                                              'balance': balance}
                                    log_trade(trade2)
                                    state['positions'][sym] = trade2
                                    save_state(state)
                                    send_telegram(
                                        f"🔴 *SHORT ENTRY* ({regime})\n"
                                        f"Symbol: {sym}\nPrice: ${price:.4f}\nQty: {qty}\n"
                                        f"Flipped from long to short")
                                    print(f"  >>> FLIPPED TO SHORT {sym} qty={qty}")

                # EXIT SHORT
                elif signal in ('EXIT_SHORT', 'LONG') and has_short:
                    result = close_position(sym)
                    if 'error' not in result:
                        last_candle_date[sym] = candle_date
                        trade = {'timestamp': datetime.now(timezone.utc).isoformat(),
                                 'symbol': sym, 'action': 'BUY', 'qty': abs(current_pos),
                                 'price': price, 'signal': 'EXIT_SHORT', 'regime': regime,
                                 'balance': balance}
                        log_trade(trade)
                        if sym in state['positions']: del state['positions'][sym]
                        save_state(state)
                        send_telegram(
                            f"🟢 *EXIT SHORT* ({regime})\n"
                            f"Symbol: {sym}\nPrice: ${price:.4f}\nQty: {abs(current_pos)}\n"
                            f"Reason: {signal}")
                        print(f"  >>> EXITED SHORT {sym}")

                        # If signal is LONG, immediately enter long
                        if signal == 'LONG':
                            qty = calc_position_size(balance, price, symbol_infos[sym])
                            if qty > 0:
                                r2 = place_market_order(sym, "BUY", qty)
                                if 'error' not in r2:
                                    trade2 = {'timestamp': datetime.now(timezone.utc).isoformat(),
                                              'symbol': sym, 'action': 'BUY', 'qty': qty,
                                              'price': price, 'signal': 'LONG', 'regime': regime,
                                              'balance': balance}
                                    log_trade(trade2)
                                    state['positions'][sym] = trade2
                                    save_state(state)
                                    send_telegram(
                                        f"🟢 *LONG ENTRY* ({regime})\n"
                                        f"Symbol: {sym}\nPrice: ${price:.4f}\nQty: {qty}\n"
                                        f"Flipped from short to long")
                                    print(f"  >>> FLIPPED TO LONG {sym} qty={qty}")

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
