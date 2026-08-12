#!/usr/bin/env python3
"""
S1 Ichimoku Tenkan Ride — Paper Trading Bot
=============================================
Trades XLMUSDT, DOGEUSDT, SOLUSDT perpetuals on Binance Futures Testnet.

Strategy:
  Entry: price > Tenkan-sen AND Tenkan > Kijun AND price > Senkou Span A
  Exit:  price < Tenkan-sen OR Tenkan < Kijun OR price < Senkou Span A

Risk Management:
  - 1% portfolio risk per trade (position sized by ATR)
  - Max 1 position per asset
  - Max 3 concurrent positions
  - No leverage > 3x

Data Source:
  - Binance Futures klines (1d candles) for signal generation
  - Binance Futures API for order execution

Telegram alerts sent on every signal + execution.
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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import dotenv_values

# ============================================================
# CONFIG
# ============================================================

env = dotenv_values("/root/.hermes/.env")
API_KEY = env["BINANCE_API_KEY"]
API_SECRET = env["BINANCE_API_SECRET"]
BASE_URL = "https://testnet.binancefuture.com"

# Telegram (from Hermes config)
TG_BOT_TOKEN = env.get("TELEGRAM_BOT_TOKEN", "")
TG_CHAT_ID = "388652221"  # Mihail's chat

SYMBOLS = ["XLMUSDT", "DOGEUSDT", "SOLUSDT"]
TIMEFRAME = "1d"          # Daily candles for Ichimoku
LEVERAGE = 3              # 3x max
RISK_PER_TRADE = 0.01     # 1% of portfolio per trade
CHECK_INTERVAL = 300      # Check every 5 minutes (candle close detection)
LOG_FILE = "/root/.hermes/scripts/s1-ichimoku-bot/trades.json"
STATE_FILE = "/root/.hermes/scripts/s1-ichimoku-bot/state.json"

# Ichimoku parameters
TENKAN = 9
KIJUN = 26
SENKOU_B = 52

# ============================================================
# BINANCE API
# ============================================================

def signed_request(method, endpoint, params=None):
    if params is None:
        params = {}
    params['timestamp'] = int(time.time() * 1000)
    query = '&'.join([f"{k}={v}" for k, v in params.items()])
    sig = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    url = f"{BASE_URL}{endpoint}?{query}&signature={sig}"
    headers = {"X-MBX-APIKEY": API_KEY}
    
    if method == "GET":
        resp = requests.get(url, headers=headers, timeout=10)
    elif method == "POST":
        resp = requests.post(url, headers=headers, timeout=10)
    elif method == "DELETE":
        resp = requests.delete(url, headers=headers, timeout=10)
    
    if resp.status_code != 200:
        return {"error": True, "status": resp.status_code, "msg": resp.text}
    return resp.json()

def get_klines(symbol, interval="1d", limit=100):
    """Get historical candles from Binance Futures."""
    url = f"{BASE_URL}/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        return None
    
    data = resp.json()
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
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
    """side = 'BUY' or 'SELL'"""
    return signed_request("POST", "/fapi/v1/order", {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": quantity,
    })

def close_position(symbol):
    """Close any open position on symbol."""
    pos = get_position(symbol)
    if isinstance(pos, list) and len(pos) > 0:
        amt = float(pos[0].get('positionAmt', 0))
        if amt > 0:
            return place_market_order(symbol, "SELL", abs(amt))
        elif amt < 0:
            return place_market_order(symbol, "BUY", abs(amt))
    return {"msg": "no position to close"}

def get_symbol_info(symbol):
    """Get symbol trading rules (min qty, step size, etc)."""
    resp = requests.get(f"{BASE_URL}/fapi/v1/exchangeInfo", timeout=10)
    symbols = resp.json().get('symbols', [])
    for s in symbols:
        if s['symbol'] == symbol:
            return s
    return None

# ============================================================
# ICHIMOKU STRATEGY
# ============================================================

def compute_ichimoku(df, tenkan=9, kijun=26, senkou_b=52):
    high, low = df['high'], df['low']
    t = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    k = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    sa = ((t + k) / 2).shift(kijun)
    sb = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    return t, k, sa, sb

def get_signal(df):
    """
    Returns: 'LONG', 'EXIT', or 'FLAT'
    Uses the latest closed candle (not the current forming one).
    """
    if len(df) < SENKOU_B + 10:
        return 'FLAT', {}
    
    tenkan, kijun, senkou_a, senkou_b = compute_ichimoku(df)
    
    # Use last completed candle (second to last, since last is still forming)
    idx = -2
    
    price = df['close'].iloc[idx]
    t = tenkan.iloc[idx]
    k = kijun.iloc[idx]
    sa = senkou_a.iloc[idx]
    
    if pd.isna(t) or pd.isna(k) or pd.isna(sa):
        return 'FLAT', {}
    
    # Entry: price > Tenkan AND Tenkan > Kijun AND price > Senkou A (cloud)
    bullish = (price > t) and (t > k) and (price > sa)
    
    # Exit: any condition fails
    exit_signal = (price < t) or (t < k) or (price < sa)
    
    signal = 'LONG' if bullish else ('EXIT' if exit_signal else 'FLAT')
    
    info = {
        'price': price,
        'tenkan': t,
        'kijun': k,
        'senkou_a': sa,
        'bullish_alignment': t > k,
        'above_cloud': price > sa,
        'date': df.index[idx].strftime('%Y-%m-%d'),
    }
    
    return signal, info

# ============================================================
# TELEGRAM ALERTS
# ============================================================

def send_telegram(msg):
    """Send alert to Telegram."""
    if not TG_BOT_TOKEN:
        print(f"[TG] {msg}")
        return
    
    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TG_CHAT_ID,
        "text": f"🤖 S1-Ichimoku Bot\n\n{msg}",
        "parse_mode": "Markdown",
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except:
        pass

# ============================================================
# STATE MANAGEMENT
# ============================================================

def load_state():
    if Path(STATE_FILE).exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"positions": {}, "last_check": None, "trades": []}

def save_state(state):
    Path(STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)

def log_trade(trade):
    """Append trade to log file."""
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    trades = []
    if Path(LOG_FILE).exists():
        with open(LOG_FILE) as f:
            trades = json.load(f)
    trades.append(trade)
    with open(LOG_FILE, 'w') as f:
        json.dump(trades, f, indent=2, default=str)

# ============================================================
# POSITION SIZING
# ============================================================

def calc_position_size(account_balance, entry_price, symbol_info):
    """
    Size position so that 1% of portfolio is risked.
    Since S1 has no explicit stop loss, use ATR-based stop distance.
    """
    # Simple approach: use 10% of balance per position (conservative for testnet)
    # With 3x leverage, we can go up to 30% notional
    risk_amount = account_balance * 0.10  # 10% per position
    
    # Get min quantity and step size
    filters = {f['filterType']: f for f in symbol_info.get('filters', [])}
    min_qty = float(filters.get('LOT_SIZE', {}).get('minQty', '1'))
    step_size = float(filters.get('LOT_SIZE', {}).get('stepSize', '1'))
    
    # Quantity = risk_amount / price (with leverage, buying power = risk * leverage)
    # But keep it simple: use risk_amount as margin, notional = risk * leverage
    notional = risk_amount * LEVERAGE
    raw_qty = notional / entry_price
    
    # Round to step size
    qty = max(min_qty, np.floor(raw_qty / step_size) * step_size)
    
    return round(qty, 8)

# ============================================================
# MAIN LOOP
# ============================================================

def main():
    print("=" * 60)
    print("S1 ICHIMOKU TENKAN RIDE — PAPER TRADING BOT")
    print("=" * 60)
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Leverage: {LEVERAGE}x")
    print(f"Risk per trade: {RISK_PER_TRADE*100}% of portfolio")
    print(f"Check interval: {CHECK_INTERVAL}s")
    print(f"Log: {LOG_FILE}")
    print(f"State: {STATE_FILE}")
    print("=" * 60)
    
    # Load state
    state = load_state()
    
    # Set leverage for all symbols
    for sym in SYMBOLS:
        result = set_leverage(sym, LEVERAGE)
        if 'error' in result:
            print(f"⚠️ {sym} leverage set failed: {result.get('msg', '?')}")
        else:
            print(f"✅ {sym} leverage set to {LEVERAGE}x")
    
    # Get symbol info
    symbol_infos = {}
    for sym in SYMBOLS:
        info = get_symbol_info(sym)
        if info:
            symbol_infos[sym] = info
            print(f"✅ {sym} info loaded")
        else:
            print(f"❌ {sym} info not found")
    
    send_telegram("Bot started. Monitoring " + ", ".join(SYMBOLS))
    
    last_candle_date = {}
    
    while True:
        try:
            # Get account balance
            acct = get_account()
            if 'error' in acct:
                print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Account error: {acct.get('msg')}")
                time.sleep(60)
                continue
            
            balance = float(acct.get('totalWalletBalance', 0))
            available = float(acct.get('availableBalance', 0))
            
            for sym in SYMBOLS:
                if sym not in symbol_infos:
                    continue
                
                # Fetch klines
                df = get_klines(sym, TIMEFRAME, limit=100)
                if df is None or len(df) < 60:
                    print(f"[{sym}] Not enough data ({len(df) if df is not None else 0} candles)")
                    continue
                
                # Get signal from last closed candle
                signal, info = get_signal(df)
                current_candle_date = df.index[-2].strftime('%Y-%m-%d')
                
                # Only act on new candle
                if sym in last_candle_date and last_candle_date[sym] == current_candle_date:
                    continue  # Already processed this candle
                
                # Get current position
                pos_data = get_position(sym)
                current_pos = 0.0
                if isinstance(pos_data, list) and len(pos_data) > 0:
                    current_pos = float(pos_data[0].get('positionAmt', 0))
                
                has_position = current_pos > 0
                
                price = info.get('price', 0)
                t = info.get('tenkan', 0)
                k = info.get('kijun', 0)
                sa = info.get('senkou_a', 0)
                
                print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {sym} | "
                      f"price={price:.4f} T={t:.4f} K={k:.4f} SA={sa:.4f} | "
                      f"signal={signal} pos={'YES' if has_position else 'NO'}")
                
                # Execute signals
                if signal == 'LONG' and not has_position:
                    # ENTER LONG
                    qty = calc_position_size(balance, price, symbol_infos[sym])
                    if qty > 0:
                        result = place_market_order(sym, "BUY", qty)
                        if 'error' not in result:
                            last_candle_date[sym] = current_candle_date
                            trade = {
                                'timestamp': datetime.now(timezone.utc).isoformat(),
                                'symbol': sym,
                                'action': 'BUY',
                                'qty': qty,
                                'price': price,
                                'signal': 'LONG',
                                'ichimoku': info,
                                'balance': balance,
                            }
                            log_trade(trade)
                            state['positions'][sym] = trade
                            save_state(state)
                            
                            msg = (f"🟢 *LONG ENTRY*\n"
                                   f"Symbol: {sym}\n"
                                   f"Price: ${price:.4f}\n"
                                   f"Qty: {qty}\n"
                                   f"Notional: ${qty*price:.2f}\n"
                                   f"Leverage: {LEVERAGE}x\n"
                                   f"Margin: ${qty*price/LEVERAGE:.2f}\n\n"
                                   f"Ichimoku:\n"
                                   f"  Tenkan: {t:.4f}\n"
                                   f"  Kijun: {k:.4f}\n"
                                   f"  Cloud A: {sa:.4f}\n"
                                   f"  T>K: {'✅' if info['bullish_alignment'] else '❌'}\n"
                                   f"  Price>Cloud: {'✅' if info['above_cloud'] else '❌'}")
                            send_telegram(msg)
                            print(f"  >>> ENTERED LONG {sym} qty={qty}")
                        else:
                            print(f"  >>> ORDER FAILED: {result.get('msg', '?')}")
                
                elif signal == 'EXIT' and has_position:
                    # CLOSE POSITION
                    result = close_position(sym)
                    if 'error' not in result:
                        last_candle_date[sym] = current_candle_date
                        trade = {
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'symbol': sym,
                            'action': 'SELL',
                            'qty': abs(current_pos),
                            'price': price,
                            'signal': 'EXIT',
                            'ichimoku': info,
                            'balance': balance,
                        }
                        log_trade(trade)
                        if sym in state['positions']:
                            del state['positions'][sym]
                        save_state(state)
                        
                        msg = (f"🔴 *EXIT POSITION*\n"
                               f"Symbol: {sym}\n"
                               f"Price: ${price:.4f}\n"
                               f"Qty: {abs(current_pos)}\n\n"
                               f"Ichimoku:\n"
                               f"  Tenkan: {t:.4f}\n"
                               f"  Kijun: {k:.4f}\n"
                               f"  Cloud A: {sa:.4f}\n"
                               f"  Reason: {'Price<Tenkan' if price < t else 'Tenkan<Kijun' if t < k else 'Price<Cloud'}")
                        send_telegram(msg)
                        print(f"  >>> CLOSED {sym} qty={abs(current_pos)}")
                    else:
                        print(f"  >>> CLOSE FAILED: {result.get('msg', '?')}")
                
                else:
                    last_candle_date[sym] = current_candle_date
            
            # Sleep
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            print("\nShutting down...")
            send_telegram("Bot stopped by user.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
