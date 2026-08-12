# Regime Switch Master — Crypto Trading Bot (v2)

A multi-regime perpetual futures trading bot that detects market conditions and switches between strategies accordingly. Derived from technical analysis methodology published by [@cantonmeow](https://x.com/cantonmeow) (Cantonese Cat).

## How It Works

The bot classifies the market into regimes every daily candle close and activates the matching strategy. **v2 is long-only** — shorting was removed after backtesting showed it loses money on crypto daily (see [Backtest Results](#backtest-results)).

| Regime | % of Time | Strategy | Signal |
|---|---|---|---|
| **TREND\_UP** | 17-27% | S1: Ichimoku Tenkan Ride | LONG |
| **TREND\_DOWN** | 24-31% | Flat | — |
| **CHOP** (BB squeeze) | 14-15% | S18: Keltner+Squeeze | LONG (bounce off KC lower) |
| **OVERSOLD** | 0.4-0.8% | S11: Keltner Bounce | LONG |
| **NEUTRAL** | 33-38% | Flat | — |

### Regime Detection

```
TREND_UP:      price > Tenkan(9) > Kijun(26) AND price > cloud_top = max(Senkou A, Senkou B)
TREND_DOWN:    price < Tenkan(9) < Kijun(26) AND price < Senkou A  (detected but NOT shorted)
CHOP:          BB bandwidth in bottom 20th percentile of last 50 periods
OVERSOLD:      price < lower BB AND RSI(14) < 35 (and not in trend down)
NEUTRAL:       none of the above
```

### Strategy Details

**S1: Ichimoku Tenkan Ride** (TREND\_UP)
- Entry: price > Tenkan-sen > Kijun-sen AND price above full cloud (Senkou A & B)
- Exit: any condition breaks (regime changes to non-TREND_UP)
- No TP/SL — pure trend following

**S18: Keltner+Squeeze** (CHOP)
- Entry: was in squeeze recently (BB BW < 25th pctile in last 15 bars) + price dips below Keltner lower + closes back above
- Exit: price reaches BB upper (take profit)
- Stop: price < KC lower × 0.98

**S11: Keltner Bounce** (OVERSOLD)
- Entry: squeeze context + price dips below Keltner lower + closes back above
- Exit: price reaches Keltner mid (mean reversion target)
- Stop: price < KC lower × 0.98

### What Changed in v2

| Change | v1 | v2 |
|---|---|---|
| Shorting | S17 + S14 (shorts) | Removed — loses money on crypto daily |
| Ichimoku cloud | Senkou A only | Full cloud: max(Senkou A, Senkou B) |
| S18 exit | KC mid (too early) | BB upper (captures more profit) |
| S18 entry | Single bar check | Rolling 15-bar squeeze context |
| Position limit | Unlimited | Max 3 concurrent positions |
| Stale shorts | — | Auto-closes v1 shorts on startup |

## Backtest Results

### v1 vs v2 Comparison (2020-2026, daily candles)

| Asset | v1 Sharpe | v2 Sharpe | v1 Return | v2 Return | v1 Trades | v2 Trades |
|---|---|---|---|---|---|---|
| XLM/USD | 0.08 | 0.48 | +46% | +496% | 469 | 194 |
| DOGE/USD | 0.41 | 0.56 | +696% | +1,015% | 431 | 176 |
| SOL/USD | 1.32 | 2.08 | +7,579% | +15,797% | 338 | 153 |

### Walk-Forward Validation (12mo IS → 6mo OOS)

| Strategy | DOGE Comp OOS | DOGE Win Rate | SOL Comp OOS | SOL Win Rate |
|---|---|---|---|---|
| S1 alone | +364% | 80% | +363% | 80% |
| **Bot v2 (long-only)** | **+587%** | **100%** | **+274%** | **80%** |

### Why Shorting Was Removed

S17 (Ichimoku short) alone returned **-66% on XLM, -38% DOGE, -70% SOL** over 2020-2026. Crypto's structural upward bias makes shorting negative expected value on daily timeframes, even with regime detection. The bot was shorting ~47% of the time, dragging Sharpe from 0.48 (long-only) down to 0.08.

## Installation

```bash
git clone git@github.com:dobrinsm/s1-ichimoku-bot.git
cd s1-ichimoku-bot
pip install -r requirements.txt
```

## Configuration

Create a `.env` file (see `.env.example`):

```env
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
BINANCE_TESTNET=true
TELEGRAM_BOT_TOKEN=your_bot_token  # optional
```

### Binance Futures Testnet

1. Go to [Binance Futures Testnet](https://testnet.binancefuture.com)
2. Create account, generate API keys
3. Fund with fake USDT

## Usage

```bash
python bot.py
```

The bot will:
1. Connect to Binance Futures
2. Set 3x leverage on XLMUSDT, DOGEUSDT, SOLUSDT
3. Auto-close any stale short positions from v1
4. Detect regime on each daily candle close
5. Enter LONG based on the active regime strategy
6. Exit when regime changes or take-profit target hit
7. Send Telegram alerts on every trade

### Config Options

Edit the config section at the top of `bot.py`:

```python
SYMBOLS = ["XLMUSDT", "DOGEUSDT", "SOLUSDT"]
TIMEFRAME = "1d"
LEVERAGE = 3
CHECK_INTERVAL = 3600  # seconds between checks
```

## Risk Management

- 3x leverage maximum
- 10% of portfolio per position (margin)
- Max 3 concurrent positions (30% balance utilization)
- No shorting — long/flat only
- No explicit TP/SL — exits are signal-based (regime change or mean reversion target)

## Project Structure

```
s1-ichimoku-bot/
├── bot.py                        # Regime Switch Master bot (v2)
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
├── README.md
└── backtests/
    ├── cantonese_cat_strategies.py   # 10 strategies from @cantonmeow
    ├── cantonese_cat_walkforward.py  # Walk-forward validation
    ├── s1_s2_combined.py             # S1+S2 combination test
    └── regime_switching.py           # Regime-switching backtest
```

## Origin

Strategies extracted from 100+ tweets by [@cantonmeow](https://x.com/cantonmeow) (Cantonese Cat, ~91K followers):
- **Grok/xAI API** to scrape tweets (April 2024 – August 2026)
- **Vision analysis** on 10 chart images for exact indicator parameters
- **10 strategies** coded and backtested
- **Regime Switch Master** combines strategies into one adaptive system

## Disclaimer

Not financial advice. Testnet trading with fake money. Past performance ≠ future results. Crypto trading carries significant risk.

## License

MIT
