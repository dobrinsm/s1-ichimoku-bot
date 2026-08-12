# Regime Switch Master — Crypto Trading Bot

A multi-regime perpetual futures trading bot that detects market conditions and switches between strategies accordingly. Derived from technical analysis methodology published by [@cantonmeow](https://x.com/cantonmeow) (Cantonese Cat).

## How It Works

The bot classifies the market into regimes every candle close and activates the matching strategy:

| Regime | % of Time | Strategy | Signal | CC Origin |
|---|---|---|---|---|
| **TREND\_UP** | 17-27% | S1: Ichimoku Tenkan Ride | LONG | "Riding Tenkan up" |
| **TREND\_DOWN** | 24-31% | S17: Ichimoku Short | SHORT | "Bearish Tenkan-Kijun cross" |
| **CHOP** (BB squeeze) | 14-15% | S18: Keltner+Squeeze | LONG (bounce) | "BB squeezing + retrace to mean" |
| **OVEREXTENDED** | 0.3-0.8% | S14: Overextended Fade | SHORT | "Overextended above upper BB" |
| **OVERSOLD** | 0.4-0.8% | S11: Keltner Bounce | LONG | "Retraced back to the mean" |
| **NEUTRAL** | 33-38% | Flat | — | — |

### Regime Detection

```
TREND_UP:      price > Tenkan(9) > Kijun(26) AND price > Senkou Span A (cloud)
TREND_DOWN:    price < Tenkan(9) < Kijun(26) AND price < Senkou Span A (cloud)
CHOP:          BB bandwidth in bottom 20th percentile of last 50 periods
OVEREXTENDED:  price > upper BB + 0.5 std dev (and not in trend up)
OVERSOLD:      price < lower BB AND RSI(14) < 35 (and not in trend down)
NEUTRAL:       none of the above
```

### Strategy Details

**S1: Ichimoku Tenkan Ride** (TREND\_UP)
- Entry: price > Tenkan-sen > Kijun-sen AND price above cloud
- Exit: any condition breaks
- No TP/SL — pure trend following

**S17: Ichimoku Short** (TREND\_DOWN)
- Entry: price < Tenkan-sen < Kijun-sen AND price below cloud
- Exit: any condition breaks
- Mirror of S1

**S18: Keltner+Squeeze** (CHOP)
- Entry: price dips below Keltner lower band, closes back above it
- Exit: price reaches Keltner mid (mean reversion target)
- Hard stop: price < KC lower - 2%

**S14: Overextended Fade** (OVEREXTENDED)
- Entry: price > 1 std dev above upper Bollinger Band
- Exit: price reverts to mid BB

**S11: Keltner Bounce** (OVERSOLD)
- Entry: price dips below Keltner lower, closes back above
- Exit: price reaches Keltner mid

## Backtest Results

### Walk-Forward Validation (12mo IS → 6mo OOS, rolling)

| Strategy | XLM Comp OOS | DOGE Comp OOS | SOL Comp OOS | Win Rate |
|---|---|---|---|---|
| S1 alone | +157% | +197% | +39% | 50-75% |
| **Regime Switch Master** | **+1,506%** | **+3,674%** | **+332%** | **75-100%** |

The Regime Switch Master is **8-19x better** than S1 alone on compound OOS returns, with 75-100% walk-forward win rate.

### Full-Sample Results (2019-2026, Daily)

| Asset | Regime Switch Return | Sharpe | Max Drawdown |
|---|---|---|---|
| XLM/USD | +1,019,357% | 4.43 | -38.3% |
| DOGE/USD | +20,689,951% | 5.72 | -30.8% |
| SOL/USD | +31,775,565% | 14.10 | -32.1% |

### Regime Distribution

| Regime | XLM | DOGE | SOL |
|---|---|---|---|
| TREND_UP | 17.8% | 16.3% | 26.8% |
| TREND_DOWN | 29.5% | 30.5% | 24.0% |
| CHOP | 14.7% | 14.3% | 15.1% |
| OVEREXTENDED | 0.8% | 0.8% | 0.3% |
| OVERSOLD | 0.5% | 0.4% | 0.8% |
| NEUTRAL | 36.9% | 38.0% | 33.0% |

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
3. Detect regime on each daily candle close
4. Enter LONG/SHORT based on the active regime strategy
5. Exit or flip positions when regime changes
6. Send Telegram alerts on every trade

### Config Options

Edit the config section at the top of `bot.py`:

```python
SYMBOLS = ["XLMUSDT", "DOGEUSDT", "SOLUSDT"]
TIMEFRAME = "1d"
LEVERAGE = 3
CHECK_INTERVAL = 300  # seconds between checks
```

## Risk Management

- 3x leverage maximum
- 10% of portfolio per position (margin)
- Max 1 position per asset
- Position flips (long→short, short→long) when regime changes
- No explicit TP/SL — exits are signal-based

## Project Structure

```
s1-ichimoku-bot/
├── bot.py                        # Regime Switch Master bot
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
- **Regime Switch Master** combines 5 strategies into one adaptive system

## Disclaimer

Not financial advice. Testnet trading with fake money. Past performance ≠ future results. Crypto trading carries significant risk.

## License

MIT
