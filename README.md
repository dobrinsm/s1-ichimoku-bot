# S1 Ichimoku Tenkan Ride — Crypto Trading Bot

A trend-following perpetual futures trading bot based on the Ichimoku Kinko Hyo indicator, derived from technical analysis methodology published by [@cantonmeow](https://x.com/cantonmeow) (Cantonese Cat).

## Strategy

**Entry (LONG):** All three conditions must be true on the daily candle close:
1. Price > Tenkan-sen (9-period conversion line)
2. Tenkan-sen > Kijun-sen (26-period base line)
3. Price > Senkou Span A (above the Ichimoku cloud)

**Exit:** Any condition breaks:
1. Price < Tenkan-sen, OR
2. Tenkan-sen < Kijun-sen, OR
3. Price < Senkou Span A (below cloud)

**No take-profit or stop-loss** — pure trend following. The bot rides the trend until the Ichimoku signal breaks.

## Backtest Results

### Full-Sample (2019–2026, Daily Candles)

| Asset | Total Return | Sharpe | Max Drawdown | Trades | Time in Market |
|---|---|---|---|---|---|
| XLM/USD | +4,511,697% | 4.80 | -20.1% | 178 | 17.8% |
| DOGE/USD | +324,546,346% | 4.70 | -25.2% | 184 | 16.3% |
| SOL/USD | +41,397,209% | 16.48 | -20.9% | 139 | 26.8% |

### Walk-Forward Validation (12mo IS → 6mo OOS, rolling)

| Asset | Avg OOS Sharpe | Compound OOS Return | Win Rate | Sharpe > 0.5 |
|---|---|---|---|---|
| XLM/USD | 8.14 | +7,642% | 83% (5/6) | 5/6 |
| DOGE/USD | 8.16 | +8,893% | 100% (6/6) | 6/6 |
| SOL/USD | 4.18 | +416% | 100% (4/4) | 4/4 |

### Comparison vs Other Strategies

| Strategy | XLM Sharpe | Source |
|---|---|---|
| **S1 Ichimoku Tenkan Ride** | **1.35** (full sample) | This bot |
| EMA(20/100) Crossover | 0.99 | Baseline |
| EMA(20/100) Multi-Timeframe | 0.81 | Phase 3 |
| MACE 4-Layer Hybrid | 1.12 | Phase 5 |
| BB Squeeze Breakout | 1.85 (overfit) | Phase 6 |

S1 had the best walk-forward performance — 83-100% OOS win rate across all three assets.

## Installation

```bash
git clone git@github.com:dobrinsm/s1-ichimoku-bot.git
cd s1-ichimoku-bot
pip install -r requirements.txt
```

## Configuration

Create a `.env` file:

```env
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
BINANCE_TESTNET=true
TELEGRAM_BOT_TOKEN=your_bot_token  # optional, for alerts
```

### Binance Testnet Setup

1. Go to [Binance Futures Testnet](https://testnet.binancefuture.com)
2. Create an account and generate API keys
3. Fund your testnet account with fake USDT

## Usage

```bash
python bot.py
```

The bot will:
1. Connect to Binance Futures testnet
2. Set 3x leverage on XLMUSDT, DOGEUSDT, SOLUSDT
3. Check daily candle signals every 5 minutes
4. Enter LONG when Ichimoku conditions are met
5. Exit when any condition breaks
6. Send Telegram alerts on every trade

### Configuration Options

Edit the config section at the top of `bot.py`:

```python
SYMBOLS = ["XLMUSDT", "DOGEUSDT", "SOLUSDT"]  # Trading pairs
TIMEFRAME = "1d"          # Candle timeframe
LEVERAGE = 3              # Max leverage
RISK_PER_TRADE = 0.01     # 1% of portfolio per trade
CHECK_INTERVAL = 300      # Signal check interval (seconds)
```

## Risk Management

- **3x leverage** maximum
- **10% of portfolio** per position (margin)
- **Max 3 concurrent positions** (one per asset)
- **No stop loss** — exits are purely signal-based
- **Daily timeframe** — low frequency, ~1 trade per week per asset

## How It Works

```
┌─────────────────────────────────────────────────┐
│                 Binance Futures                  │
│                                                  │
│  ┌──────────┐    ┌──────────────┐               │
│  │  Klines  │───▶│  Ichimoku    │               │
│  │  (1d)    │    │  Tenkan(9)   │               │
│  └──────────┘    │  Kijun(26)   │               │
│                  │  SenkouA(52) │               │
│                  └──────┬───────┘               │
│                         │                        │
│                    ┌────▼────┐                   │
│                    │ Signal? │                   │
│                    └────┬────┘                   │
│              ┌─────────┼─────────┐               │
│              ▼         ▼         ▼               │
│           LONG      EXIT      FLAT               │
│              │         │                         │
│         ┌────▼────┐ ┌──▼──────┐                  │
│         │  BUY    │ │  SELL   │                  │
│         │  ORDER  │ │  ORDER  │                  │
│         └────┬────┘ └──┬──────┘                  │
│              │         │                         │
│              ▼         ▼                         │
│         ┌──────────────────┐                     │
│         │ Telegram Alert   │                     │
│         │ + Trade Log      │                     │
│         └──────────────────┘                     │
└─────────────────────────────────────────────────┘
```

## Project Structure

```
s1-ichimoku-bot/
├── bot.py              # Main trading bot
├── requirements.txt    # Python dependencies
├── .env.example        # Example environment config
├── .gitignore
└── README.md
```

## Backtest Scripts

The full backtest suite (10 strategies, walk-forward validation, multi-asset testing) is available in the [backtests](backtests/) directory:

- `cantonese_cat_strategies.py` — All 10 strategies extracted from @cantonmeow
- `cantonese_cat_walkforward.py` — Walk-forward validation suite
- `s1_s2_combined.py` — S1+S2 combination backtest (S1 alone wins)

## Origin

The strategy was extracted from 100+ tweets by [@cantonmeow](https://x.com/cantonmeow) (Cantonese Cat), a market analyst with ~91K followers. The analysis used:

- **Grok/xAI API** to scrape tweets (April 2024 – August 2026)
- **Vision analysis** on 10 chart images to extract exact indicator parameters
- **10 strategies** coded and backtested
- **Walk-forward validation** on the top 3 candidates
- **S1 (Ichimoku Tenkan Ride)** emerged as the winner with 83-100% OOS win rate

## Disclaimer

This is not financial advice. This bot trades on a testnet with fake money. Past performance does not guarantee future results. Crypto trading carries significant risk. Always do your own research.

## License

MIT
