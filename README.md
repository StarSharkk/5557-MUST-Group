# AI Intraday Quant Trading Simulator

**Live app:** https://5557-must-group.streamlit.app/

An AI-powered finance application for intraday trading education. Users select a ticker, time range, candle interval, and strategy. The app loads minute-level OHLCV data, calculates short-term indicators, predicts short-horizon price direction, generates Buy/Sell/Hold signals, runs a backtest, and visualises trading performance.

## Features

- Tickers: AAPL, TSLA, NVDA, CBA.AX
- Time ranges: recent 1 month or 3 months
- Candles: 1-minute or 5-minute
- Strategies:
  - Mini TradingAgents: a Technical Analyst, Factor Analyst, News Analyst, and Risk Manager feed a final Manager Buy/Hold/Sell decision
  - Multi-factor model: blends momentum, mean reversion, and volume/trend-flow z-scores with a volatility penalty
  - Freqtrade Sample Strategy: RSI cross conditions with TEMA and Bollinger middle-band trend guards
  - ML Classifier: trains a Random Forest, Logistic Regression, or XGBoost model on recent indicator features and trades on its predicted probability, with held-out accuracy/F1 and a feature-importance chart shown in the app
- Indicators:
  - VWAP
  - RSI
  - MACD
  - Bollinger Bands
  - Short-term volatility
  - Volume spike
  - Moving average crossover
- ML prediction:
  - Probability of price increasing over the next 5 or 15 minutes (ML Classifier strategy)
  - Random Forest, Logistic Regression, and optional XGBoost, evaluated on a chronological 70/30 train/test split
- News sentiment: keyword-based scoring of recent headlines, fed into the Mini TradingAgents News Analyst
- Backtesting:
  - Trade log with entry price, exit price, profit/loss, exit reason, and the stop-loss/take-profit used for that trade
  - Volatility-adaptive stop-loss/take-profit (scaled to recent volatility and the chosen max holding period), or fixed-percentage mode, plus position sizing
- Dashboard:
  - Total return
  - Sharpe ratio
  - Max drawdown
  - Win rate
  - Average profit per trade
  - Number of trades
  - Profit factor
  - Equity curve

## Installation

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Reproducible multi-stock scan

The ML Classifier uses a purged walk-forward loop: 200 bars of warm-up, a maximum
1,000-bar causal training window, and retraining every 200 bars. The original label
formula is unchanged, and its threshold is frozen from pre-roll history. XGBoost
receives the per-fold negative/positive class ratio as `scale_pos_weight`; folds with
one class are recorded as skipped rather than silently replaced by another model.

To produce the report matrix on live Yahoo Finance data:

```bash
python run_parameter_scan.py --output-dir analysis_results/scan_YYYYMMDD_HHMMSS
```

The scanner downloads one 60-day/5-minute snapshot per ticker, evaluates the latest
30 calendar days, and fails if any ticker falls back to demo data. It writes the
frozen OHLCV snapshots, `data_manifest.json`, `scan_runs_long.csv`,
`parameter_stock_matrix.csv`, `parameter_window_summary.csv`,
`ml_fold_diagnostics.csv`, `ml_window_comparison.csv`, `parameter_aggregate_summary.csv`, `all_trades.csv`, and `summary.md`. Every completed trade,
including losses, remains in `all_trades.csv`. Historical Yahoo news snapshots are
not available, so the scan uses an explicitly recorded neutral news score instead of
leaking current headlines into past bars.

The aggregate scan selected the Multi-factor default candidate
`momentum=0.55, mean_reversion=0.25, flow=0.20, volatility_penalty=0.20`:
it improved both Sharpe and profit factor on at least three of four tickers in
three of four sub-windows. This is a reproducible selection rule for the current
frozen sample, not a guarantee of future profitability; the original baseline and
all negative outcomes remain reported.

## Data Notes

The app uses `yfinance` for intraday OHLCV data. Yahoo Finance availability for 1-minute candles is limited, so 3-month/1-minute requests may be automatically reduced or may fall back to demo data if the provider does not return enough bars.

## Academic Integrity and AI Disclosure

This project is an educational simulator and not financial advice. It does not execute real trades. If AI coding tools are used to build or debug the project, disclose this in the written report.
