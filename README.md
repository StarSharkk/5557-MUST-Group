# AI Intraday Quant Trading Simulator

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

## Data Notes

The app uses `yfinance` for intraday OHLCV data. Yahoo Finance availability for 1-minute candles is limited, so 3-month/1-minute requests may be automatically reduced or may fall back to demo data if the provider does not return enough bars.

## Academic Integrity and AI Disclosure

This project is an educational simulator and not financial advice. It does not execute real trades. If AI coding tools are used to build or debug the project, disclose this in the written report.
