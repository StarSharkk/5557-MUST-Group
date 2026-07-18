# AI Intraday Quant Trading Simulator

An AI-powered finance application for intraday trading education. Users select a ticker, time range, candle interval, and strategy. The app loads minute-level OHLCV data, calculates short-term indicators, predicts short-horizon price direction, generates Buy/Sell/Hold signals, runs a backtest, and visualises trading performance.

## Features

- Tickers: AAPL, TSLA, NVDA, CBA.AX
- Time ranges: recent 1 month or 3 months
- Candles: 1-minute or 5-minute
- Strategies:
  - Momentum strategy
  - Mean reversion strategy
  - ML classifier
  - News sentiment strategy
- Indicators:
  - VWAP
  - RSI
  - MACD
  - Bollinger Bands
  - Short-term volatility
  - Volume spike
  - Moving average crossover
- ML prediction:
  - Probability of price increasing over the next 5 or 15 minutes
  - Random Forest, Logistic Regression, and optional XGBoost
- Backtesting:
  - Trade log with entry price, exit price, profit/loss, exit reason
  - Stop-loss, take-profit, and position sizing
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
