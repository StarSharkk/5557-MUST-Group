# Multi-stock parameter scan summary

This report includes every live-data run, including negative returns and zero-trade configurations.
The label formula, 2 bps-per-side cost, position sizing, and data-window rule were fixed before scanning.
Signals execute one completed bar after they are generated; open positions are force-closed at the window end.

## Pre-registered stability rule

A change is called stable only if it improves Sharpe and profit factor on at least 3 of 4 tickers, has positive pooled-PF evidence, and has no zero-trade ticker. Changes improving fewer tickers are retained but marked idiosyncratic/inconclusive.
Profit factor cells are reported raw; pooled profit factor aggregates gross gains and gross losses rather than averaging infinity values.

## Data and limitations

Run timestamp: `2026-07-18T06:00:15.489098+00:00`; source: `Yahoo Finance via yfinance`; requested period: `60d`; interval: `5m`.
Yahoo Finance does not provide historical news snapshots, so the scan freezes the Mini TradingAgents news input at neutral (0.0). This prevents current headlines from being leaked into historical bars; it is a limitation, not a positive-result adjustment.
The recent 30 calendar days are evaluated after a 60-day download supplies causal indicator/model pre-roll. No fallback/demo rows are included.

## Configurations with majority improvement

- `ML Classifier` / `ml_xgboost_b0.60_s0.48`: both-improved `4/4`, Sharpe-improved `4/4`, PF-improved `4/4`, pooled PF `1.1831215878241084`.
- `ML Classifier` / `risk_sl1_tp4`: both-improved `3/4`, Sharpe-improved `3/4`, PF-improved `3/4`, pooled PF `0.9218268849963789`.

## Window consistency

- `evaluation_early_third`: `1` of `60` configurations met the same 3/4 majority rule; inspect `parameter_window_summary.csv` for the complete cross-window matrix.
- `evaluation_full`: `2` of `60` configurations met the same 3/4 majority rule; inspect `parameter_window_summary.csv` for the complete cross-window matrix.
- `evaluation_late_third`: `1` of `60` configurations met the same 3/4 majority rule; inspect `parameter_window_summary.csv` for the complete cross-window matrix.
- `evaluation_middle_third`: `2` of `60` configurations met the same 3/4 majority rule; inspect `parameter_window_summary.csv` for the complete cross-window matrix.

## Changes that are not broadly reliable

Configurations with fewer than 3 of 4 tickers improving both Sharpe and profit factor are retained in the matrix but are treated as idiosyncratic/inconclusive. A positive result on only one or two stocks, or one driven by very few trades, is not used as the selected methodology.

The complete parameter-by-stock matrix is in `parameter_stock_matrix.csv`; the long table, fold diagnostics, frozen OHLCV files, and all completed trades are retained alongside this summary.