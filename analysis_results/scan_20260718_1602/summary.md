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

## Aggregate evidence across all windows

No configuration improved both metrics on at least 3 of 4 stocks in at least 3 of 4 windows while also having a positive mean Sharpe delta across all paired ticker-window cells. Therefore no parameter change is promoted as universally stable.

## ML training-window before/after check

On the same frozen OHLCV snapshots, fixed 70/30 RF coverage was `30.0%` versus walk-forward coverage `95.6%`. The paired full-window mean Sharpe delta was `-1.378` and mean total-return delta was `-0.36%`; per-ticker values are retained in `ml_window_comparison.csv`.
- `AAPL`: coverage `30.0%` -> `95.7%`, Sharpe `1.744` -> `1.624`, PF `1.2786577960781345` -> `1.226605834668499`, trades `43` -> `60`.
- `TSLA`: coverage `30.0%` -> `95.7%`, Sharpe `-1.718` -> `-3.955`, PF `0.8513138520918748` -> `0.6807878956203913`, trades `64` -> `87`.
- `NVDA`: coverage `30.0%` -> `95.7%`, Sharpe `0.279` -> `0.316`, PF `1.0317881716069135` -> `1.0379071941866311`, trades `47` -> `58`.
- `CBA.AX`: coverage `30.0%` -> `95.3%`, Sharpe `2.669` -> `-0.525`, PF `1.945906256739532` -> `0.9087367581323308`, trades `9` -> `29`.

## Window consistency

- `evaluation_early_third`: `1` of `60` configurations met the same 3/4 majority rule; inspect `parameter_window_summary.csv` for the complete cross-window matrix.
- `evaluation_full`: `2` of `60` configurations met the same 3/4 majority rule; inspect `parameter_window_summary.csv` for the complete cross-window matrix.
- `evaluation_late_third`: `1` of `60` configurations met the same 3/4 majority rule; inspect `parameter_window_summary.csv` for the complete cross-window matrix.
- `evaluation_middle_third`: `2` of `60` configurations met the same 3/4 majority rule; inspect `parameter_window_summary.csv` for the complete cross-window matrix.

## Changes that are not broadly reliable

Configurations with fewer than 3 of 4 tickers improving both Sharpe and profit factor are retained in the matrix but are treated as idiosyncratic/inconclusive. A positive result on only one or two stocks, or one driven by very few trades, is not used as the selected methodology.

The complete parameter-by-stock matrix is in `parameter_stock_matrix.csv`; aggregate evidence is in `parameter_aggregate_summary.csv`; the ML paired comparison is in `ml_window_comparison.csv`; the long table, fold diagnostics, frozen OHLCV files, and all completed trades are retained alongside this summary.