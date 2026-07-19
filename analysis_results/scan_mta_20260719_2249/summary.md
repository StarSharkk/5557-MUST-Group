# Mini TradingAgents weight and Manager-threshold scan summary

This second-round scan reuses the SHA-256-verified frozen snapshots from `analysis_results/scan_20260718_1602`; it never downloads new market data.
The first-round methodology is unchanged: recent 30-day evaluation after causal pre-roll, 2 bps per side, one-bar signal delay, long-only execution, adaptive risk limits, and force-closing at the evaluation-window end.

## Frozen-data verification and two-step self-check

All four source snapshot hashes matched the first-round `data_manifest.json` before loading.
Step 1 uses the historical first-round threshold pair 0.35/-0.35 and must reproduce the archived figures. Only after Step 1 passes does Step 2 evaluate the corrected app-default baseline at 0.20/-0.15. Both steps use team weights 0.35/0.35/0.15/0.15.

| Ticker | Step 1 historical return | Step 1 Sharpe | Step 2 corrected return | Step 2 Sharpe |
|---|---:|---:|---:|---:|
| AAPL | +0.85% | 4.77 | -0.11% | -0.35 |
| TSLA | -0.19% | -0.91 | -0.23% | -0.68 |
| NVDA | -0.98% | -3.53 | -0.96% | -2.73 |
| CBA.AX | +0.60% | 4.48 | +0.07% | 0.39 |

## Threshold mislabeling in round 1

An audit of the first-round configuration builder found that Mini TradingAgents rows were assigned `MULTIFACTOR_BUY_THRESHOLD=0.35` and `MULTIFACTOR_SELL_THRESHOLD=-0.35`, although the app's real team/Manager defaults are `TEAM_BUY_THRESHOLD=0.20` and `TEAM_SELL_THRESHOLD=-0.15`. The first-round archive is retained unchanged as a historical record.

This mislabeling affects the first-round Mini TradingAgents baseline table: the archived figures are valid for the historical 0.35/-0.35 execution path, but they must not be labelled as the app-default 0.20/-0.15 baseline. Step 1 documents the historical reproduction; Step 2 records the corrected baseline used for this scan. Treating the discrepancy explicitly makes the experiment auditable and does not alter the archived files.

## Fixed news constraint

The scan news score is fixed at neutral 0.0 because historical headlines are unavailable. Therefore `news_weight` is fixed at the default 0.15 and is not treated as an explored dimension. Only the relative weights of the Technical Analyst, Factor Analyst and Risk Manager are varied.

## Pre-registered stability rule

A change is called stable only if it improves Sharpe and profit factor on at least 3 of 4 tickers, has positive pooled-PF evidence, and has no zero-trade ticker.
The same rule is re-applied independently to `evaluation_early_third`, `evaluation_middle_third`, `evaluation_late_third` and `evaluation_full`. No configuration is removed because of a negative result.

## Declared grid

Manager threshold pairs are 0.20/-0.15 (corrected baseline), 0.35/-0.35 (round-1 mislabelled pair promoted to a test configuration), and 0.50/-0.35.
The scan contains exactly 18 configurations: six team-weight profiles multiplied by three Manager buy/sell threshold pairs. Stop-loss is fixed at 1.5σ, take-profit at 3σ, factor weights at 0.40/0.35/0.25 with volatility weight 0.35, and news weight at 0.15.

## Complete full-window configuration list

| Configuration | Both improved | Pooled PF | Zero-trade stocks | Stable this window | Trade counts AAPL / TSLA / NVDA / CBA.AX |
|---|---:|---:|---:|---|---|
| `mta_baseline_b0.20_s-0.15` | 0/4 | 0.887 | 0 | FAIL | 62/51/62/68 |
| `mta_baseline_b0.35_s-0.35` | 2/4 | 1.056 | 0 | FAIL | 28/22/34/31 |
| `mta_baseline_b0.50_s-0.35` | 3/4 | 1.925 | 0 | PASS | 9/8/5/11 |
| `mta_factor_heavy_b0.20_s-0.15` | 2/4 | 0.895 | 0 | FAIL | 62/53/63/69 |
| `mta_factor_heavy_b0.35_s-0.35` | 4/4 | 1.123 | 0 | PASS | 35/31/42/36 |
| `mta_factor_heavy_b0.50_s-0.35` | 2/4 | 1.065 | 0 | FAIL | 15/10/9/16 |
| `mta_risk_heavy_b0.20_s-0.15` | 2/4 | 0.856 | 0 | FAIL | 72/65/82/72 |
| `mta_risk_heavy_b0.35_s-0.35` | 2/4 | 0.858 | 0 | FAIL | 46/36/48/48 |
| `mta_risk_heavy_b0.50_s-0.35` | 2/4 | 0.801 | 0 | FAIL | 16/14/17/19 |
| `mta_technical_factor_no_risk_b0.20_s-0.15` | 4/4 | 1.157 | 0 | PASS | 46/35/54/52 |
| `mta_technical_factor_no_risk_b0.35_s-0.35` | 3/4 | 1.121 | 0 | PASS | 19/17/15/27 |
| `mta_technical_factor_no_risk_b0.50_s-0.35` | 4/4 | 2.163 | 0 | PASS | 7/9/3/9 |
| `mta_technical_heavy_b0.20_s-0.15` | 4/4 | 1.237 | 0 | PASS | 56/36/51/59 |
| `mta_technical_heavy_b0.35_s-0.35` | 2/4 | 0.962 | 0 | FAIL | 17/16/17/23 |
| `mta_technical_heavy_b0.50_s-0.35` | 4/4 | 2.590 | 0 | PASS | 5/9/2/4 |
| `mta_uniform_b0.20_s-0.15` | 1/4 | 0.846 | 0 | FAIL | 72/65/82/72 |
| `mta_uniform_b0.35_s-0.35` | 3/4 | 0.890 | 0 | FAIL | 46/36/46/48 |
| `mta_uniform_b0.50_s-0.35` | 1/4 | 0.829 | 0 | FAIL | 15/13/15/19 |

## Sub-window pass status

| Configuration | Early | Middle | Late | Full | Stable windows |
|---|---|---|---|---|---:|
| `mta_baseline_b0.20_s-0.15` | FAIL | FAIL | FAIL | FAIL | 0/4 |
| `mta_baseline_b0.35_s-0.35` | PASS | FAIL | FAIL | FAIL | 1/4 |
| `mta_baseline_b0.50_s-0.35` | FAIL | FAIL | PASS | PASS | 2/4 |
| `mta_factor_heavy_b0.20_s-0.15` | FAIL | FAIL | FAIL | FAIL | 0/4 |
| `mta_factor_heavy_b0.35_s-0.35` | PASS | PASS | FAIL | PASS | 3/4 |
| `mta_factor_heavy_b0.50_s-0.35` | FAIL | PASS | FAIL | FAIL | 1/4 |
| `mta_risk_heavy_b0.20_s-0.15` | FAIL | FAIL | FAIL | FAIL | 0/4 |
| `mta_risk_heavy_b0.35_s-0.35` | FAIL | FAIL | FAIL | FAIL | 0/4 |
| `mta_risk_heavy_b0.50_s-0.35` | FAIL | FAIL | FAIL | FAIL | 0/4 |
| `mta_technical_factor_no_risk_b0.20_s-0.15` | PASS | FAIL | FAIL | PASS | 2/4 |
| `mta_technical_factor_no_risk_b0.35_s-0.35` | PASS | FAIL | FAIL | PASS | 2/4 |
| `mta_technical_factor_no_risk_b0.50_s-0.35` | FAIL | FAIL | FAIL | PASS | 1/4 |
| `mta_technical_heavy_b0.20_s-0.15` | PASS | PASS | FAIL | PASS | 3/4 |
| `mta_technical_heavy_b0.35_s-0.35` | FAIL | FAIL | PASS | FAIL | 1/4 |
| `mta_technical_heavy_b0.50_s-0.35` | FAIL | FAIL | FAIL | PASS | 1/4 |
| `mta_uniform_b0.20_s-0.15` | FAIL | FAIL | FAIL | FAIL | 0/4 |
| `mta_uniform_b0.35_s-0.35` | PASS | FAIL | FAIL | FAIL | 1/4 |
| `mta_uniform_b0.50_s-0.35` | FAIL | FAIL | FAIL | FAIL | 0/4 |

## Conclusion

Mini TradingAgents stable improvement across stocks: candidates listed below passed the unchanged cross-stock rule in at least three of the four repeated windows; they remain frozen-sample candidates rather than claims of future profitability.
- `mta_factor_heavy_b0.35_s-0.35`: stable in `3/4` windows; mean pooled PF `1.122`; mean Sharpe delta `1.525`.
- `mta_technical_heavy_b0.20_s-0.15`: stable in `3/4` windows; mean pooled PF `1.229`; mean Sharpe delta `1.862`.

The complete configuration-by-stock matrix is in `parameter_stock_matrix.csv`; all four sub-window rows are in `parameter_window_summary.csv`; aggregate repeated-window evidence is in `parameter_aggregate_summary.csv`; every completed trade is in `all_trades.csv`.