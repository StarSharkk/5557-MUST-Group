"""Second-round Mini TradingAgents weight and Manager-threshold scan.

This scan deliberately reuses the frozen OHLCV CSV snapshots from
analysis_results/scan_20260718_1602.  It never calls yfinance.  The first
operation is a SHA-256 check, followed by a hard baseline self-check against
the first-round figures before any of the 18 configurations is evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import app
import run_parameter_scan as first_scan


SOURCE_DIR = Path("analysis_results/scan_20260718_1602")
TICKERS = ["AAPL", "TSLA", "NVDA", "CBA.AX"]
EVALUATION_DAYS = 30
INITIAL_CAPITAL = 100_000.0
POSITION_SIZE = 0.20
TRANSACTION_COST_BPS = 2.0
MAX_HOLD_BARS = 24
STOP_LOSS_MULT = 1.5
TAKE_PROFIT_MULT = 3.0
HORIZON_BARS = 1
NEWS_SCORE = 0.0
NEWS_WEIGHT = 0.15
FACTOR_WEIGHTS = {
    "momentum_weight": 0.40,
    "mean_reversion_weight": 0.35,
    "flow_weight": 0.25,
    "volatility_weight": 0.35,
}

# The requested Manager-threshold grid.  TEAM_* are the current app defaults.
THRESHOLD_GRID = [
    ("default", 0.20, -0.15),
    ("wide", 0.35, -0.25),
    ("strict", 0.50, -0.35),
]

WEIGHT_PROFILES = {
    "baseline": (0.35, 0.35, 0.15, 0.15),
    "technical_heavy": (0.55, 0.20, 0.15, 0.10),
    "factor_heavy": (0.20, 0.55, 0.15, 0.10),
    "risk_heavy": (0.28, 0.27, 0.15, 0.30),
    "technical_factor_no_risk": (0.42, 0.43, 0.15, 0.00),
    "uniform": (0.28, 0.28, 0.15, 0.29),
}

EXPECTED_BASELINE = {
    "AAPL": {"total_return": 0.0085, "sharpe": 4.77},
    "TSLA": {"total_return": -0.0019, "sharpe": -0.91},
    "NVDA": {"total_return": -0.0098, "sharpe": -3.53},
    "CBA.AX": {"total_return": 0.0060, "sharpe": 4.48},
}


def json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def make_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for profile, (technical, factor, news, risk) in WEIGHT_PROFILES.items():
        for threshold_name, buy, sell in THRESHOLD_GRID:
            config_id = f"mta_{profile}_b{buy:.2f}_s{sell:.2f}"
            configs.append(
                {
                    "strategy": app.MINI_TRADINGAGENTS_STRATEGY,
                    "family": "mta_team_weights_manager_threshold",
                    "config_id": config_id,
                    "params": {
                        **FACTOR_WEIGHTS,
                        "technical_weight": technical,
                        "factor_weight": factor,
                        "news_weight": news,
                        "risk_weight": risk,
                        "news_score": NEWS_SCORE,
                        "buy_threshold": buy,
                        "sell_threshold": sell,
                        "stop_loss_mult": STOP_LOSS_MULT,
                        "take_profit_mult": TAKE_PROFIT_MULT,
                    },
                    "profile": profile,
                    "threshold_name": threshold_name,
                }
            )
    if len(configs) != 18:
        raise RuntimeError(f"Grid construction error: expected 18 configurations, got {len(configs)}")
    return configs


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen_snapshot(ticker: str, manifest: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    filename = f"{ticker.replace('.', '_')}_ohlcv.csv"
    path = SOURCE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen snapshot: {path}")
    actual_hash = sha256_file(path)
    expected_hash = manifest["tickers_data"][ticker]["sha256"]
    if actual_hash != expected_hash:
        raise RuntimeError(f"SHA-256 mismatch for {ticker}: {actual_hash} != {expected_hash}")
    raw = pd.read_csv(path, index_col="Datetime", parse_dates=True)
    raw.index.name = "Datetime"
    required = ["open", "high", "low", "close", "volume"]
    if list(raw.columns) != required:
        raise RuntimeError(f"Unexpected columns in {path}: {list(raw.columns)}")
    raw = raw[required].copy()
    raw = raw.sort_index().dropna()
    metadata = dict(manifest["tickers_data"][ticker])
    metadata.update({"sha256_verified": True, "snapshot_file": filename})
    return raw, metadata


def make_windows(evaluation_index: pd.DatetimeIndex) -> Dict[str, pd.DatetimeIndex]:
    split = len(evaluation_index) // 3
    return {
        "evaluation_full": evaluation_index,
        "evaluation_early_third": evaluation_index[:split],
        "evaluation_middle_third": evaluation_index[split : 2 * split],
        "evaluation_late_third": evaluation_index[2 * split :],
    }


def make_team_result(data: pd.DataFrame, params: Dict[str, Any]) -> app.ModelResult:
    factors = app.build_multifactor_components(
        data,
        float(params["momentum_weight"]),
        float(params["mean_reversion_weight"]),
        float(params["flow_weight"]),
        float(params["volatility_weight"]),
    )
    technical = app.build_technical_components(data)
    team = app.build_team_components(
        data,
        factors,
        technical,
        NEWS_SCORE,
        float(params["technical_weight"]),
        float(params["factor_weight"]),
        float(params["news_weight"]),
        float(params["risk_weight"]),
    )
    return app.ModelResult(
        probability=team["manager_score"],
        latest_probability=float(team["manager_score"].iloc[-1]),
        accuracy=None,
        f1=None,
        feature_importance=pd.Series(dtype=float),
        model_name=app.MINI_TRADINGAGENTS_STRATEGY,
    )


def evaluate_one_window(
    ticker: str,
    data: pd.DataFrame,
    window_id: str,
    window_index: pd.DatetimeIndex,
    item: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    params = item["params"]
    result = make_team_result(data, params)
    signals = app.build_strategy_signals(
        data,
        app.MINI_TRADINGAGENTS_STRATEGY,
        result,
        NEWS_SCORE,
        float(params["buy_threshold"]),
        float(params["sell_threshold"]),
        float(params["momentum_weight"]),
        float(params["mean_reversion_weight"]),
        float(params["flow_weight"]),
        float(params["volatility_weight"]),
        float(params["technical_weight"]),
        float(params["factor_weight"]),
        float(params["news_weight"]),
        float(params["risk_weight"]),
    )
    stop_loss, take_profit = app.build_adaptive_risk_series(
        data,
        STOP_LOSS_MULT,
        TAKE_PROFIT_MULT,
        MAX_HOLD_BARS,
    )
    window_index = window_index.intersection(data.index)
    window_data = data.loc[window_index]
    trades, curve = app.backtest(
        window_data,
        signals.loc[window_index],
        INITIAL_CAPITAL,
        POSITION_SIZE,
        stop_loss.loc[window_index],
        take_profit.loc[window_index],
        TRANSACTION_COST_BPS,
        MAX_HOLD_BARS,
        allow_short=False,
    )
    values = first_scan.metric_row(trades, curve)
    row = {
        "window_id": window_id,
        "ticker": ticker,
        "strategy": app.MINI_TRADINGAGENTS_STRATEGY,
        "family": item["family"],
        "config_id": item["config_id"],
        "profile": item["profile"],
        "threshold_name": item["threshold_name"],
        "params_json": json_text(params),
        "data_status": "frozen_verified",
        "window_start": str(window_index.min()),
        "window_end": str(window_index.max()),
        "news_score": NEWS_SCORE,
        "news_weight_fixed": NEWS_WEIGHT,
        **values,
    }
    trade_rows: List[Dict[str, Any]] = []
    if not trades.empty:
        copy = trades.copy()
        copy.insert(0, "run_id", f"{ticker}|{window_id}|{app.MINI_TRADINGAGENTS_STRATEGY}|{item['config_id']}")
        copy.insert(1, "window_id", window_id)
        copy.insert(2, "ticker", ticker)
        copy.insert(3, "strategy", app.MINI_TRADINGAGENTS_STRATEGY)
        copy.insert(4, "family", item["family"])
        copy.insert(5, "config_id", item["config_id"])
        copy.insert(6, "profile", item["profile"])
        copy.insert(7, "threshold_name", item["threshold_name"])
        trade_rows.extend(copy.to_dict("records"))
    return row, trade_rows


def baseline_self_check(data_by_ticker: Dict[str, Tuple[pd.DataFrame, pd.DatetimeIndex, Dict[str, Any]]], configs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    baseline = next(item for item in configs if item["config_id"] == "mta_baseline_b0.20_s-0.15")
    actual: Dict[str, Dict[str, float]] = {}
    failures: List[str] = []
    for ticker, (data, evaluation_index, _) in data_by_ticker.items():
        row, _ = evaluate_one_window(ticker, data, "evaluation_full", evaluation_index, baseline)
        actual[ticker] = {"total_return": float(row["total_return"]), "sharpe": float(row["sharpe"])}
        expected = EXPECTED_BASELINE[ticker]
        if not math.isclose(actual[ticker]["total_return"], expected["total_return"], abs_tol=5e-6):
            failures.append(f"{ticker} total_return {actual[ticker]['total_return']:.8f} != {expected['total_return']:.8f}")
        if not math.isclose(actual[ticker]["sharpe"], expected["sharpe"], abs_tol=0.02):
            failures.append(f"{ticker} sharpe {actual[ticker]['sharpe']:.6f} != {expected['sharpe']:.6f}")
    if failures:
        raise RuntimeError("BASELINE SELF-CHECK FAILED; scan aborted: " + "; ".join(failures))
    return actual


def better(value: float, baseline: float) -> bool:
    if math.isinf(value) and not math.isinf(baseline):
        return True
    if math.isinf(value) and math.isinf(baseline):
        return False
    if not np.isfinite(value) or not np.isfinite(baseline):
        return False
    return value > baseline


def build_matrix(long_df: pd.DataFrame, window_id: str) -> pd.DataFrame:
    full = long_df[long_df["window_id"] == window_id].copy()
    baseline = full[full["config_id"] == "mta_baseline_b0.20_s-0.15"]
    rows: List[Dict[str, Any]] = []
    for (family, config_id, profile, threshold_name, params_json), group in full.groupby(
        ["family", "config_id", "profile", "threshold_name", "params_json"], dropna=False
    ):
        row: Dict[str, Any] = {
            "window_id": window_id,
            "strategy": app.MINI_TRADINGAGENTS_STRATEGY,
            "family": family,
            "config_id": config_id,
            "profile": profile,
            "threshold_name": threshold_name,
            "params_json": params_json,
        }
        sharpes: List[float] = []
        gross_profit = gross_loss = 0.0
        for ticker in TICKERS:
            target = group[group["ticker"] == ticker]
            base = baseline[baseline["ticker"] == ticker]
            if target.empty or base.empty:
                raise RuntimeError(f"Missing matrix cell for {config_id}/{ticker}/{window_id}")
            target_row = target.iloc[0]
            base_row = base.iloc[0]
            prefix = ticker.replace(".", "_")
            for field in ("total_return", "sharpe", "max_drawdown", "win_rate", "profit_factor", "trades"):
                row[f"{prefix}_{field}"] = target_row[field]
            row[f"{prefix}_sharpe_delta"] = float(target_row["sharpe"] - base_row["sharpe"])
            row[f"{prefix}_pf_improved"] = better(float(target_row["profit_factor"]), float(base_row["profit_factor"]))
            row[f"{prefix}_sharpe_improved"] = float(target_row["sharpe"]) > float(base_row["sharpe"])
            row[f"{prefix}_both_improved"] = bool(row[f"{prefix}_pf_improved"] and row[f"{prefix}_sharpe_improved"])
            sharpes.append(float(target_row["sharpe"]))
            gross_profit += float(target_row["gross_profit"])
            gross_loss += float(target_row["gross_loss"])
        row["mean_sharpe"] = float(np.mean(sharpes))
        row["median_sharpe"] = float(np.median(sharpes))
        row["pooled_profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else (math.inf if gross_profit > 0 else 0.0)
        row["sharpe_improved_n"] = int(sum(bool(row[f"{t.replace('.', '_')}_sharpe_improved"]) for t in TICKERS))
        row["pf_improved_n"] = int(sum(bool(row[f"{t.replace('.', '_')}_pf_improved"]) for t in TICKERS))
        row["both_improved_n"] = int(sum(bool(row[f"{t.replace('.', '_')}_both_improved"]) for t in TICKERS))
        row["zero_trade_n"] = int(sum(float(row[f"{t.replace('.', '_')}_trades"]) == 0 for t in TICKERS))
        row["stable_this_window"] = bool(row["both_improved_n"] >= 3 and row["pooled_profit_factor"] > 1 and row["zero_trade_n"] == 0)
        rows.append(row)
    return pd.DataFrame(rows)


def build_aggregate(window_matrix: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    keys = ["strategy", "family", "config_id", "profile", "threshold_name", "params_json"]
    for key_values, group in window_matrix.groupby(keys, dropna=False):
        row = dict(zip(keys, key_values))
        stable = group["stable_this_window"].astype(bool)
        row.update(
            {
                "windows": int(group["window_id"].nunique()),
                "stable_windows": int(stable.sum()),
                "mean_sharpe": float(group["mean_sharpe"].mean()),
                "mean_pooled_profit_factor": float(group["pooled_profit_factor"].replace([np.inf, -np.inf], np.nan).mean()),
                "mean_sharpe_delta_all_cells": float(group[[c for c in group.columns if c.endswith("_sharpe_delta")]].to_numpy(dtype=float).mean()),
                "cross_window_stable": bool(stable.sum() >= 3),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary(
    output_dir: Path,
    manifest: Dict[str, Any],
    configs: List[Dict[str, Any]],
    full_matrix: pd.DataFrame,
    window_matrix: pd.DataFrame,
    aggregate: pd.DataFrame,
    baseline_actual: Dict[str, Dict[str, float]],
) -> None:
    lines = [
        "# Mini TradingAgents weight and Manager-threshold scan summary",
        "",
        "This second-round scan reuses the SHA-256-verified frozen snapshots from `analysis_results/scan_20260718_1602`; it never downloads new market data.",
        "The first-round methodology is unchanged: recent 30-day evaluation after causal pre-roll, 2 bps per side, one-bar signal delay, long-only execution, adaptive risk limits, and force-closing at the evaluation-window end.",
        "",
        "## Frozen-data verification and baseline self-check",
        "",
        "All four source snapshot hashes matched the first-round `data_manifest.json` before loading.",
        "The baseline configuration uses team weights 0.35/0.35/0.15/0.15 and Manager thresholds 0.20/-0.15. The scan was allowed to proceed only after the known first-round values were reproduced:",
        "",
        "| Ticker | Total return | Sharpe |",
        "|---|---:|---:|",
    ]
    for ticker in TICKERS:
        lines.append(f"| {ticker} | {baseline_actual[ticker]['total_return']:+.2%} | {baseline_actual[ticker]['sharpe']:.2f} |")
    lines += [
        "",
        "## Fixed news constraint",
        "",
        "The scan news score is fixed at neutral 0.0 because historical headlines are unavailable. Therefore `news_weight` is fixed at the default 0.15 and is not treated as an explored dimension. Only the relative weights of the Technical Analyst, Factor Analyst and Risk Manager are varied.",
        "",
        "## Pre-registered stability rule",
        "",
        "A change is called stable only if it improves Sharpe and profit factor on at least 3 of 4 tickers, has positive pooled-PF evidence, and has no zero-trade ticker.",
        "The same rule is re-applied independently to `evaluation_early_third`, `evaluation_middle_third`, `evaluation_late_third` and `evaluation_full`. No configuration is removed because of a negative result.",
        "",
        "## Declared grid",
        "",
        "The scan contains exactly 18 configurations: six team-weight profiles multiplied by three Manager buy/sell threshold pairs. Stop-loss is fixed at 1.5σ, take-profit at 3σ, factor weights at 0.40/0.35/0.25 with volatility weight 0.35, and news weight at 0.15.",
        "",
        "## Complete full-window configuration list",
        "",
        "| Configuration | Both improved | Pooled PF | Zero-trade stocks | Stable this window | Trade counts AAPL / TSLA / NVDA / CBA.AX |",
        "|---|---:|---:|---:|---|---|",
    ]
    for _, row in full_matrix.sort_values("config_id").iterrows():
        counts = "/".join(str(int(row[f"{t.replace('.', '_')}_trades"])) for t in TICKERS)
        pf = "inf" if math.isinf(float(row["pooled_profit_factor"])) else f"{float(row['pooled_profit_factor']):.3f}"
        lines.append(
            f"| `{row['config_id']}` | {int(row['both_improved_n'])}/4 | {pf} | {int(row['zero_trade_n'])} | {'PASS' if row['stable_this_window'] else 'FAIL'} | {counts} |"
        )
    lines += [
        "",
        "## Sub-window pass status",
        "",
        "| Configuration | Early | Middle | Late | Full | Stable windows |",
        "|---|---|---|---|---|---:|",
    ]
    for _, row in aggregate.sort_values("config_id").iterrows():
        statuses = {}
        subset = window_matrix[window_matrix["config_id"] == row["config_id"]]
        for window_id in ("evaluation_early_third", "evaluation_middle_third", "evaluation_late_third", "evaluation_full"):
            item = subset[subset["window_id"] == window_id]
            statuses[window_id] = "PASS" if (not item.empty and bool(item.iloc[0]["stable_this_window"])) else "FAIL"
        lines.append(
            f"| `{row['config_id']}` | {statuses['evaluation_early_third']} | {statuses['evaluation_middle_third']} | {statuses['evaluation_late_third']} | {statuses['evaluation_full']} | {int(row['stable_windows'])}/4 |"
        )
    stable = aggregate[aggregate["cross_window_stable"]]
    lines += [
        "",
        "## Conclusion",
        "",
    ]
    if stable.empty:
        lines.append("Mini TradingAgents 是否存在跨股票稳定的权重改进: 无。No configuration passed the unchanged cross-stock rule in at least three of the four repeated evaluation windows. This is a valid result: the expanded weight and threshold coverage did not produce evidence of a stable improvement over the baseline in this frozen sample.")
    else:
        lines.append("Mini TradingAgents 是否存在跨股票稳定的权重改进: 有候选。The configurations listed below passed the unchanged cross-stock rule in at least three of the four repeated windows; they remain frozen-sample candidates rather than claims of future profitability.")
        for _, row in stable.sort_values("stable_windows", ascending=False).iterrows():
            lines.append(f"- `{row['config_id']}`: stable in `{int(row['stable_windows'])}/4` windows; mean pooled PF `{row['mean_pooled_profit_factor']:.3f}`; mean Sharpe delta `{row['mean_sharpe_delta_all_cells']:.3f}`.")
    lines += [
        "",
        "The complete configuration-by-stock matrix is in `parameter_stock_matrix.csv`; all four sub-window rows are in `parameter_window_summary.csv`; aggregate repeated-window evidence is in `parameter_aggregate_summary.csv`; every completed trade is in `all_trades.csv`.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(output_dir: Path) -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Frozen first-round directory not found: {SOURCE_DIR}")
    source_manifest = json.loads((SOURCE_DIR / "data_manifest.json").read_text(encoding="utf-8"))
    configs = make_configs()
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest: Dict[str, Any] = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_scan": str(SOURCE_DIR),
        "source": source_manifest["source"],
        "period_requested": source_manifest["period_requested"],
        "interval": source_manifest["interval"],
        "evaluation_days": source_manifest["evaluation_days"],
        "tickers": TICKERS,
        "fixed_assumptions": {
            "initial_capital": INITIAL_CAPITAL,
            "position_size": POSITION_SIZE,
            "transaction_cost_bps_per_side": TRANSACTION_COST_BPS,
            "max_hold_bars": MAX_HOLD_BARS,
            "horizon_bars": HORIZON_BARS,
            "stop_loss_mult": STOP_LOSS_MULT,
            "take_profit_mult": TAKE_PROFIT_MULT,
            "news_score": NEWS_SCORE,
            "news_weight": NEWS_WEIGHT,
            **FACTOR_WEIGHTS,
        },
        "grid": {"profiles": WEIGHT_PROFILES, "thresholds": THRESHOLD_GRID, "configuration_count": len(configs)},
        "tickers_data": {},
    }
    data_by_ticker: Dict[str, Tuple[pd.DataFrame, pd.DatetimeIndex, Dict[str, Any]]] = {}
    for ticker in TICKERS:
        raw, metadata = load_frozen_snapshot(ticker, source_manifest)
        data = app.add_indicators(raw)
        cutoff = pd.Timestamp(metadata["evaluation_cutoff"])
        evaluation_index = pd.DatetimeIndex(data.index[data.index >= cutoff])
        if len(evaluation_index) != int(metadata["evaluation_bars"]):
            raise RuntimeError(f"Evaluation bar mismatch for {ticker}: {len(evaluation_index)} != {metadata['evaluation_bars']}")
        metadata["status"] = "frozen_verified"
        manifest["tickers_data"][ticker] = metadata
        data_by_ticker[ticker] = (data, evaluation_index, metadata)
    manifest["baseline_expected"] = EXPECTED_BASELINE
    (output_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    baseline_actual = baseline_self_check(data_by_ticker, configs)
    (output_dir / "baseline_self_check.json").write_text(json.dumps({"status": "passed", "actual": baseline_actual}, indent=2), encoding="utf-8")

    long_rows: List[Dict[str, Any]] = []
    trade_rows: List[Dict[str, Any]] = []
    for ticker, (data, evaluation_index, _) in data_by_ticker.items():
        windows = make_windows(evaluation_index)
        for item in configs:
            for window_id, window_index in windows.items():
                row, trades = evaluate_one_window(ticker, data, window_id, window_index, item)
                long_rows.append(row)
                trade_rows.extend(trades)

    long_df = pd.DataFrame(long_rows)
    trades_df = pd.DataFrame(trade_rows)
    full_matrix = build_matrix(long_df, "evaluation_full")
    window_matrices = [build_matrix(long_df, window_id) for window_id in ("evaluation_early_third", "evaluation_middle_third", "evaluation_late_third", "evaluation_full")]
    window_matrix = pd.concat(window_matrices, ignore_index=True)
    aggregate = build_aggregate(window_matrix)
    long_df.to_csv(output_dir / "scan_runs_long.csv", index=False)
    full_matrix.to_csv(output_dir / "parameter_stock_matrix.csv", index=False)
    window_matrix.to_csv(output_dir / "parameter_window_summary.csv", index=False)
    aggregate.to_csv(output_dir / "parameter_aggregate_summary.csv", index=False)
    trades_df.to_csv(output_dir / "all_trades.csv", index=False)
    write_summary(output_dir, manifest, configs, full_matrix, window_matrix, aggregate, baseline_actual)
    print(json.dumps({"output_dir": str(output_dir), "configs": len(configs), "long_rows": len(long_df), "trades": len(trades_df), "baseline_self_check": "passed"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
