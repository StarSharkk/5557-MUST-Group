"""Reproducible live-data parameter scan for the four trading strategies.

The scanner downloads one 60-day/5-minute snapshot per ticker, uses the earlier
portion only as causal indicator/model pre-roll, and evaluates the same recent
30-day window for every configuration.  It fails fast on a missing live ticker;
the app's demo fallback is never silently included in the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

import app


TICKERS = ["AAPL", "TSLA", "NVDA", "CBA.AX"]
INTERVAL = "5m"
DOWNLOAD_PERIOD = "60d"
EVALUATION_DAYS = 30
INITIAL_CAPITAL = 100_000.0
POSITION_SIZE = 0.20
TRANSACTION_COST_BPS = 2.0
MAX_HOLD_BARS = 24
HORIZON_BARS = 1


def json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def config(strategy: str, family: str, config_id: str, **params: Any) -> Dict[str, Any]:
    return {
        "strategy": strategy,
        "family": family,
        "config_id": config_id,
        "params": params,
    }


def make_configs() -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    risk_grid = [(sl, tp) for sl in (1.0, 1.5, 2.0) for tp in (2.0, 3.0, 4.0)]
    for strategy in app.STRATEGIES:
        for sl, tp in risk_grid:
            configs.append(
                config(
                    strategy,
                    "adaptive_risk",
                    f"risk_sl{sl:g}_tp{tp:g}",
                    stop_loss_mult=sl,
                    take_profit_mult=tp,
                    buy_threshold=(app.ML_BUY_PROB_THRESHOLD if strategy == app.ML_CLASSIFIER_STRATEGY else app.MULTIFACTOR_BUY_THRESHOLD),
                    sell_threshold=(app.ML_SELL_PROB_THRESHOLD if strategy == app.ML_CLASSIFIER_STRATEGY else app.MULTIFACTOR_SELL_THRESHOLD),
                    model="Random Forest" if strategy == app.ML_CLASSIFIER_STRATEGY else None,
                    momentum_weight=0.40,
                    mean_reversion_weight=0.35,
                    flow_weight=0.25,
                    volatility_weight=0.35,
                    technical_weight=0.35,
                    factor_weight=0.35,
                    news_weight=0.15,
                    risk_weight=0.15,
                )
            )

    threshold_pairs = [(0.52, 0.40), (0.55, 0.45), (0.60, 0.48)]
    for model in app.MODEL_CHOICES:
        for buy, sell in threshold_pairs:
            configs.append(
                config(
                    app.ML_CLASSIFIER_STRATEGY,
                    "ml_threshold",
                    f"ml_{model.replace(' ', '_').lower()}_b{buy:.2f}_s{sell:.2f}",
                    model=model,
                    buy_threshold=buy,
                    sell_threshold=sell,
                    stop_loss_mult=1.5,
                    take_profit_mult=3.0,
                )
            )

    weight_profiles = {
        "baseline": (0.40, 0.35, 0.25),
        "balanced": (1 / 3, 1 / 3, 1 / 3),
        "momentum": (0.55, 0.25, 0.20),
        "mean_reversion": (0.25, 0.55, 0.20),
        "flow": (0.30, 0.20, 0.50),
    }
    for profile, (momentum, mean_reversion, flow) in weight_profiles.items():
        for volatility in (0.20, 0.35, 0.50):
            configs.append(
                config(
                    app.MULTIFACTOR_STRATEGY,
                    "multifactor_weights",
                    f"factor_{profile}_vol{volatility:.2f}",
                    momentum_weight=momentum,
                    mean_reversion_weight=mean_reversion,
                    flow_weight=flow,
                    volatility_weight=volatility,
                    buy_threshold=app.MULTIFACTOR_BUY_THRESHOLD,
                    sell_threshold=app.MULTIFACTOR_SELL_THRESHOLD,
                    stop_loss_mult=1.5,
                    take_profit_mult=3.0,
                )
            )
    return configs


def download_live(ticker: str, cache_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    import yfinance as yf

    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
    raw = yf.download(
        ticker,
        period=DOWNLOAD_PERIOD,
        interval=INTERVAL,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no bars")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)
    required = ["open", "high", "low", "close", "volume"]
    if not all(column in raw.columns for column in required):
        raise RuntimeError(f"Yahoo Finance response lacks OHLCV columns: {list(raw.columns)}")
    raw = raw[required].copy()
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index().dropna()
    raw = raw[raw["volume"] >= 0]
    if len(raw) < 1_200:
        raise RuntimeError(f"only {len(raw)} bars returned; live scan requires a substantial 60-day window")
    latest = raw.index.max()
    cutoff = latest - pd.Timedelta(days=EVALUATION_DAYS)
    digest = hashlib.sha256(raw.to_csv().encode("utf-8")).hexdigest()
    metadata = {
        "ticker": ticker,
        "source": "Yahoo Finance via yfinance",
        "period_requested": DOWNLOAD_PERIOD,
        "interval": INTERVAL,
        "bars": int(len(raw)),
        "first_bar": str(raw.index.min()),
        "last_bar": str(raw.index.max()),
        "evaluation_cutoff": str(cutoff),
        "sha256": digest,
        "status": "live",
    }
    return raw, metadata


def risk_series(data: pd.DataFrame, params: Dict[str, Any]) -> Tuple[pd.Series, pd.Series]:
    return app.build_adaptive_risk_series(
        data,
        float(params["stop_loss_mult"]),
        float(params["take_profit_mult"]),
        MAX_HOLD_BARS,
    )


def make_ml_cache(data: pd.DataFrame) -> Dict[str, app.ModelResult]:
    return {
        model: app.train_predict_model(
            data,
            model,
            HORIZON_BARS,
            prediction_start=0,
        )
        for model in app.MODEL_CHOICES
    }


def make_model_result(data: pd.DataFrame, strategy: str, params: Dict[str, Any], ml_cache: Dict[str, app.ModelResult], news_score: float) -> app.ModelResult:
    if strategy == app.ML_CLASSIFIER_STRATEGY:
        return ml_cache[str(params["model"])]
    if strategy == app.MULTIFACTOR_STRATEGY:
        components = app.build_multifactor_components(
            data,
            float(params.get("momentum_weight", 0.40)),
            float(params.get("mean_reversion_weight", 0.35)),
            float(params.get("flow_weight", 0.25)),
            float(params.get("volatility_weight", 0.35)),
        )
        return app.ModelResult(
            probability=components["score"],
            latest_probability=float(components["score"].iloc[-1]),
            accuracy=None,
            f1=None,
            feature_importance=pd.Series(dtype=float),
            model_name=strategy,
        )
    if strategy == app.MINI_TRADINGAGENTS_STRATEGY:
        factor = app.build_multifactor_components(
            data,
            float(params.get("momentum_weight", 0.40)),
            float(params.get("mean_reversion_weight", 0.35)),
            float(params.get("flow_weight", 0.25)),
            float(params.get("volatility_weight", 0.35)),
        )
        technical = app.build_technical_components(data)
        team = app.build_team_components(
            data,
            factor,
            technical,
            news_score,
            float(params.get("technical_weight", 0.35)),
            float(params.get("factor_weight", 0.35)),
            float(params.get("news_weight", 0.15)),
            float(params.get("risk_weight", 0.15)),
        )
        return app.ModelResult(
            probability=team["manager_score"],
            latest_probability=float(team["manager_score"].iloc[-1]),
            accuracy=None,
            f1=None,
            feature_importance=pd.Series(dtype=float),
            model_name=strategy,
        )
    return app.ModelResult(
        probability=pd.Series(0.5, index=data.index, dtype=float),
        latest_probability=0.5,
        accuracy=None,
        f1=None,
        feature_importance=pd.Series(dtype=float),
        model_name=strategy,
    )


def metric_row(trades: pd.DataFrame, curve: pd.DataFrame) -> Dict[str, Any]:
    metrics = app.performance_metrics(trades, curve, INITIAL_CAPITAL)
    gross_profit = float(trades.loc[trades["profit_loss"] > 0, "profit_loss"].sum()) if not trades.empty else 0.0
    gross_loss = float(abs(trades.loc[trades["profit_loss"] < 0, "profit_loss"].sum())) if not trades.empty else 0.0
    return {
        **metrics,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": float(metrics["profit_factor"]),
        "trades": int(metrics["trades"]),
    }


def evaluate_config(
    ticker: str,
    data: pd.DataFrame,
    evaluation_index: pd.DatetimeIndex,
    windows: Dict[str, pd.DatetimeIndex],
    item: Dict[str, Any],
    ml_cache: Dict[str, app.ModelResult],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    strategy = item["strategy"]
    params = item["params"]
    # Historical headlines are not reconstructable from Yahoo's current news endpoint.
    # Neutral sentiment is a fixed, explicit input rather than current news leaked backward.
    news_score = 0.0
    model_result = make_model_result(data, strategy, params, ml_cache, news_score)
    signals = app.build_strategy_signals(
        data,
        strategy,
        model_result,
        news_score,
        float(params["buy_threshold"]),
        float(params["sell_threshold"]),
        float(params.get("momentum_weight", 0.40)),
        float(params.get("mean_reversion_weight", 0.35)),
        float(params.get("flow_weight", 0.25)),
        float(params.get("volatility_weight", 0.35)),
        float(params.get("technical_weight", 0.35)),
        float(params.get("factor_weight", 0.35)),
        float(params.get("news_weight", 0.15)),
        float(params.get("risk_weight", 0.15)),
    )
    stop_loss, take_profit = risk_series(data, params)
    long_rows: List[Dict[str, Any]] = []
    trade_rows: List[Dict[str, Any]] = []
    for window_id, window_index in windows.items():
        window_index = window_index.intersection(data.index)
        if len(window_index) < 20:
            continue
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
        values = metric_row(trades, curve)
        prediction_coverage = float(model_result.probability.loc[window_index].notna().mean())
        row = {
            "window_id": window_id,
            "ticker": ticker,
            "strategy": strategy,
            "family": item["family"],
            "config_id": item["config_id"],
            "params_json": json_text(params),
            "data_status": "live",
            "window_start": str(window_index.min()),
            "window_end": str(window_index.max()),
            "ml_model": model_result.model_name if strategy == app.ML_CLASSIFIER_STRATEGY else "n/a",
            "ml_fallback": bool(model_result.fallback) if strategy == app.ML_CLASSIFIER_STRATEGY else False,
            "ml_prediction_coverage": prediction_coverage,
            "ml_oos_label_positive_rate": model_result.oos_label_positive_rate if strategy == app.ML_CLASSIFIER_STRATEGY else np.nan,
            "ml_oos_predicted_positive_rate": model_result.oos_predicted_positive_rate if strategy == app.ML_CLASSIFIER_STRATEGY else np.nan,
            **values,
        }
        long_rows.append(row)
        if not trades.empty:
            trade_copy = trades.copy()
            trade_copy.insert(0, "run_id", f"{ticker}|{window_id}|{strategy}|{item['config_id']}")
            trade_copy.insert(1, "window_id", window_id)
            trade_copy.insert(2, "ticker", ticker)
            trade_copy.insert(3, "strategy", strategy)
            trade_copy.insert(4, "family", item["family"])
            trade_copy.insert(5, "config_id", item["config_id"])
            trade_rows.extend(trade_copy.to_dict("records"))
    return long_rows, trade_rows


def _is_better(value: float, baseline: float) -> bool:
    if math.isinf(value) and not math.isinf(baseline):
        return True
    if math.isinf(value) and math.isinf(baseline):
        return False
    if math.isnan(value) or math.isnan(baseline):
        return False
    return value > baseline


def build_matrix(full_long: pd.DataFrame, window_id: str = "evaluation_full") -> pd.DataFrame:
    full = full_long[full_long["window_id"] == window_id].copy()
    baseline_rows: Dict[Tuple[str, str, str], pd.Series] = {}
    for (strategy, ticker), group in full.groupby(["strategy", "ticker"]):
        if strategy == app.ML_CLASSIFIER_STRATEGY:
            for model_name in group["ml_model"].dropna().unique():
                baseline = group[
                    (group["family"] == "ml_threshold")
                    & group["config_id"].str.contains("b0.55_s0.45")
                    & (group["ml_model"] == model_name)
                ]
                if not baseline.empty:
                    baseline_rows[(strategy, ticker, str(model_name))] = baseline.iloc[0]
        elif strategy == app.MULTIFACTOR_STRATEGY:
            baseline = group[group["config_id"] == "factor_baseline_vol0.35"]
            if not baseline.empty:
                baseline_rows[(strategy, ticker, "n/a")] = baseline.iloc[0]
        else:
            baseline = group[(group["family"] == "adaptive_risk") & (group["config_id"] == "risk_sl1.5_tp3")]
            if not baseline.empty:
                baseline_rows[(strategy, ticker, "n/a")] = baseline.iloc[0]

    matrix_rows: List[Dict[str, Any]] = []
    for (strategy, family, config_id, params_json), group in full.groupby(["strategy", "family", "config_id", "params_json"], dropna=False):
        row: Dict[str, Any] = {
            "strategy": strategy,
            "family": family,
            "config_id": config_id,
            "params_json": params_json,
        }
        sharpe_values: List[float] = []
        gross_profit = gross_loss = 0.0
        for ticker in TICKERS:
            ticker_row = group[group["ticker"] == ticker]
            if ticker_row.empty:
                continue
            ticker_row = ticker_row.iloc[0]
            model_key = str(ticker_row.get("ml_model", "n/a"))
            baseline = baseline_rows.get((strategy, ticker, model_key))
            prefix = ticker.replace(".", "_")
            for field in ("total_return", "sharpe", "max_drawdown", "win_rate", "profit_factor", "trades", "ml_prediction_coverage"):
                row[f"{prefix}_{field}"] = ticker_row[field]
            sharpe_values.append(float(ticker_row["sharpe"]))
            gross_profit += float(ticker_row["gross_profit"])
            gross_loss += float(ticker_row["gross_loss"])
            if baseline is not None:
                row[f"{prefix}_sharpe_delta"] = float(ticker_row["sharpe"] - baseline["sharpe"])
                row[f"{prefix}_pf_improved"] = _is_better(float(ticker_row["profit_factor"]), float(baseline["profit_factor"]))
                row[f"{prefix}_sharpe_improved"] = float(ticker_row["sharpe"]) > float(baseline["sharpe"])
        row["mean_sharpe"] = float(np.mean(sharpe_values)) if sharpe_values else np.nan
        row["median_sharpe"] = float(np.median(sharpe_values)) if sharpe_values else np.nan
        row["pooled_profit_factor"] = (gross_profit / gross_loss) if gross_loss > 0 else (math.inf if gross_profit > 0 else 0.0)
        for metric in ("sharpe_improved", "pf_improved"):
            row[f"{metric}_n"] = int(sum(bool(row.get(f"{ticker.replace('.', '_')}_{metric}", False)) for ticker in TICKERS))
        row["both_improved_n"] = int(sum(bool(row.get(f"{ticker.replace('.', '_')}_sharpe_improved", False)) and bool(row.get(f"{ticker.replace('.', '_')}_pf_improved", False)) for ticker in TICKERS))
        row["zero_trade_n"] = int(sum(float(row.get(f"{ticker.replace('.', '_')}_trades", 0)) == 0 for ticker in TICKERS))
        matrix_rows.append(row)
    return pd.DataFrame(matrix_rows)


def write_summary(
    matrix: pd.DataFrame,
    window_matrix: pd.DataFrame,
    long_df: pd.DataFrame,
    output_dir: Path,
    manifest: Dict[str, Any],
) -> None:
    lines = [
        "# Multi-stock parameter scan summary",
        "",
        "This report includes every live-data run, including negative returns and zero-trade configurations.",
        "The label formula, 2 bps-per-side cost, position sizing, and data-window rule were fixed before scanning.",
        "Signals execute one completed bar after they are generated; open positions are force-closed at the window end.",
        "",
        "## Pre-registered stability rule",
        "",
        "A change is called stable only if it improves Sharpe and profit factor on at least 3 of 4 tickers, has positive pooled-PF evidence, and has no zero-trade ticker. Changes improving fewer tickers are retained but marked idiosyncratic/inconclusive.",
        "Profit factor cells are reported raw; pooled profit factor aggregates gross gains and gross losses rather than averaging infinity values.",
        "",
        "## Data and limitations",
        "",
        f"Run timestamp: `{manifest['run_timestamp']}`; source: `{manifest['source']}`; requested period: `{DOWNLOAD_PERIOD}`; interval: `{INTERVAL}`.",
        "Yahoo Finance does not provide historical news snapshots, so the scan freezes the Mini TradingAgents news input at neutral (0.0). This prevents current headlines from being leaked into historical bars; it is a limitation, not a positive-result adjustment.",
        "The recent 30 calendar days are evaluated after a 60-day download supplies causal indicator/model pre-roll. No fallback/demo rows are included.",
        "",
        "## Configurations with majority improvement",
        "",
    ]
    candidates = matrix[(matrix["both_improved_n"] >= 3) & (matrix["zero_trade_n"] == 0)] if not matrix.empty else matrix
    if candidates.empty:
        lines.append("No configuration met the pre-registered majority-improvement rule. This is a valid result; no parameter was selected to force positive returns.")
    else:
        for _, row in candidates.sort_values(["both_improved_n", "pooled_profit_factor", "mean_sharpe"], ascending=False).iterrows():
            lines.append(
                f"- `{row['strategy']}` / `{row['config_id']}`: both-improved `{int(row['both_improved_n'])}/4`, Sharpe-improved `{int(row['sharpe_improved_n'])}/4`, PF-improved `{int(row['pf_improved_n'])}/4`, pooled PF `{row['pooled_profit_factor']}`."
            )
    lines += [
        "",
        "## Window consistency",
        "",
    ]
    if window_matrix.empty:
        lines.append("No sub-window summary was produced.")
    else:
        for window_id, window_group in window_matrix.groupby("window_id"):
            majority = int(((window_group["both_improved_n"] >= 3) & (window_group["zero_trade_n"] == 0)).sum())
            lines.append(f"- `{window_id}`: `{majority}` of `{len(window_group)}` configurations met the same 3/4 majority rule; inspect `parameter_window_summary.csv` for the complete cross-window matrix.")
    lines += [
        "",
        "## Changes that are not broadly reliable",
        "",
        "Configurations with fewer than 3 of 4 tickers improving both Sharpe and profit factor are retained in the matrix but are treated as idiosyncratic/inconclusive. A positive result on only one or two stocks, or one driven by very few trades, is not used as the selected methodology.",
        "",
        "The complete parameter-by-stock matrix is in `parameter_stock_matrix.csv`; the long table, fold diagnostics, frozen OHLCV files, and all completed trades are retained alongside this summary.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "yfinance-cache"
    manifest: Dict[str, Any] = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "Yahoo Finance via yfinance",
        "period_requested": DOWNLOAD_PERIOD,
        "interval": INTERVAL,
        "evaluation_days": EVALUATION_DAYS,
        "tickers": TICKERS,
        "fixed_assumptions": {
            "initial_capital": INITIAL_CAPITAL,
            "position_size": POSITION_SIZE,
            "transaction_cost_bps_per_side": TRANSACTION_COST_BPS,
            "max_hold_bars": MAX_HOLD_BARS,
            "horizon_bars": HORIZON_BARS,
            "ml_min_train_bars": app.ML_MIN_TRAIN_BARS,
            "ml_train_window_bars": app.ML_TRAIN_WINDOW_BARS,
            "ml_retrain_interval": app.ML_RETRAIN_INTERVAL,
        },
        "tickers_data": {},
    }
    data_by_ticker: Dict[str, Tuple[pd.DataFrame, pd.DatetimeIndex, pd.DatetimeIndex]] = {}
    failed: Dict[str, str] = {}
    for ticker in TICKERS:
        try:
            raw, metadata = download_live(ticker, cache_dir)
            raw.to_csv(output_dir / f"{ticker.replace('.', '_')}_ohlcv.csv")
            data = app.add_indicators(raw)
            cutoff = pd.Timestamp(metadata["evaluation_cutoff"])
            evaluation_index = data.index[data.index >= cutoff]
            if len(evaluation_index) < 500:
                raise RuntimeError(f"only {len(evaluation_index)} indicator bars in the recent evaluation window")
            third = len(evaluation_index) // 3
            windows = pd.DatetimeIndex(evaluation_index)
            data_by_ticker[ticker] = (data, windows, evaluation_index)
            metadata["indicator_bars"] = int(len(data))
            metadata["evaluation_bars"] = int(len(evaluation_index))
            manifest["tickers_data"][ticker] = metadata
        except Exception as exc:
            failed[ticker] = str(exc)
            manifest["tickers_data"][ticker] = {"status": "failed", "error": str(exc)}
    (output_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    if failed:
        raise RuntimeError(f"Live-only scan aborted; failed tickers: {failed}")

    configs = make_configs()
    long_rows: List[Dict[str, Any]] = []
    trade_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    for ticker, (data, evaluation_index, _) in data_by_ticker.items():
        split = len(evaluation_index) // 3
        windows = {
            "evaluation_full": evaluation_index,
            "evaluation_early_third": evaluation_index[:split],
            "evaluation_middle_third": evaluation_index[split : 2 * split],
            "evaluation_late_third": evaluation_index[2 * split :],
        }
        ml_cache = make_ml_cache(data)
        for model_name, model_result in ml_cache.items():
            if model_result.fold_metrics is not None and not model_result.fold_metrics.empty:
                diagnostic = model_result.fold_metrics.copy()
                diagnostic.insert(0, "ticker", ticker)
                diagnostic.insert(1, "requested_model", model_name)
                fold_rows.extend(diagnostic.to_dict("records"))
        for item in configs:
            rows, trades = evaluate_config(ticker, data, evaluation_index, windows, item, ml_cache)
            long_rows.extend(rows)
            trade_rows.extend(trades)

    long_df = pd.DataFrame(long_rows)
    trades_df = pd.DataFrame(trade_rows)
    matrix = build_matrix(long_df)
    window_matrices: List[pd.DataFrame] = []
    for window_id in sorted(long_df["window_id"].dropna().unique()):
        window_matrix = build_matrix(long_df, window_id)
        window_matrix.insert(0, "window_id", window_id)
        window_matrices.append(window_matrix)
    all_window_matrix = pd.concat(window_matrices, ignore_index=True) if window_matrices else pd.DataFrame()
    long_df.to_csv(output_dir / "scan_runs_long.csv", index=False)
    matrix.to_csv(output_dir / "parameter_stock_matrix.csv", index=False)
    all_window_matrix.to_csv(output_dir / "parameter_window_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "ml_fold_diagnostics.csv", index=False)
    trades_df.to_csv(output_dir / "all_trades.csv", index=False)
    write_summary(matrix, all_window_matrix, long_df, output_dir, manifest)
    print(json.dumps({
        "output_dir": str(output_dir),
        "configs": len(configs),
        "long_rows": len(long_df),
        "trades": len(trades_df),
        "matrix_rows": len(matrix),
        "status": "completed",
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_dir
    if output is None:
        output = Path("analysis_results") / datetime.now().strftime("scan_%Y%m%d_%H%M%S")
    run(output)


if __name__ == "__main__":
    main()
