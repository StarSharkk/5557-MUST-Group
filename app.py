from __future__ import annotations

import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    import plotly.express as px
except Exception:  # pragma: no cover - handled in UI
    go = None
    px = None

try:
    import yfinance as yf
except Exception:  # pragma: no cover - handled in UI
    yf = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - handled in UI
    RandomForestClassifier = None
    LogisticRegression = None
    accuracy_score = None
    balanced_accuracy_score = None
    confusion_matrix = None
    f1_score = None
    make_pipeline = None
    StandardScaler = None

APP_TITLE = "AI Intraday Quant Trading Simulator"
TICKERS = ["AAPL", "TSLA", "NVDA", "CBA.AX"]
MINI_TRADINGAGENTS_STRATEGY = "Mini TradingAgents"
MULTIFACTOR_STRATEGY = "Multi-factor model"
FREQTRADE_STRATEGY = "Freqtrade Sample Strategy"
ML_CLASSIFIER_STRATEGY = "ML Classifier"
STRATEGIES = [MINI_TRADINGAGENTS_STRATEGY, MULTIFACTOR_STRATEGY, FREQTRADE_STRATEGY, ML_CLASSIFIER_STRATEGY]
MODEL_CHOICES = ["Random Forest", "Logistic Regression", "XGBoost"]
FREQTRADE_BUY_RSI = 30
FREQTRADE_SELL_RSI = 70
MULTIFACTOR_BUY_THRESHOLD = 0.35
MULTIFACTOR_SELL_THRESHOLD = -0.35
# Candidate selected before this default change by the frozen multi-stock,
# four-window rule: both Sharpe and PF improved on >=3/4 tickers in 3/4 windows.
# Keep the original weights in the scanner as the explicit comparison baseline.
MULTIFACTOR_DEFAULT_MOMENTUM_WEIGHT = 0.55
MULTIFACTOR_DEFAULT_MEAN_REVERSION_WEIGHT = 0.25
MULTIFACTOR_DEFAULT_FLOW_WEIGHT = 0.20
MULTIFACTOR_DEFAULT_VOLATILITY_WEIGHT = 0.20
TEAM_BUY_THRESHOLD = 0.20
TEAM_SELL_THRESHOLD = -0.15
ML_BUY_PROB_THRESHOLD = 0.55
ML_SELL_PROB_THRESHOLD = 0.45
ML_MIN_TRAIN_BARS = 200
ML_TRAIN_WINDOW_BARS = 1000
ML_RETRAIN_INTERVAL = 200
FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_6",
    "rsi",
    "macd_hist",
    "bb_percent",
    "volatility",
    "volume_spike",
    "ma_gap",
    "vwap_gap",
]


@dataclass
class ModelResult:
    probability: pd.Series
    latest_probability: float
    accuracy: float | None
    f1: float | None
    feature_importance: pd.Series
    model_name: str
    fallback: bool = False
    prediction_coverage: float = 0.0
    walk_forward_folds: int = 0
    oos_label_positive_rate: float | None = None
    oos_predicted_positive_rate: float | None = None
    fold_metrics: pd.DataFrame | None = None
    fallback_reason: str | None = None
    prediction_start: int | None = None


st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")



def clean_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.\-]{1,12}", ticker):
        raise ValueError("Ticker contains invalid characters.")
    return ticker


def crossed_above(series: pd.Series, threshold: float) -> pd.Series:
    previous = series.shift(1)
    return (series > threshold) & (previous <= threshold)


def rolling_zscore(series: pd.Series, window: int = 80) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0, np.nan)
    return ((series - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0)


def configure_yfinance_cache() -> None:
    """Keep yfinance's timezone SQLite cache inside the project/runtime directory."""
    if yf is None or not hasattr(yf, "set_tz_cache_location"):
        return
    cache_dir = Path(os.environ.get("YFINANCE_CACHE_DIR", ".yfinance-cache"))
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(cache_dir))
    except Exception:
        # A read-only runtime can still use yfinance without the optional cache.
        pass


@st.cache_data(show_spinner=False, ttl=1800)
def load_price_data(ticker: str, months: int, interval: str) -> Tuple[pd.DataFrame, str]:
    ticker = clean_ticker(ticker)
    period = "1mo" if months == 1 else "3mo"

    if yf is None:
        return make_demo_data(ticker, months, interval), "demo: yfinance is not installed"

    try:
        configure_yfinance_cache()
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        return make_demo_data(ticker, months, interval), f"demo: yfinance request failed ({exc})"

    if data is None or data.empty:
        return make_demo_data(ticker, months, interval), "demo: provider returned no intraday bars"

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.rename(columns=str.lower)
    required = ["open", "high", "low", "close", "volume"]
    if not all(col in data.columns for col in required):
        return make_demo_data(ticker, months, interval), "demo: provider response missing OHLCV columns"

    data = data[required].copy()
    data.index = pd.to_datetime(data.index)
    data = data.dropna()
    data = data[data["volume"] >= 0]

    if len(data) < 120:
        return make_demo_data(ticker, months, interval), "demo: not enough intraday bars for modelling"

    return data, "live: yfinance intraday OHLCV"


def make_demo_data(ticker: str, months: int, interval: str) -> pd.DataFrame:
    seed = abs(hash((ticker, months, interval))) % (2**32)
    rng = np.random.default_rng(seed)
    bars_per_day = 390 if interval == "1m" else 78
    days = 21 if months == 1 else 63
    total = bars_per_day * days
    freq = "1min" if interval == "1m" else "5min"
    idx = pd.date_range(end=pd.Timestamp.now().floor("min"), periods=total, freq=freq)

    base_price = {"AAPL": 195, "TSLA": 240, "NVDA": 125, "CBA.AX": 120}.get(ticker, 100)
    drift = {"AAPL": 0.000005, "TSLA": 0.00001, "NVDA": 0.000012, "CBA.AX": 0.000003}.get(ticker, 0)
    shock = rng.normal(drift, 0.0018 if ticker in {"TSLA", "NVDA"} else 0.0011, total)
    close = base_price * np.exp(np.cumsum(shock))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.00025, total))
    spread = np.abs(rng.normal(0.0009, 0.00035, total))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.lognormal(12.3 if ticker != "CBA.AX" else 11.3, 0.45, total).astype(int)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    typical_price = (out["high"] + out["low"] + out["close"]) / 3
    out["vwap"] = (typical_price * out["volume"]).cumsum() / out["volume"].replace(0, np.nan).cumsum()
    out["vwap_gap"] = out["close"] / out["vwap"] - 1

    out["return_1"] = out["close"].pct_change()
    out["return_3"] = out["close"].pct_change(3)
    out["return_6"] = out["close"].pct_change(6)

    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))

    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]

    rolling_mean = out["close"].rolling(20).mean()
    rolling_std = out["close"].rolling(20).std()
    out["bb_middle"] = rolling_mean
    out["bb_upper"] = rolling_mean + 2 * rolling_std
    out["bb_lower"] = rolling_mean - 2 * rolling_std
    out["bb_percent"] = (out["close"] - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"])
    out["z_score"] = (out["close"] - rolling_mean) / rolling_std

    out["volatility"] = out["return_1"].rolling(20).std()
    volume_avg = out["volume"].rolling(20).mean()
    out["volume_spike"] = out["volume"] / volume_avg.replace(0, np.nan)

    out["ma_fast"] = out["close"].rolling(8).mean()
    out["ma_slow"] = out["close"].rolling(21).mean()
    out["ma_gap"] = out["ma_fast"] / out["ma_slow"] - 1
    out["ma_cross"] = np.sign(out["ma_gap"]).diff().fillna(0)

    ema1 = out["close"].ewm(span=9, adjust=False).mean()
    ema2 = ema1.ewm(span=9, adjust=False).mean()
    ema3 = ema2.ewm(span=9, adjust=False).mean()
    out["tema"] = 3 * (ema1 - ema2) + ema3

    return out.replace([np.inf, -np.inf], np.nan).dropna()


def calculate_label_threshold(df: pd.DataFrame, horizon_bars: int) -> float:
    """Return the original, pre-declared label threshold without looking at future folds."""
    volatility = float(df["return_1"].std())
    if not math.isfinite(volatility):
        volatility = 0.0
    return max(volatility * math.sqrt(horizon_bars) * 0.25, 0.0005)


def make_labels(
    df: pd.DataFrame,
    horizon_bars: int,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Create matured labels only; the final ``horizon_bars`` rows are never labelled as zero."""
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be at least 1")
    out = df.copy()
    out["future_return"] = out["close"].shift(-horizon_bars) / out["close"] - 1
    label_threshold = calculate_label_threshold(out, horizon_bars) if threshold is None else float(threshold)
    matured = out["future_return"].notna()
    out["target"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[matured, "target"] = (out.loc[matured, "future_return"] > label_threshold).astype(int)
    return out.loc[matured].dropna()


def class_balance_ratio(y_train: pd.Series) -> float | None:
    """Return n_negative / n_positive for XGBoost, including ratios below one."""
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    if positives == 0 or negatives == 0:
        return None
    return negatives / positives


def _build_classifier(model_choice: str, y_train: pd.Series) -> Tuple[Any, str, float | None]:
    balance_ratio = class_balance_ratio(y_train)
    if model_choice == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except Exception:
            XGBClassifier = None
        if XGBClassifier is not None:
            model = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                scale_pos_weight=balance_ratio,
                random_state=42,
                n_jobs=1,
                verbosity=0,
            )
            return model, "XGBoost", balance_ratio
        model_choice = "Random Forest"
        fallback_name = "Random Forest (XGBoost unavailable)"
    else:
        fallback_name = "Random Forest"

    if model_choice == "Logistic Regression":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42),
        )
        return model, "Logistic Regression", balance_ratio

    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=7,
        min_samples_leaf=8,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    return model, fallback_name, balance_ratio


def _model_feature_importance(model: Any) -> pd.Series | None:
    estimator = list(model.named_steps.values())[-1] if hasattr(model, "named_steps") else model
    if hasattr(estimator, "feature_importances_"):
        return pd.Series(estimator.feature_importances_, index=FEATURE_COLUMNS, dtype=float)
    if hasattr(estimator, "coef_"):
        return pd.Series(np.abs(estimator.coef_[0]), index=FEATURE_COLUMNS, dtype=float)
    return None


def train_predict_model(
    df: pd.DataFrame,
    model_choice: str,
    horizon_bars: int,
    train_window_bars: int = ML_TRAIN_WINDOW_BARS,
    retrain_interval: int = ML_RETRAIN_INTERVAL,
    min_train_bars: int = ML_MIN_TRAIN_BARS,
    prediction_start: int | None = None,
) -> ModelResult:
    """Generate strictly out-of-sample probabilities with a purged walk-forward loop.

    The label formula is unchanged. Its threshold is estimated once from the initial
    pre-roll training data and then frozen, so later evaluation volatility cannot
    change earlier labels. Every fold trains a fresh model using only labels whose
    future horizon has fully matured before that fold's first prediction.
    """
    if horizon_bars < 1 or train_window_bars < 1 or retrain_interval < 1 or min_train_bars < 1:
        raise ValueError("walk-forward bar counts must all be positive")

    minimum_prediction_start = min_train_bars + horizon_bars
    first_prediction = minimum_prediction_start if prediction_start is None else max(
        int(prediction_start), minimum_prediction_start
    )
    probabilities = pd.Series(np.nan, index=df.index, dtype=float)

    if RandomForestClassifier is None or len(df) <= first_prediction:
        reason = "scikit-learn is unavailable" if RandomForestClassifier is None else "not enough bars for walk-forward training"
        return ModelResult(
            probability=probabilities,
            latest_probability=0.5,
            accuracy=None,
            f1=None,
            feature_importance=heuristic_importance(df),
            model_name=f"{model_choice} (not trained)",
            fallback=True,
            fold_metrics=pd.DataFrame(),
            fallback_reason=reason,
            prediction_start=first_prediction,
        )

    initial_label_end = first_prediction - horizon_bars
    initial_train_start = max(0, initial_label_end - train_window_bars)
    frozen_threshold = calculate_label_threshold(
        df.iloc[initial_train_start:initial_label_end], horizon_bars
    )
    future_return = df["close"].shift(-horizon_bars) / df["close"] - 1

    fold_records: List[Dict[str, object]] = []
    oos_true_parts: List[pd.Series] = []
    oos_pred_parts: List[pd.Series] = []
    importance_total = pd.Series(0.0, index=FEATURE_COLUMNS)
    importance_weight = 0
    trained_folds = 0
    actual_model_names: List[str] = []

    for fold_number, predict_start in enumerate(range(first_prediction, len(df), retrain_interval), start=1):
        predict_end = min(predict_start + retrain_interval, len(df))
        label_end = predict_start - horizon_bars
        train_start = max(0, label_end - train_window_bars)

        # Include the h future bars needed to mature the final training label, but no prediction-block bars.
        training_history = df.iloc[train_start:predict_start]
        train = make_labels(training_history, horizon_bars, threshold=frozen_threshold)
        y_train = train["target"].astype(int)
        positives = int((y_train == 1).sum())
        negatives = int((y_train == 0).sum())
        record: Dict[str, object] = {
            "fold": fold_number,
            "train_start": train.index.min() if not train.empty else None,
            "train_end": train.index.max() if not train.empty else None,
            "predict_start": df.index[predict_start],
            "predict_end": df.index[predict_end - 1],
            "train_size": int(len(train)),
            "n_positive": positives,
            "n_negative": negatives,
            "train_positive_rate": float(y_train.mean()) if len(y_train) else np.nan,
            "label_threshold": frozen_threshold,
            "status": "trained",
        }

        if len(train) < min_train_bars or y_train.nunique() < 2:
            record.update(
                {
                    "status": "skipped_insufficient_rows" if len(train) < min_train_bars else "skipped_one_class",
                    "model": None,
                    "scale_pos_weight": np.nan,
                    "eval_size": 0,
                    "accuracy": np.nan,
                    "balanced_accuracy": np.nan,
                    "f1": np.nan,
                    "oos_positive_rate": np.nan,
                    "predicted_positive_rate": np.nan,
                    "tn": np.nan,
                    "fp": np.nan,
                    "fn": np.nan,
                    "tp": np.nan,
                }
            )
            fold_records.append(record)
            continue

        model, actual_model_name, balance_ratio = _build_classifier(model_choice, y_train)
        model.fit(train[FEATURE_COLUMNS], y_train)
        prediction_frame = df.iloc[predict_start:predict_end]
        fold_probability = pd.Series(
            model.predict_proba(prediction_frame[FEATURE_COLUMNS])[:, 1],
            index=prediction_frame.index,
            dtype=float,
        )
        probabilities.loc[prediction_frame.index] = fold_probability
        trained_folds += 1
        actual_model_names.append(actual_model_name)

        importance = _model_feature_importance(model)
        if importance is not None:
            importance_total = importance_total.add(importance * len(prediction_frame), fill_value=0)
            importance_weight += len(prediction_frame)

        matured_mask = future_return.iloc[predict_start:predict_end].notna()
        eval_index = prediction_frame.index[matured_mask.to_numpy()]
        y_true = (future_return.loc[eval_index] > frozen_threshold).astype(int)
        y_pred = pd.Series(model.predict(df.loc[eval_index, FEATURE_COLUMNS]), index=eval_index, dtype=int)
        if len(y_true):
            oos_true_parts.append(y_true)
            oos_pred_parts.append(y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            fold_accuracy = float(accuracy_score(y_true, y_pred))
            fold_balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
            fold_f1 = float(f1_score(y_true, y_pred, zero_division=0))
        else:
            tn = fp = fn = tp = np.nan
            fold_accuracy = fold_balanced_accuracy = fold_f1 = np.nan

        record.update(
            {
                "model": actual_model_name,
                "scale_pos_weight": balance_ratio,
                "eval_size": int(len(y_true)),
                "accuracy": fold_accuracy,
                "balanced_accuracy": fold_balanced_accuracy,
                "f1": fold_f1,
                "oos_positive_rate": float(y_true.mean()) if len(y_true) else np.nan,
                "predicted_positive_rate": float(y_pred.mean()) if len(y_pred) else np.nan,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )
        fold_records.append(record)

    if oos_true_parts:
        all_true = pd.concat(oos_true_parts).sort_index()
        all_pred = pd.concat(oos_pred_parts).sort_index()
        accuracy = float(accuracy_score(all_true, all_pred))
        f1 = float(f1_score(all_true, all_pred, zero_division=0))
        label_positive_rate = float(all_true.mean())
        predicted_positive_rate = float(all_pred.mean())
    else:
        accuracy = f1 = label_positive_rate = predicted_positive_rate = None

    importance = (
        (importance_total / importance_weight).sort_values(ascending=False)
        if importance_weight
        else heuristic_importance(df)
    )
    if actual_model_names:
        model_name = actual_model_names[-1]
        fallback = model_name.endswith("unavailable)")
        fallback_reason = "XGBoost unavailable; Random Forest used" if fallback else None
    else:
        model_name = f"{model_choice} (not trained)"
        fallback = True
        fallback_reason = "every walk-forward fold lacked enough rows or both target classes"

    latest_probability = float(probabilities.iloc[-1]) if pd.notna(probabilities.iloc[-1]) else 0.5
    return ModelResult(
        probability=probabilities,
        latest_probability=latest_probability,
        accuracy=accuracy,
        f1=f1,
        feature_importance=importance,
        model_name=model_name,
        fallback=fallback,
        prediction_coverage=float(probabilities.notna().mean()),
        walk_forward_folds=trained_folds,
        oos_label_positive_rate=label_positive_rate,
        oos_predicted_positive_rate=predicted_positive_rate,
        fold_metrics=pd.DataFrame(fold_records),
        fallback_reason=fallback_reason,
        prediction_start=first_prediction,
    )


def heuristic_probability(df: pd.DataFrame) -> pd.Series:
    score = (
        7.5 * df["return_3"].fillna(0)
        + 4.5 * df["ma_gap"].fillna(0)
        + 0.08 * (50 - df["rsi"].fillna(50)) / 50
        - 1.8 * df["volatility"].fillna(0)
        + 0.025 * (df["volume_spike"].fillna(1) - 1)
    )
    return (1 / (1 + np.exp(-score * 8))).clip(0.05, 0.95)


def heuristic_importance(df: pd.DataFrame) -> pd.Series:
    vals = {
        "return_3": abs(df["return_3"].corr(df["close"].pct_change().shift(-3))) if len(df) > 10 else 0.2,
        "ma_gap": 0.18,
        "rsi": 0.16,
        "macd_hist": 0.14,
        "volatility": 0.12,
        "volume_spike": 0.10,
        "vwap_gap": 0.08,
    }
    return pd.Series(vals).replace([np.inf, -np.inf], np.nan).fillna(0.1).sort_values(ascending=False)


def fetch_news_sentiment(ticker: str) -> Tuple[float, List[str]]:
    positive = {
        "beat",
        "growth",
        "upgrade",
        "record",
        "surge",
        "strong",
        "profit",
        "optimistic",
        "bullish",
        "rally",
        "outperform",
    }
    negative = {
        "miss",
        "downgrade",
        "risk",
        "weak",
        "loss",
        "lawsuit",
        "bearish",
        "fall",
        "drop",
        "concern",
        "underperform",
    }
    if yf is None:
        return 0.0, ["News API unavailable; using neutral sentiment."]
    try:
        configure_yfinance_cache()
        news = yf.Ticker(ticker).news or []
    except Exception:
        return 0.0, ["News request failed; using neutral sentiment."]

    headlines = []
    score = 0
    for item in news[:10]:
        title = item.get("title") or item.get("content", {}).get("title", "")
        if not title:
            continue
        headlines.append(title)
        words = set(re.findall(r"[a-z]+", title.lower()))
        score += len(words & positive) - len(words & negative)
    if not headlines:
        return 0.0, ["No recent headlines found; using neutral sentiment."]
    return float(np.tanh(score / max(3, len(headlines)))), headlines[:5]


def build_multifactor_score(df: pd.DataFrame) -> pd.Series:
    momentum_score = (
        0.45 * rolling_zscore(df["return_3"])
        + 0.35 * rolling_zscore(df["return_6"])
        + 0.20 * rolling_zscore(df["ma_gap"])
    )
    mean_reversion_score = (
        -0.45 * rolling_zscore(df["rsi"])
        - 0.30 * rolling_zscore(df["bb_percent"])
        - 0.25 * rolling_zscore(df["z_score"])
    )
    flow_score = (
        0.50 * rolling_zscore(df["volume_spike"])
        + 0.30 * rolling_zscore(df["vwap_gap"])
        + 0.20 * rolling_zscore(df["macd_hist"])
    )
    volatility_penalty = 0.35 * rolling_zscore(df["volatility"])

    score = 0.40 * momentum_score + 0.35 * mean_reversion_score + 0.25 * flow_score - volatility_penalty
    return score.clip(-3, 3)


def build_multifactor_components(
    df: pd.DataFrame,
    momentum_weight: float,
    mean_reversion_weight: float,
    flow_weight: float,
    volatility_weight: float,
) -> pd.DataFrame:
    components = pd.DataFrame(index=df.index)
    components["momentum"] = (
        0.45 * rolling_zscore(df["return_3"])
        + 0.35 * rolling_zscore(df["return_6"])
        + 0.20 * rolling_zscore(df["ma_gap"])
    )
    components["mean_reversion"] = (
        -0.45 * rolling_zscore(df["rsi"])
        - 0.30 * rolling_zscore(df["bb_percent"])
        - 0.25 * rolling_zscore(df["z_score"])
    )
    components["flow_trend"] = (
        0.50 * rolling_zscore(df["volume_spike"])
        + 0.30 * rolling_zscore(df["vwap_gap"])
        + 0.20 * rolling_zscore(df["macd_hist"])
    )
    components["volatility_penalty"] = -rolling_zscore(df["volatility"])
    components["score"] = (
        momentum_weight * components["momentum"]
        + mean_reversion_weight * components["mean_reversion"]
        + flow_weight * components["flow_trend"]
        + volatility_weight * components["volatility_penalty"]
    ).clip(-3, 3)
    return components


def build_technical_components(df: pd.DataFrame) -> pd.DataFrame:
    technical = pd.DataFrame(index=df.index)
    technical["trend"] = (
        0.40 * rolling_zscore(df["ma_gap"])
        + 0.35 * rolling_zscore(df["vwap_gap"])
        + 0.25 * rolling_zscore(df["macd_hist"])
    )
    technical["timing"] = (
        -0.55 * rolling_zscore(df["rsi"])
        - 0.25 * rolling_zscore(df["bb_percent"])
        + 0.20 * rolling_zscore(df["return_1"])
    )
    technical["technical_score"] = (0.65 * technical["trend"] + 0.35 * technical["timing"]).clip(-3, 3)
    return technical


def build_team_components(
    df: pd.DataFrame,
    factor_components: pd.DataFrame,
    technical_components: pd.DataFrame,
    news_score: float,
    technical_weight: float,
    factor_weight: float,
    news_weight: float,
    risk_weight: float,
) -> pd.DataFrame:
    team = pd.DataFrame(index=df.index)
    team["technical_analyst"] = technical_components["technical_score"]
    team["factor_analyst"] = factor_components["score"]
    team["news_analyst"] = pd.Series(news_score, index=df.index, dtype=float).clip(-1, 1)
    team["risk_manager"] = (
        -0.60 * rolling_zscore(df["volatility"])
        - 0.25 * rolling_zscore((df["close"] / df["bb_middle"] - 1).abs().fillna(0))
        + 0.20 * (df["close"] >= df["bb_middle"]).astype(float)
    ).clip(-2, 2)
    team["manager_score"] = (
        technical_weight * team["technical_analyst"]
        + factor_weight * team["factor_analyst"]
        + news_weight * team["news_analyst"]
        + risk_weight * team["risk_manager"]
    ).clip(-3, 3)
    return team


def build_strategy_signals(
    df: pd.DataFrame,
    strategy: str,
    ml_result: ModelResult,
    news_score: float,
    buy_threshold: float,
    sell_threshold: float,
    momentum_weight: float = 0.40,
    mean_reversion_weight: float = 0.35,
    flow_weight: float = 0.25,
    volatility_weight: float = 0.35,
    technical_weight: float = 0.35,
    factor_weight: float = 0.35,
    news_weight: float = 0.15,
    risk_weight: float = 0.15,
) -> pd.Series:
    signal = pd.Series("Hold", index=df.index, dtype=object)
    if strategy == MINI_TRADINGAGENTS_STRATEGY:
        factor_components = build_multifactor_components(
            df,
            momentum_weight,
            mean_reversion_weight,
            flow_weight,
            volatility_weight,
        )
        technical_components = build_technical_components(df)
        team = build_team_components(
            df,
            factor_components,
            technical_components,
            news_score,
            technical_weight,
            factor_weight,
            news_weight,
            risk_weight,
        )
        trend_filter = df["close"] >= df["bb_middle"]
        buy = (team["manager_score"] >= buy_threshold) & trend_filter & (team["risk_manager"] > -0.75) & (df["volume"] > 0)
        sell = (team["manager_score"] <= sell_threshold) | (team["risk_manager"] < -0.90) | (df["close"] < df["bb_middle"])
    elif strategy == MULTIFACTOR_STRATEGY:
        score = build_multifactor_components(
            df,
            momentum_weight,
            mean_reversion_weight,
            flow_weight,
            volatility_weight,
        )["score"]
        trend_filter = df["close"] >= df["bb_middle"]
        buy = (score >= buy_threshold) & trend_filter & (df["volume"] > 0)
        sell = (score <= sell_threshold) | (df["close"] < df["bb_middle"])
    elif strategy == ML_CLASSIFIER_STRATEGY:
        probability = ml_result.probability.reindex(df.index)
        buy = (probability >= buy_threshold) & (df["volume"] > 0)
        sell = probability <= sell_threshold
    else:
        buy = (
            crossed_above(df["rsi"], FREQTRADE_BUY_RSI)
            & (df["tema"] <= df["bb_middle"])
            & (df["tema"] > df["tema"].shift(1))
            & (df["volume"] > 0)
        )
        sell = (
            crossed_above(df["rsi"], FREQTRADE_SELL_RSI)
            & (df["tema"] > df["bb_middle"])
            & (df["tema"] < df["tema"].shift(1))
            & (df["volume"] > 0)
        )

    signal.loc[buy] = "Buy"
    signal.loc[sell] = "Sell"
    return signal


def build_adaptive_risk_series(
    df: pd.DataFrame,
    stop_loss_mult: float,
    take_profit_mult: float,
    max_hold_bars: int,
) -> Tuple[pd.Series, pd.Series]:
    """Build the app's pre-declared volatility-scaled risk limits."""
    vol_floor = df["volatility"].dropna().median() if df["volatility"].notna().any() else 0.001
    vol = df["volatility"].fillna(vol_floor).clip(lower=vol_floor * 0.25)
    horizon_scale = math.sqrt(min(max_hold_bars, 60))
    stop_loss = (stop_loss_mult * vol * horizon_scale).clip(lower=0.0015, upper=0.05)
    take_profit = (take_profit_mult * vol * horizon_scale).clip(lower=0.003, upper=0.10)
    return stop_loss, take_profit


def backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    initial_capital: float,
    position_size: float,
    stop_loss: pd.Series,
    take_profit: pd.Series,
    transaction_cost_bps: float,
    max_hold_bars: int,
    allow_short: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cash = initial_capital
    equity = initial_capital
    position = None
    trades: List[Dict[str, object]] = []
    curve = []
    cost_rate = transaction_cost_bps / 10000
    execution_signal = signal.reindex(df.index).shift(1).fillna("Hold")

    rows = list(df.iterrows())
    for i, (ts, row) in enumerate(rows):
        price = float(row["close"])
        current_signal = execution_signal.loc[ts]

        if position is not None:
            side = position["side"]
            raw_return = (price / position["entry_price"] - 1) * side
            hold_bars = i - position["entry_i"]
            exit_reason = None
            if raw_return <= -position["stop_loss"]:
                exit_reason = "stop-loss"
            elif raw_return >= position["take_profit"]:
                exit_reason = "take-profit"
            elif hold_bars >= max_hold_bars:
                exit_reason = "time exit"
            elif (side == 1 and current_signal == "Sell") or (side == -1 and current_signal == "Buy"):
                exit_reason = "opposite signal"
            elif i == len(rows) - 1:
                exit_reason = "end of window"

            mark_to_market = cash + position["notional"] * raw_return
            equity = mark_to_market

            if exit_reason:
                net_return = raw_return - 2 * cost_rate
                pnl = position["notional"] * net_return
                cash += pnl
                equity = cash
                trades.append(
                    {
                        "entry_time": position["entry_time"],
                        "exit_time": ts,
                        "side": "LONG" if side == 1 else "SHORT",
                        "entry_price": position["entry_price"],
                        "exit_price": price,
                        "return_pct": net_return * 100,
                        "profit_loss": pnl,
                        "exit_reason": exit_reason,
                        "stop_loss_pct": position["stop_loss"] * 100,
                        "take_profit_pct": position["take_profit"] * 100,
                        "execution_note": "Signal shifted by one bar; entry/exit use the next completed candle close",
                    }
                )
                position = None

        can_open = i < len(rows) - 1 and (current_signal == "Buy" or (allow_short and current_signal == "Sell"))
        if position is None and can_open:
            notional = max(cash * position_size, 0)
            if notional > 0:
                position = {
                    "side": 1 if current_signal == "Buy" else -1,
                    "entry_price": price,
                    "entry_time": ts,
                    "entry_i": i,
                    "notional": notional,
                    "stop_loss": float(stop_loss.loc[ts]),
                    "take_profit": float(take_profit.loc[ts]),
                }

        curve.append(
            {
                "time": ts,
                "equity": equity,
                "raw_signal": signal.loc[ts],
                "execution_signal": current_signal,
                "close": price,
            }
        )

    trades_df = pd.DataFrame(trades)
    curve_df = pd.DataFrame(curve).set_index("time")
    return trades_df, curve_df


def performance_metrics(trades: pd.DataFrame, equity_curve: pd.DataFrame, initial_capital: float) -> Dict[str, float]:
    if equity_curve.empty:
        return {k: 0.0 for k in ["total_return", "sharpe", "max_drawdown", "win_rate", "avg_profit", "trades", "profit_factor"]}

    total_return = equity_curve["equity"].iloc[-1] / initial_capital - 1
    returns = equity_curve["equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = 0.0
    if len(returns) > 2 and returns.std() > 0:
        sharpe = float((returns.mean() / returns.std()) * math.sqrt(min(252 * 78, len(returns) * 12)))

    running_max = equity_curve["equity"].cummax()
    drawdown = equity_curve["equity"] / running_max - 1
    max_drawdown = float(drawdown.min())

    if trades.empty:
        win_rate = avg_profit = profit_factor = 0.0
        n_trades = 0
    else:
        pnl = trades["profit_loss"]
        win_rate = float((pnl > 0).mean())
        avg_profit = float(pnl.mean())
        gains = pnl[pnl > 0].sum()
        losses = abs(pnl[pnl < 0].sum())
        profit_factor = float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0
        n_trades = int(len(trades))

    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "avg_profit": avg_profit,
        "trades": n_trades,
        "profit_factor": profit_factor,
    }


def explain_signal(df: pd.DataFrame, signal: str, ml_result: ModelResult, strategy: str, news_score: float) -> str:
    latest = df.iloc[-1]
    multifactor_score = ml_result.latest_probability
    if strategy == MINI_TRADINGAGENTS_STRATEGY:
        return (
            f"The mini TradingAgents team generated **{signal}** from a manager score of **{multifactor_score:.2f}**. "
            f"Here, the Technical Analyst reviews trend and timing, the Factor Analyst reviews the multi-factor score, "
            f"the News Analyst contributes sentiment, and the Risk Manager can weaken the final decision when volatility "
            f"or downside risk rises. Latest RSI is {latest['rsi']:.1f}, MACD histogram is {latest['macd_hist']:.4f}, "
            f"and short-term volatility is {latest['volatility']:.2%}."
        )
    if strategy == MULTIFACTOR_STRATEGY:
        return (
            f"The multi-factor model generated **{signal}** from a combined score of **{multifactor_score:.2f}**. "
            f"The score blends momentum (`return_3`, `return_6`, `ma_gap`), mean reversion (`RSI`, `z_score`, "
            f"`bb_percent`), and flow/trend confirmation (`volume_spike`, `vwap_gap`, `macd_hist`), with a penalty "
            f"for high short-term volatility. Latest RSI is {latest['rsi']:.1f}, MA gap is {latest['ma_gap']:.2%}, "
            f"and volatility is {latest['volatility']:.2%}."
        )
    if strategy == ML_CLASSIFIER_STRATEGY:
        accuracy_note = (
            f"walk-forward OOS accuracy {ml_result.accuracy:.1%} (F1 {ml_result.f1:.2f})"
            if ml_result.accuracy is not None
            else "no evaluable walk-forward predictions are available yet"
        )
        return (
            f"The {ml_result.model_name} classifier generated **{signal}** from a predicted probability of "
            f"**{multifactor_score:.2f}** that price rises over the chosen horizon ({accuracy_note}). "
            f"Latest RSI is {latest['rsi']:.1f}, MACD histogram is {latest['macd_hist']:.4f}, "
            f"and short-term volatility is {latest['volatility']:.2%}."
        )
    return (
        f"The Freqtrade sample strategy generated **{signal}** using RSI cross conditions with TEMA and the "
        f"Bollinger middle band as trend guards. Latest RSI is {latest['rsi']:.1f}, TEMA is {latest['tema']:.2f}, "
        f"and the Bollinger middle band is {latest['bb_middle']:.2f}."
    )


def format_pct(value: float) -> str:
    return f"{value:.2%}"


def inject_visual_theme() -> None:
    """Apply the light research-console theme without changing app content or behavior."""
    st.markdown(
        """
        <style>
        :root {
            --navy: #17344f;
            --teal: #145b69;
            --ink: #263849;
            --muted: #647789;
            --canvas: #f5f8fa;
            --panel: #ffffff;
            --line: #d7e2e8;
            --table-blue: #dcecf0;
            --amber: #f2b134;
            --amber-soft: #fff3d9;
            --green: #2d806c;
            --green-soft: #e8f5ef;
        }

        .stApp {
            background: var(--canvas);
            color: var(--ink);
        }

        [data-testid="stAppViewContainer"] {
            background:
                linear-gradient(180deg, rgba(220, 236, 240, 0.36) 0%, rgba(245, 248, 250, 0) 280px),
                var(--canvas);
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        .block-container {
            max-width: 1480px;
            padding: 2.25rem 3rem 4rem;
        }

        h1, h2, h3, h4 {
            color: var(--navy) !important;
            letter-spacing: -0.02em;
        }

        h1 {
            margin: 0 0 0.2rem 0;
            padding: 0 0 0.8rem 0;
            border-bottom: 3px solid var(--amber);
            font-size: clamp(2rem, 3vw, 2.85rem) !important;
            font-weight: 800 !important;
        }

        h2, h3 {
            color: var(--teal) !important;
            font-weight: 760 !important;
        }

        h4 {
            font-weight: 720 !important;
        }

        p, li, label, [data-testid="stCaptionContainer"] {
            color: var(--ink);
        }

        [data-testid="stCaptionContainer"] {
            color: var(--muted) !important;
        }

        .section-label {
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--muted);
            margin: 0 0 0.5rem 0.1rem;
        }

        [data-testid="stSidebar"] {
            background: #edf4f6;
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding: 1.15rem 1rem 2rem;
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            margin-top: 1.15rem;
            margin-bottom: 0.7rem;
            padding: 0.35rem 0 0.35rem 0.7rem;
            border-left: 4px solid var(--amber);
            color: var(--navy) !important;
            font-size: 1.05rem !important;
        }

        [data-testid="stSidebar"] [data-testid="stHeading"] {
            margin-bottom: 0.35rem;
        }

        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.62rem;
        }

        [data-testid="stMetric"] {
            min-height: 6.15rem;
            padding: 1rem 1.1rem;
            border: 1px solid var(--line);
            border-top: 3px solid #8fc8d1;
            border-radius: 14px;
            background: var(--panel);
            box-shadow: 0 8px 24px rgba(23, 52, 79, 0.07);
        }

        [data-testid="stMetricLabel"] p {
            color: var(--muted) !important;
            font-size: 0.76rem !important;
            font-weight: 700 !important;
            letter-spacing: 0.04em;
            line-height: 1.12 !important;
            white-space: normal !important;
            text-transform: uppercase;
        }

        [data-testid="stMetricValue"] {
            color: var(--navy) !important;
            overflow: visible !important;
            font-weight: 800 !important;
        }

        [data-testid="stMetricValue"] > div {
            overflow: visible !important;
            white-space: normal !important;
            text-overflow: clip !important;
            font-size: clamp(1.2rem, 2.05vw, 1.85rem) !important;
            line-height: 1.12 !important;
        }

        [data-testid="stMetricDelta"] {
            color: var(--teal) !important;
        }

        [data-testid="stTabs"] {
            margin-top: 1.2rem;
        }

        [data-testid="stTabs"] > div:first-child {
            border-bottom: 1px solid var(--line);
        }

        [data-testid="stTabs"] button[role="tab"] {
            color: var(--muted);
            font-weight: 700;
            padding: 0.72rem 0.82rem;
        }

        [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
            color: var(--navy);
        }

        [data-testid="stTabs"] button[role="tab"] [data-testid="stMarkdownContainer"] p {
            color: inherit;
        }

        [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background: var(--amber);
            height: 3px;
        }

        [data-testid="stPlotlyChart"],
        [data-testid="stDataFrame"],
        [data-testid="stExpander"] {
            border: 1px solid var(--line);
            border-radius: 14px;
            background: var(--panel);
            box-shadow: 0 8px 24px rgba(23, 52, 79, 0.05);
        }

        [data-testid="stPlotlyChart"] {
            padding: 0.3rem 0.35rem 0.1rem;
            overflow: hidden;
        }

        [data-testid="stDataFrame"] {
            overflow: hidden;
        }

        [data-testid="stExpander"] summary {
            color: var(--navy);
            font-weight: 700;
        }

        [data-testid="stAlert"] {
            border-radius: 12px;
            border-width: 1px;
        }

        [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
            color: inherit;
        }

        [data-baseweb="select"] > div,
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input {
            border-color: #c4d5dd;
            border-radius: 9px;
            background: var(--panel);
        }

        [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
            background: var(--teal);
            border-color: var(--teal);
        }

        [data-testid="stSlider"] [data-baseweb="slider"] > div:first-child > div:first-child {
            background: var(--teal) !important;
        }

        [data-testid="stSlider"] [data-testid="stThumbValue"] {
            color: var(--teal) !important;
            background: transparent !important;
            font-size: 0.84rem !important;
            font-weight: 700 !important;
        }

        [data-testid="stSlider"] [data-testid="stTickBarMin"],
        [data-testid="stSlider"] [data-testid="stTickBarMax"] {
            color: var(--muted) !important;
            background: transparent !important;
            font-size: 0.72rem !important;
            font-weight: 600 !important;
        }

        [data-testid="stDownloadButton"] button,
        [data-testid="stButton"] button {
            border: 1px solid #9bc5cc;
            border-radius: 9px;
            color: var(--navy);
            background: #f7fbfc;
            font-weight: 700;
        }

        [data-testid="stDownloadButton"] button:hover,
        [data-testid="stButton"] button:hover {
            border-color: var(--teal);
            color: var(--teal);
        }

        hr {
            border-color: var(--line) !important;
        }

        code {
            color: var(--teal);
            background: var(--table-blue);
            border-radius: 4px;
        }

        @media (max-width: 900px) {
            .block-container {
                padding: 1.35rem 1rem 3rem;
            }

            [data-testid="stMetric"] {
                min-height: 5.25rem;
                padding: 0.8rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_plotly_figure(fig: Any, height: int | None = None) -> Any:
    """Keep every chart aligned with the PDF-inspired navy/teal research-console theme."""
    layout = {
        "template": "plotly_white",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "#fbfdfe",
        "font": {"family": "Arial, sans-serif", "color": "#17344f", "size": 12},
        "hoverlabel": {"bgcolor": "#17344f", "font": {"color": "#ffffff"}},
        "legend": {"bgcolor": "rgba(255,255,255,0.88)", "bordercolor": "#d7e2e8", "borderwidth": 1},
        "margin": {"l": 18, "r": 18, "t": 48, "b": 18},
    }
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)
    fig.update_xaxes(showgrid=True, gridcolor="#e3edf0", linecolor="#cbd9df", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e3edf0", linecolor="#cbd9df", zeroline=False)
    return fig


def main() -> None:
    inject_visual_theme()
    st.title(APP_TITLE)
    st.caption("Educational simulator only. This is not financial advice and does not execute real trades.")

    with st.sidebar:
        st.header("User choices")
        ticker = st.selectbox("Stock ticker", TICKERS, index=0)
        months_label = st.radio("Time range", ["Recent 1 month", "Recent 3 months"], index=0)
        months = 1 if months_label == "Recent 1 month" else 3
        interval_label = st.radio("Candle interval", ["1-minute candles", "5-minute candles"], index=1)
        interval = "1m" if interval_label.startswith("1") else "5m"
        strategy = st.selectbox("Strategy type", STRATEGIES, index=0)
        if strategy == MINI_TRADINGAGENTS_STRATEGY:
            st.caption(
                "Mini TradingAgents: Technical Analyst, Factor Analyst, News Analyst, and Risk Manager feed a final Manager decision."
            )
            st.header("Team tuning")
            technical_weight = st.slider("Technical Analyst", 0.0, 1.0, 0.35, 0.05)
            factor_weight = st.slider("Factor Analyst", 0.0, 1.0, 0.35, 0.05)
            news_weight = st.slider("News Analyst", 0.0, 1.0, 0.15, 0.05)
            risk_weight = st.slider("Risk Manager", 0.0, 1.0, 0.15, 0.05)
            st.header("Factor tuning")
            momentum_weight = st.slider("Momentum weight", 0.0, 1.0, 0.40, 0.05)
            mean_reversion_weight = st.slider("Mean-reversion weight", 0.0, 1.0, 0.35, 0.05)
            flow_weight = st.slider("Flow/trend weight", 0.0, 1.0, 0.25, 0.05)
            volatility_weight = st.slider("Volatility penalty", 0.0, 1.0, 0.35, 0.05)
            buy_threshold = st.slider("Manager buy threshold", 0.10, 1.20, TEAM_BUY_THRESHOLD, 0.05)
            sell_threshold = st.slider("Manager sell threshold", -1.20, -0.10, TEAM_SELL_THRESHOLD, 0.05)
        elif strategy == MULTIFACTOR_STRATEGY:
            st.caption(
                "Multi-factor model: blends momentum, mean reversion, volume-flow confirmation, and volatility filtering."
            )
            technical_weight = 0.35
            factor_weight = 0.35
            news_weight = 0.15
            risk_weight = 0.15
            st.header("Factor tuning")
            momentum_weight = st.slider("Momentum weight", 0.0, 1.0, MULTIFACTOR_DEFAULT_MOMENTUM_WEIGHT, 0.05)
            mean_reversion_weight = st.slider("Mean-reversion weight", 0.0, 1.0, MULTIFACTOR_DEFAULT_MEAN_REVERSION_WEIGHT, 0.05)
            flow_weight = st.slider("Flow/trend weight", 0.0, 1.0, MULTIFACTOR_DEFAULT_FLOW_WEIGHT, 0.05)
            volatility_weight = st.slider("Volatility penalty", 0.0, 1.0, MULTIFACTOR_DEFAULT_VOLATILITY_WEIGHT, 0.05)
            buy_threshold = st.slider("Factor buy threshold", 0.10, 1.20, MULTIFACTOR_BUY_THRESHOLD, 0.05)
            sell_threshold = st.slider("Factor sell threshold", -1.20, -0.10, MULTIFACTOR_SELL_THRESHOLD, 0.05)
        elif strategy == ML_CLASSIFIER_STRATEGY:
            st.caption(
                "ML Classifier: trains a Random Forest, Logistic Regression, or XGBoost model on recent "
                "indicator features to predict the probability that price rises over the chosen horizon."
            )
            technical_weight = 0.35
            factor_weight = 0.35
            news_weight = 0.15
            risk_weight = 0.15
            momentum_weight = 0.40
            mean_reversion_weight = 0.35
            flow_weight = 0.25
            volatility_weight = 0.35
            st.header("Model settings")
            model_choice = st.selectbox("Model", MODEL_CHOICES, index=0)
            horizon_label = st.radio("Prediction horizon", ["5 minutes", "15 minutes"], index=0)
            horizon_minutes = 5 if horizon_label.startswith("5") else 15
            buy_threshold = st.slider("Buy probability threshold", 0.50, 0.90, ML_BUY_PROB_THRESHOLD, 0.01)
            sell_threshold = st.slider("Sell probability threshold", 0.10, 0.50, ML_SELL_PROB_THRESHOLD, 0.01)
        else:
            st.caption(
                "Freqtrade sample strategy: RSI crosses with TEMA and Bollinger middle-band filters."
            )
            technical_weight = 0.35
            factor_weight = 0.35
            news_weight = 0.15
            risk_weight = 0.15
            momentum_weight = 0.40
            mean_reversion_weight = 0.35
            flow_weight = 0.25
            volatility_weight = 0.35
            buy_threshold = MULTIFACTOR_BUY_THRESHOLD
            sell_threshold = MULTIFACTOR_SELL_THRESHOLD

        st.header("Risk control")
        initial_capital = st.number_input("Initial capital", min_value=1000.0, value=100000.0, step=5000.0)
        position_size = st.slider("Position size per trade", 1, 100, 20) / 100
        adaptive_risk = st.checkbox(
            "Volatility-adaptive stop-loss/take-profit",
            value=True,
            help=(
                "Scales each trade's stop-loss/take-profit off the rolling short-term volatility at entry, "
                "instead of a single fixed percentage. A fixed 1% stop is often wider than a typical 5-minute "
                "AAPL move, so most exits end up being time-based rather than real risk control; scaling by "
                "recent volatility keeps the stop meaningful across tickers and candle intervals."
            ),
        )
        if adaptive_risk:
            stop_loss_mult = st.slider("Stop-loss (x recent volatility)", 0.5, 5.0, 1.5, 0.1)
            take_profit_mult = st.slider("Take-profit (x recent volatility)", 0.5, 8.0, 3.0, 0.1)
            stop_loss = take_profit = None
        else:
            stop_loss = st.slider("Stop-loss", 0.1, 5.0, 0.5, 0.1) / 100
            take_profit = st.slider("Take-profit", 0.1, 8.0, 1.0, 0.1) / 100
            stop_loss_mult = take_profit_mult = None
        transaction_cost_bps = st.slider("Transaction cost (bps per side)", 0.0, 20.0, 2.0, 0.5)
        max_hold_bars = st.slider("Max holding bars", 3, 80, 24)

    start = time.time()
    with st.spinner("Loading intraday data and calculating indicators..."):
        raw, data_status = load_price_data(ticker, months, interval)
        data = add_indicators(raw)
        factor_components = build_multifactor_components(
            data,
            momentum_weight,
            mean_reversion_weight,
            flow_weight,
            volatility_weight,
        )
        technical_components = build_technical_components(data)
        news_score, headlines = fetch_news_sentiment(ticker)
        team_components = build_team_components(
            data,
            factor_components,
            technical_components,
            news_score,
            technical_weight,
            factor_weight,
            news_weight,
            risk_weight,
        )
        if strategy == MINI_TRADINGAGENTS_STRATEGY:
            ml_result = ModelResult(
                probability=team_components["manager_score"],
                latest_probability=float(team_components["manager_score"].iloc[-1]),
                accuracy=None,
                f1=None,
                feature_importance=pd.Series(dtype=float),
                model_name=strategy,
                fallback=False,
            )
        elif strategy == MULTIFACTOR_STRATEGY:
            ml_result = ModelResult(
                probability=factor_components["score"],
                latest_probability=float(factor_components["score"].iloc[-1]),
                accuracy=None,
                f1=None,
                feature_importance=pd.Series(dtype=float),
                model_name=strategy,
                fallback=False,
            )
        elif strategy == ML_CLASSIFIER_STRATEGY:
            bar_minutes = 1 if interval == "1m" else 5
            horizon_bars = max(1, round(horizon_minutes / bar_minutes))
            ml_result = train_predict_model(data, model_choice, horizon_bars)
        else:
            ml_result = ModelResult(
                probability=pd.Series(0.5, index=data.index, dtype=float),
                latest_probability=0.5,
                accuracy=None,
                f1=None,
                feature_importance=pd.Series(dtype=float),
                model_name=strategy,
                fallback=False,
            )
        signals = build_strategy_signals(
            data,
            strategy,
            ml_result,
            news_score,
            buy_threshold,
            sell_threshold,
            momentum_weight,
            mean_reversion_weight,
            flow_weight,
            volatility_weight,
            technical_weight,
            factor_weight,
            news_weight,
            risk_weight,
        )
        if adaptive_risk:
            stop_loss_series, take_profit_series = build_adaptive_risk_series(
                data,
                stop_loss_mult,
                take_profit_mult,
                max_hold_bars,
            )
        else:
            stop_loss_series = pd.Series(stop_loss, index=data.index)
            take_profit_series = pd.Series(take_profit, index=data.index)

        trades, equity_curve = backtest(
            data,
            signals,
            initial_capital,
            position_size,
            stop_loss_series,
            take_profit_series,
            transaction_cost_bps,
            max_hold_bars,
            allow_short=False,
        )
        metrics = performance_metrics(trades, equity_curve, initial_capital)
    elapsed = time.time() - start

    st.markdown('<p class="section-label">Run summary</p>', unsafe_allow_html=True)
    with st.container(border=True):
        status_cols = st.columns(4)
        status_cols[0].metric("Ticker", ticker)
        status_cols[1].metric("Bars loaded", f"{len(data):,}")
        status_cols[2].metric("Strategy", strategy)
        status_cols[3].metric("Latest signal", signals.iloc[-1])

        if data_status.startswith("demo"):
            st.warning(f"{data_status}. The app remains runnable for demonstration, but final report metrics should use live data.")
        else:
            st.success(f"{data_status}. Computed in {elapsed:.1f}s.")
        if adaptive_risk:
            st.caption(
                f"Adaptive risk control this run: median stop-loss {stop_loss_series.median():.2%}, "
                f"median take-profit {take_profit_series.median():.2%} (scaled from recent volatility, "
                f"not the flat 1%/2% defaults)."
            )

    st.markdown('<p class="section-label">AI explanation</p>', unsafe_allow_html=True)
    with st.container(border=True):
        st.write(explain_signal(data, signals.iloc[-1], ml_result, strategy, news_score))

    st.markdown('<p class="section-label">Performance</p>', unsafe_allow_html=True)
    with st.container(border=True):
        perf_cols = st.columns(7)
        perf_cols[0].metric("Total return", format_pct(metrics["total_return"]))
        perf_cols[1].metric("Sharpe ratio", f"{metrics['sharpe']:.2f}")
        perf_cols[2].metric("Max drawdown", format_pct(metrics["max_drawdown"]))
        perf_cols[3].metric("Win rate", format_pct(metrics["win_rate"]))
        perf_cols[4].metric("Avg P/L per trade", f"${metrics['avg_profit']:,.2f}")
        perf_cols[5].metric("Number of trades", f"{metrics['trades']:,}")
        pf = metrics["profit_factor"]
        perf_cols[6].metric("Profit factor", "inf" if math.isinf(pf) else f"{pf:.2f}")

    tab_chart, tab_trades, tab_model, tab_breakdown, tab_data, tab_news = st.tabs(
        ["Price & Signals", "Trade Log", "Strategy Rules", "Factor Breakdown", "Data & Indicators", "News Sentiment"]
    )

    with tab_chart:
        if go is None:
            st.error("Plotly is not installed. Run `pip install -r requirements.txt`.")
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data["open"],
                    high=data["high"],
                    low=data["low"],
                    close=data["close"],
                    name="OHLC",
                )
            )
            fig.add_trace(go.Scatter(x=data.index, y=data["vwap"], name="VWAP", line=dict(color="#f59e0b")))
            fig.add_trace(go.Scatter(x=data.index, y=data["bb_upper"], name="BB upper", line=dict(color="#94a3b8", width=1)))
            fig.add_trace(go.Scatter(x=data.index, y=data["bb_lower"], name="BB lower", line=dict(color="#94a3b8", width=1)))
            buy_points = data[signals == "Buy"]
            sell_points = data[signals == "Sell"]
            fig.add_trace(
                go.Scatter(
                    x=buy_points.index,
                    y=buy_points["close"],
                    mode="markers",
                    name="Buy",
                    marker=dict(color="#16a34a", size=8, symbol="triangle-up"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=sell_points.index,
                    y=sell_points["close"],
                    mode="markers",
                    name="Sell",
                    marker=dict(color="#dc2626", size=8, symbol="triangle-down"),
                )
            )
            fig.update_layout(height=620, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
            style_plotly_figure(fig, 620)
            st.plotly_chart(fig, use_container_width=True)

            eq_fig = px.line(equity_curve, y="equity", title="Equity curve") if px else None
            if eq_fig:
                eq_fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
                style_plotly_figure(eq_fig, 360)
                st.plotly_chart(eq_fig, use_container_width=True)

    with tab_trades:
        if trades.empty:
            st.info("No completed trades for the selected settings. Try lowering thresholds or changing strategy.")
        else:
            st.dataframe(
                trades.sort_values("exit_time", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download trade log CSV",
                trades.to_csv(index=False).encode("utf-8"),
                file_name=f"{ticker}_{strategy.replace(' ', '_')}_trades.csv",
                mime="text/csv",
            )

    with tab_model:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Model", ml_result.model_name)
        if strategy == MINI_TRADINGAGENTS_STRATEGY:
            latest_score = team_components["manager_score"].iloc[-1]
            col_b.metric("Buy threshold", f"{buy_threshold:.2f}")
            col_c.metric("Manager score", f"{latest_score:.2f}")
            st.markdown(
                """
                - `Technical Analyst`: studies trend, MACD, VWAP relation, and timing quality.
                - `Factor Analyst`: reviews the multi-factor score from momentum, mean reversion, and flow factors.
                - `News Analyst`: adds a sentiment push from recent headlines.
                - `Risk Manager`: weakens the trade when volatility or downside risk increases.
                - `Manager`: combines the team opinions into the final `Buy / Hold / Sell` signal.
                """
            )
            st.caption(
                f"Current team weights: technical {technical_weight:.2f}, factor {factor_weight:.2f}, "
                f"news {news_weight:.2f}, risk {risk_weight:.2f}."
            )
        elif strategy == MULTIFACTOR_STRATEGY:
            latest_score = factor_components["score"].iloc[-1]
            col_b.metric("Buy threshold", f"{buy_threshold:.2f}")
            col_c.metric("Latest factor score", f"{latest_score:.2f}")
            st.markdown(
                """
                - `Buy`: combined factor score is strong, price is at or above the Bollinger middle band, and volume is valid.
                - `Sell`: combined factor score weakens materially or price loses the Bollinger middle band.
                - Factors used: momentum (`return_3`, `return_6`, `ma_gap`), mean reversion (`RSI`, `bb_percent`, `z_score`), flow/trend (`volume_spike`, `vwap_gap`, `macd_hist`), and a volatility penalty.
                - Backtest mode here is long-only: a sell signal closes longs and does not open a new short.
                """
            )
            st.caption(
                f"Current tuning: momentum {momentum_weight:.2f}, mean reversion {mean_reversion_weight:.2f}, "
                f"flow/trend {flow_weight:.2f}, volatility penalty {volatility_weight:.2f}, "
                f"sell threshold {sell_threshold:.2f}."
            )
        elif strategy == ML_CLASSIFIER_STRATEGY:
            col_b.metric("Walk-forward accuracy", f"{ml_result.accuracy:.1%}" if ml_result.accuracy is not None else "n/a")
            col_c.metric("F1 score", f"{ml_result.f1:.2f}" if ml_result.f1 is not None else "n/a")
            if ml_result.fallback:
                st.warning(f"Classifier fallback/skip: {ml_result.fallback_reason or 'unknown reason'}.")
            label_rate_text = (
                f"{ml_result.oos_label_positive_rate:.1%}"
                if ml_result.oos_label_positive_rate is not None
                else "n/a"
            )
            predicted_rate_text = (
                f"{ml_result.oos_predicted_positive_rate:.1%}"
                if ml_result.oos_predicted_positive_rate is not None
                else "n/a"
            )
            st.caption(
                f"Walk-forward folds trained: {ml_result.walk_forward_folds}; genuine OOS probability coverage: "
                f"{ml_result.prediction_coverage:.1%}; OOS positive-label rate: "
                f"{label_rate_text}; predicted-positive rate at 0.50: {predicted_rate_text}."
            )
            st.markdown(
                f"""
                - Uses a purged walk-forward loop: after an initial {ML_MIN_TRAIN_BARS}-bar warm-up, a fresh model
                  trains on at most {ML_TRAIN_WINDOW_BARS} past labelled bars and predicts the next
                  {ML_RETRAIN_INTERVAL} bars. The final horizon labels before each prediction block are purged.
                - The original label formula is unchanged. Its volatility threshold is estimated from the initial
                  pre-roll training data and frozen before out-of-sample prediction begins.
                - `Buy`: predicted probability of a price rise is at or above the buy threshold and volume is valid.
                - `Sell`: predicted probability drops to or below the sell threshold.
                - Features: {', '.join(FEATURE_COLUMNS)}.
                - Backtest mode here is long-only: a sell signal closes longs and does not open a new short.
                """
            )
            if ml_result.fold_metrics is not None and not ml_result.fold_metrics.empty:
                with st.expander("Walk-forward fold and class-balance diagnostics"):
                    st.dataframe(ml_result.fold_metrics, use_container_width=True, hide_index=True)
            if not ml_result.feature_importance.empty and go is not None:
                importance = ml_result.feature_importance.sort_values()
                fig_importance = go.Figure(go.Bar(x=importance.values, y=importance.index, orientation="h"))
                fig_importance.update_layout(title="Feature importance", height=360, margin=dict(l=10, r=10, t=50, b=10))
                style_plotly_figure(fig_importance, 360)
                st.plotly_chart(fig_importance, use_container_width=True)
        else:
            col_b.metric("Buy RSI cross", f">{FREQTRADE_BUY_RSI}")
            col_c.metric("Sell RSI cross", f">{FREQTRADE_SELL_RSI}")
            st.markdown(
                """
                - `Buy`: RSI crosses above 30, `TEMA <= Bollinger middle`, `TEMA` is rising, and volume is positive.
                - `Sell`: RSI crosses above 70, `TEMA > Bollinger middle`, `TEMA` is falling, and volume is positive.
                - Backtest mode here is long-only: a sell signal closes longs and does not open a new short.
                """
            )

    with tab_breakdown:
        if strategy == MINI_TRADINGAGENTS_STRATEGY:
            latest_team = team_components[
                ["technical_analyst", "factor_analyst", "news_analyst", "risk_manager"]
            ].iloc[-1]
            metric_cols = st.columns(4)
            metric_cols[0].metric("Technical", f"{latest_team['technical_analyst']:.2f}")
            metric_cols[1].metric("Factor", f"{latest_team['factor_analyst']:.2f}")
            metric_cols[2].metric("News", f"{latest_team['news_analyst']:.2f}")
            metric_cols[3].metric("Risk", f"{latest_team['risk_manager']:.2f}")
            if go is not None:
                contrib = (
                    latest_team
                    * pd.Series(
                        {
                            "technical_analyst": technical_weight,
                            "factor_analyst": factor_weight,
                            "news_analyst": news_weight,
                            "risk_manager": risk_weight,
                        }
                    )
                ).sort_values()
                fig_team = go.Figure(
                    go.Bar(
                        x=contrib.values,
                        y=contrib.index,
                        orientation="h",
                        marker_color=["#b91c1c" if v < 0 else "#15803d" for v in contrib.values],
                    )
                )
                fig_team.update_layout(title="Latest team contribution", height=360, margin=dict(l=10, r=10, t=50, b=10))
                style_plotly_figure(fig_team, 360)
                st.plotly_chart(fig_team, use_container_width=True)

                team_ts = go.Figure()
                team_ts.add_trace(go.Scatter(x=team_components.index, y=team_components["manager_score"], name="Manager score"))
                team_ts.add_hline(y=buy_threshold, line_dash="dash", line_color="green")
                team_ts.add_hline(y=sell_threshold, line_dash="dash", line_color="red")
                team_ts.update_layout(title="Manager score over time", height=360, margin=dict(l=10, r=10, t=50, b=10))
                style_plotly_figure(team_ts, 360)
                st.plotly_chart(team_ts, use_container_width=True)
        elif strategy == MULTIFACTOR_STRATEGY:
            latest_breakdown = factor_components[["momentum", "mean_reversion", "flow_trend", "volatility_penalty"]].iloc[-1]
            metric_cols = st.columns(4)
            metric_cols[0].metric("Momentum", f"{latest_breakdown['momentum']:.2f}")
            metric_cols[1].metric("Mean reversion", f"{latest_breakdown['mean_reversion']:.2f}")
            metric_cols[2].metric("Flow/trend", f"{latest_breakdown['flow_trend']:.2f}")
            metric_cols[3].metric("Volatility", f"{latest_breakdown['volatility_penalty']:.2f}")
            if go is not None:
                contrib = (latest_breakdown * pd.Series(
                    {
                        "momentum": momentum_weight,
                        "mean_reversion": mean_reversion_weight,
                        "flow_trend": flow_weight,
                        "volatility_penalty": volatility_weight,
                    }
                )).sort_values()
                fig_breakdown = go.Figure(
                    go.Bar(
                        x=contrib.values,
                        y=contrib.index,
                        orientation="h",
                        marker_color=["#b91c1c" if v < 0 else "#15803d" for v in contrib.values],
                    )
                )
                fig_breakdown.update_layout(title="Latest factor contribution", height=360, margin=dict(l=10, r=10, t=50, b=10))
                style_plotly_figure(fig_breakdown, 360)
                st.plotly_chart(fig_breakdown, use_container_width=True)

                ts_fig = go.Figure()
                ts_fig.add_trace(go.Scatter(x=factor_components.index, y=factor_components["score"], name="Factor score"))
                ts_fig.add_hline(y=buy_threshold, line_dash="dash", line_color="green")
                ts_fig.add_hline(y=sell_threshold, line_dash="dash", line_color="red")
                ts_fig.update_layout(title="Factor score over time", height=360, margin=dict(l=10, r=10, t=50, b=10))
                style_plotly_figure(ts_fig, 360)
                st.plotly_chart(ts_fig, use_container_width=True)
        elif strategy == ML_CLASSIFIER_STRATEGY:
            if go is not None:
                prob_fig = go.Figure()
                prob_fig.add_trace(go.Scatter(x=ml_result.probability.index, y=ml_result.probability, name="P(price up)"))
                prob_fig.add_hline(y=buy_threshold, line_dash="dash", line_color="green")
                prob_fig.add_hline(y=sell_threshold, line_dash="dash", line_color="red")
                prob_fig.update_layout(title="Predicted probability over time", height=360, margin=dict(l=10, r=10, t=50, b=10))
                style_plotly_figure(prob_fig, 360)
                st.plotly_chart(prob_fig, use_container_width=True)
            if not ml_result.feature_importance.empty:
                st.dataframe(ml_result.feature_importance.rename("importance"), use_container_width=True)
        else:
            st.info("Factor breakdown is available for the multi-factor model. Switch strategy to view it.")

    with tab_data:
        st.markdown("#### Latest calculated indicators")
        latest_cols = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "rsi",
            "tema",
            "bb_middle",
            "macd",
            "macd_hist",
            "bb_upper",
            "bb_lower",
            "volatility",
            "volume_spike",
            "ma_gap",
            "z_score",
            "bb_percent",
        ]
        st.dataframe(data[latest_cols].tail(200), use_container_width=True)

    with tab_news:
        st.metric("Headline sentiment score", f"{news_score:.2f}")
        for headline in headlines:
            st.write(f"- {headline}")


if __name__ == "__main__":
    main()
