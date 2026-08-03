"""Reusable, public-safe analysis helpers for the completed weekly notebooks.

This module uses only the bundled anonymized starter slice.  Its label is an
observed current-window decline proxy, so it is a teaching/capstone scaffold,
not a claim about future Google rankings.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "content_refresh_anonymized.csv"
OUT = ROOT / "work" / "outputs"
FIG = ROOT / "work" / "figures"
RANDOM_STATE = 42

NUMERIC = [
    "search_volume", "competition", "cpc", "word_count", "char_count",
    "content_age_days", "days_since_last_update", "impressions_90d",
    "clicks_90d", "sessions_90d", "engagement_rate", "ctr", "avg_position",
    "days_with_impressions", "days_with_sessions", "scroll_rate",
]
CATEGORICAL = ["content_type", "main_intent", "competition_level", "freshness_tier"]
EXCLUDED = ["content_id", "client_id", "trend_direction", "trend_pct", "provider_used", "model_used"]


def load_frame() -> pd.DataFrame:
    """Load starter data and add safe, pre-defined features and proxy label."""
    df = pd.read_csv(RAW)
    df["is_declining_label"] = (df["trend_direction"] == "down").astype(int)
    df["log_impressions_90d"] = np.log1p(df["impressions_90d"])
    df["log_sessions_90d"] = np.log1p(df["sessions_90d"])
    df["has_keyword_context"] = df["search_volume"].notna().astype(int)
    return df


def feature_columns() -> tuple[list[str], list[str]]:
    return NUMERIC + ["log_impressions_90d", "log_sessions_90d", "has_keyword_context"], CATEGORICAL


def baseline_score(df: pd.DataFrame) -> pd.DataFrame:
    """Transparent review-first score; it never consumes the label inputs."""
    result = df.copy()
    visible = (result["impressions_90d"] >= 300).astype(int)
    stale = (result["days_since_last_update"] >= 180).astype(int)
    weak_ctr = ((result["ctr"] <= 1.0) & (result["impressions_90d"] >= 300)).astype(int)
    page_two = ((result["avg_position"] >= 10) & (result["avg_position"] <= 30)).astype(int)
    result["baseline_score"] = 45 * visible * stale + 25 * weak_ctr + 15 * page_two
    result["baseline_reason"] = np.select(
        [visible.astype(bool) & stale.astype(bool) & weak_ctr.astype(bool), visible.astype(bool) & stale.astype(bool), weak_ctr.astype(bool)],
        ["visible_stale_low_ctr", "visible_and_stale", "visible_low_ctr"], default="monitor"
    )
    return result


def precision_at_k(y: pd.Series | np.ndarray, scores: pd.Series | np.ndarray, k: int = 50) -> float:
    y = np.asarray(y)
    scores = np.asarray(scores)
    return float(y[np.argsort(-scores)[: min(k, len(y))]].mean())


def split_frame(df: pd.DataFrame):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=df["client_id"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def make_pipeline(model):
    numeric, categorical = feature_columns()
    pre = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocess", pre), ("model", model)])


def run_models() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    df = baseline_score(load_frame())
    train, test = split_frame(df)
    numeric, categorical = feature_columns()
    x_train, x_test = train[numeric + categorical], test[numeric + categorical]
    y_train, y_test = train["is_declining_label"], test["is_declining_label"]
    models = {
        "logistic_regression": make_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "random_forest": make_pipeline(RandomForestClassifier(n_estimators=100, min_samples_leaf=8, random_state=RANDOM_STATE, n_jobs=-1)),
    }
    rows, fitted = [], {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_test)[:, 1]
        rows.append({"method": name, "roc_auc": roc_auc_score(y_test, prob), "average_precision": average_precision_score(y_test, prob), "precision_at_20": precision_at_k(y_test, prob, 20), "precision_at_50": precision_at_k(y_test, prob, 50), "base_rate": float(y_test.mean())})
        fitted[name] = model
    rows.append({"method": "transparent_baseline", "roc_auc": roc_auc_score(y_test, test["baseline_score"]), "average_precision": average_precision_score(y_test, test["baseline_score"]), "precision_at_20": precision_at_k(y_test, test["baseline_score"], 20), "precision_at_50": precision_at_k(y_test, test["baseline_score"], 50), "base_rate": float(y_test.mean())})
    metrics = pd.DataFrame(rows).sort_values("precision_at_50", ascending=False)
    best_name = metrics.loc[metrics["method"].isin(fitted), "method"].iloc[0]
    best = fitted[best_name]
    test = test.copy()
    test["model_probability"] = best.predict_proba(x_test)[:, 1]
    test["review_score"] = 100 * (0.8 * test["model_probability"] + 0.2 * (test["baseline_score"] / max(test["baseline_score"].max(), 1)))
    test["reason_code"] = np.where(test["baseline_reason"] != "monitor", test["baseline_reason"] + "|model_risk", "model_risk")
    return metrics, {"test_rows": int(len(test)), "test_clients": int(test["client_id"].nunique()), "train_clients": int(train["client_id"].nunique())}, test


def export_artifacts() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    metrics, split, test = run_models()
    safe_queue = test.sort_values("review_score", ascending=False)[["review_score", "model_probability", "baseline_score", "baseline_reason", "reason_code", "impressions_90d", "sessions_90d", "ctr", "avg_position"]].head(100).copy()
    safe_queue.insert(0, "rank", range(1, len(safe_queue) + 1))
    safe_queue.to_csv(OUT / "ranked_recommendations.csv", index=False)
    payload = {"scope": "bundled anonymized starter slice; current-window decline proxy", "random_state": RANDOM_STATE, "split": split, "metrics": metrics.round(4).to_dict(orient="records"), "excluded_features": EXCLUDED}
    (OUT / "capstone_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    ax = metrics.set_index("method")[["precision_at_20", "precision_at_50"]].plot.bar(rot=20, ylim=(0, 1), color=["#2266AA", "#38A89D"])
    ax.set_ylabel("Precision"); ax.set_title("Model and baseline precision on client holdout")
    plt.tight_layout(); plt.savefig(FIG / "model_vs_baseline.svg", format="svg"); plt.close()
    return metrics, split, safe_queue
