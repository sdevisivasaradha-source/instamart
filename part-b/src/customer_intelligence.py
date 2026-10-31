"""Part B: customer segmentation and leakage-aware lapse prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
SNAPSHOT_DATE = pd.Timestamp("2025-10-01")
LAPSE_CUTOFF = pd.Timestamp("2025-09-01")
LAPSE_END = pd.Timestamp("2025-10-01")


def load_clean_data(
    processed_dir: Path = PROCESSED_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load typed clean customer, product, and order data."""
    paths = {
        name: processed_dir / f"{name}.csv"
        for name in ("customers", "products", "orders")
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing processed datasets: {missing}. Run `python -m src.data_pipeline`."
        )
    customers = pd.read_csv(paths["customers"])
    products = pd.read_csv(paths["products"])
    orders = pd.read_csv(paths["orders"])
    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], format="%Y-%m-%d %H:%M:%S", errors="raise"
    )
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], format="%Y-%m-%d", errors="raise"
    )
    return customers, products, orders


def build_rfm(
    orders: pd.DataFrame, snapshot_date: pd.Timestamp = SNAPSHOT_DATE
) -> pd.DataFrame:
    """Build recency, frequency, and monetary features from completed orders."""
    completed = orders[
        orders["status"].eq("completed") & orders["order_ts"].lt(snapshot_date)
    ]
    rfm = (
        completed.groupby("customer_id", as_index=False)
        .agg(
            last_order=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
        )
        .assign(recency=lambda frame: (snapshot_date - frame["last_order"]).dt.days)
        .drop(columns="last_order")
    )
    rfm["monetary"] = rfm["monetary"].round(2)
    if rfm.empty:
        raise ValueError("RFM requires at least one completed order")
    return rfm[["customer_id", "recency", "frequency", "monetary"]]


def segment_customers(
    orders: pd.DataFrame,
    k: int = 4,
    snapshot_date: pd.Timestamp = SNAPSHOT_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Fit K-means, label clusters by value, and return diagnostics."""
    if k < 2:
        raise ValueError("k must be at least 2")
    rfm = build_rfm(orders, snapshot_date)
    feature_columns = ["recency", "frequency", "monetary"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[feature_columns])

    diagnostics = []
    for candidate in range(2, 9):
        model = KMeans(n_clusters=candidate, n_init=20, random_state=42)
        labels = model.fit_predict(scaled)
        diagnostics.append(
            {
                "k": candidate,
                "inertia": round(float(model.inertia_), 4),
                "silhouette": round(float(silhouette_score(scaled, labels)), 4),
            }
        )

    model = KMeans(n_clusters=k, n_init=20, random_state=42)
    segments = rfm.copy()
    segments["segment_id"] = model.fit_predict(scaled)
    profile = (
        segments.groupby("segment_id", as_index=False)
        .agg(
            customers=("customer_id", "size"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
        )
        .round(2)
    )
    profile["value_score"] = (
        profile["avg_monetary"].rank(pct=True)
        + profile["avg_frequency"].rank(pct=True)
        + profile["avg_recency"].rank(pct=True, ascending=False)
    )
    ordered_ids = profile.sort_values(
        ["value_score", "avg_monetary"], ascending=False
    )["segment_id"].tolist()
    names = ["Premium", "Loyal", "Promising", "At Risk"]
    name_map = {
        segment_id: names[index] if index < len(names) else f"Segment {index + 1}"
        for index, segment_id in enumerate(ordered_ids)
    }
    segments["segment_name"] = segments["segment_id"].map(name_map)
    segments["is_premium"] = segments["segment_name"].eq("Premium")
    profile["segment_name"] = profile["segment_id"].map(name_map)
    profile = profile.sort_values("value_score", ascending=False).reset_index(drop=True)
    pca_variance = PCA(n_components=2).fit(scaled).explained_variance_ratio_
    return (
        segments,
        pd.DataFrame(diagnostics),
        profile,
        pca_variance,
    )


def build_lapse_cohort(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    orders: pd.DataFrame,
    cutoff: pd.Timestamp = LAPSE_CUTOFF,
    end: pd.Timestamp = LAPSE_END,
) -> pd.DataFrame:
    """Create features strictly before cutoff and a future-window lapse label."""
    history = orders[orders["order_ts"].lt(cutoff)].copy()
    completed_history = history[history["status"].eq("completed")].copy()
    cohort = (
        completed_history.groupby("customer_id", as_index=False)
        .agg(
            last_order=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
            avg_discount=("discount_pct", "mean"),
            avg_delivery=("delivery_minutes", "mean"),
        )
        .assign(recency=lambda frame: (cutoff - frame["last_order"]).dt.days)
        .drop(columns="last_order")
    )

    order_status = (
        history.groupby("customer_id")["status"]
        .agg(
            total_orders="count",
            cancelled_orders=lambda values: int(values.eq("cancelled").sum()),
        )
        .reset_index()
    )
    cohort = cohort.merge(order_status, on="customer_id", how="left")
    cohort["cancellation_rate"] = (
        cohort["cancelled_orders"] / cohort["total_orders"]
    )

    category_counts = (
        completed_history.merge(
            products[["product_id", "category"]], on="product_id", how="left"
        )
        .groupby(["customer_id", "category"], as_index=False)
        .size()
        .sort_values(
            ["customer_id", "size", "category"], ascending=[True, False, True]
        )
    )
    favourite = category_counts.drop_duplicates("customer_id").rename(
        columns={"category": "favourite_category"}
    )[["customer_id", "favourite_category"]]
    cohort = cohort.merge(favourite, on="customer_id", how="left")
    cohort = cohort.merge(customers, on="customer_id", how="left")
    cohort["tenure_days"] = (cutoff - cohort["signup_date"]).dt.days

    future_customers = set(
        orders.loc[
            orders["order_ts"].between(cutoff, end, inclusive="left")
            & orders["status"].eq("completed"),
            "customer_id",
        ]
    )
    cohort["lapsed"] = (~cohort["customer_id"].isin(future_customers)).astype(int)
    return cohort.drop(columns=["total_orders", "cancelled_orders"])


def _model_specifications(
    numeric: list[str], categorical: list[str]
) -> dict[str, tuple[Pipeline, dict[str, list[object]]]]:
    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                categorical,
            ),
        ]
    )
    return {
        "logistic_regression": (
            Pipeline(
                [
                    ("preprocessing", preprocessing),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000, class_weight="balanced", random_state=42
                        ),
                    ),
                ]
            ),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            Pipeline(
                [
                    ("preprocessing", preprocessing),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=200,
                            class_weight="balanced",
                            random_state=42,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
            {
                "model__max_depth": [4, 8, None],
                "model__min_samples_leaf": [3, 10],
            },
        ),
    }


def train_lapse_model(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    orders: pd.DataFrame,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Train and evaluate two models using a held-out stratified test set."""
    cohort = build_lapse_cohort(customers, products, orders)
    feature_columns = [
        "recency",
        "frequency",
        "monetary",
        "avg_discount",
        "cancellation_rate",
        "avg_delivery",
        "favourite_category",
        "city",
        "age",
        "gender",
        "tenure_days",
    ]
    numeric = [
        "recency",
        "frequency",
        "monetary",
        "avg_discount",
        "cancellation_rate",
        "avg_delivery",
        "age",
        "tenure_days",
    ]
    categorical = ["favourite_category", "city", "gender"]
    features = cohort[feature_columns].copy()
    features[numeric] = features[numeric].fillna(features[numeric].median())
    features[categorical] = features[categorical].fillna("Unknown")
    labels = cohort["lapsed"]

    train_x, test_x, train_y, test_y = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    evaluations = []
    searches = {}
    for name, (pipeline, grid) in _model_specifications(
        numeric, categorical
    ).items():
        search = GridSearchCV(
            pipeline,
            grid,
            cv=5,
            scoring="f1",
            n_jobs=1,
            return_train_score=False,
        )
        search.fit(train_x, train_y)
        predictions = search.predict(test_x)
        probabilities = search.predict_proba(test_x)[:, 1]
        evaluations.append(
            {
                "model": name,
                "best_parameters": json.dumps(search.best_params_, sort_keys=True),
                "cv_f1": round(float(search.best_score_), 4),
                "test_accuracy": round(float(accuracy_score(test_y, predictions)), 4),
                "test_precision": round(
                    float(precision_score(test_y, predictions, zero_division=0)), 4
                ),
                "test_recall": round(
                    float(recall_score(test_y, predictions, zero_division=0)), 4
                ),
                "test_f1": round(
                    float(f1_score(test_y, predictions, zero_division=0)), 4
                ),
                "test_roc_auc": round(
                    float(roc_auc_score(test_y, probabilities)), 4
                ),
                "confusion_matrix": json.dumps(
                    confusion_matrix(test_y, predictions).tolist()
                ),
            }
        )
        searches[name] = search

    # Model selection uses training-fold CV only; held-out test metrics remain unbiased.
    evaluation = pd.DataFrame(evaluations).sort_values(
        ["cv_f1", "model"], ascending=[False, True]
    )
    selected_name = str(evaluation.iloc[0]["model"])
    selected = searches[selected_name].best_estimator_
    test_probabilities = selected.predict_proba(test_x)[:, 1]
    scored = cohort.loc[test_x.index, ["customer_id", "lapsed"]].copy()
    scored["lapse_probability"] = test_probabilities
    scored = scored.sort_values("lapse_probability", ascending=False)
    metadata = {
        "cohort_rows": len(cohort),
        "lapsed_customers": int(labels.sum()),
        "lapse_rate": round(float(labels.mean()), 4),
        "training_rows": len(train_x),
        "test_rows": len(test_x),
        "selected_model": selected_name,
        "feature_cutoff": str(LAPSE_CUTOFF.date()),
        "label_window_end": str(LAPSE_END.date()),
    }
    return selected, evaluation.reset_index(drop=True), scored, metadata


def run_customer_intelligence(
    processed_dir: Path = PROCESSED_DIR,
    output_dir: Path = OUTPUT_DIR,
    model_dir: Path = MODEL_DIR,
) -> dict[str, object]:
    """Run segmentation and lapse modelling and persist reproducible evidence."""
    customers, products, orders = load_clean_data(processed_dir)
    segments, diagnostics, profile, pca_variance = segment_customers(orders)
    model, evaluation, scored, metadata = train_lapse_model(
        customers, products, orders
    )
    premium_ids = set(
        segments.loc[segments["is_premium"], "customer_id"].astype(str)
    )
    retention = scored[scored["customer_id"].isin(premium_ids)].head(10).copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "segments": output_dir / "customer_segments.csv",
        "diagnostics": output_dir / "segmentation_diagnostics.csv",
        "profile": output_dir / "segment_profile.csv",
        "evaluation": output_dir / "lapse_model_evaluation.csv",
        "retention": output_dir / "retention_recommendations.csv",
        "metadata": output_dir / "customer_intelligence_summary.json",
        "model": model_dir / "lapse_model.joblib",
    }
    segments.to_csv(paths["segments"], index=False)
    diagnostics.to_csv(paths["diagnostics"], index=False)
    profile.to_csv(paths["profile"], index=False)
    evaluation.to_csv(paths["evaluation"], index=False)
    retention.to_csv(paths["retention"], index=False)
    metadata["pca_explained_variance"] = [
        round(float(value), 4) for value in pca_variance
    ]
    metadata["segment_counts"] = {
        str(key): int(value)
        for key, value in segments["segment_name"].value_counts().items()
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    joblib.dump(model, paths["model"])
    reloaded = joblib.load(paths["model"])
    if not np.allclose(
        model.predict_proba(
            build_lapse_cohort(customers, products, orders)[
                [
                    "recency",
                    "frequency",
                    "monetary",
                    "avg_discount",
                    "cancellation_rate",
                    "avg_delivery",
                    "favourite_category",
                    "city",
                    "age",
                    "gender",
                    "tenure_days",
                ]
            ]
            .assign(
                favourite_category=lambda frame: frame[
                    "favourite_category"
                ].fillna("Unknown")
            )
            .fillna(0)
            .head(5)
        ),
        reloaded.predict_proba(
            build_lapse_cohort(customers, products, orders)[
                [
                    "recency",
                    "frequency",
                    "monetary",
                    "avg_discount",
                    "cancellation_rate",
                    "avg_delivery",
                    "favourite_category",
                    "city",
                    "age",
                    "gender",
                    "tenure_days",
                ]
            ]
            .assign(
                favourite_category=lambda frame: frame[
                    "favourite_category"
                ].fillna("Unknown")
            )
            .fillna(0)
            .head(5)
        ),
    ):
        raise AssertionError("Persisted model predictions changed after reload")
    return {
        "segments": segments,
        "diagnostics": diagnostics,
        "profile": profile,
        "evaluation": evaluation,
        "retention": retention,
        "metadata": metadata,
        "paths": paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    result = run_customer_intelligence(
        args.processed_dir, args.output_dir, args.model_dir
    )
    print(result["profile"].to_string(index=False))
    print("\nModel evaluation:")
    print(result["evaluation"].to_string(index=False))


if __name__ == "__main__":
    main()
