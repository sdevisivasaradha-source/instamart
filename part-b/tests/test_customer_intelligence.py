from pathlib import Path

import joblib

from src.customer_intelligence import (
    build_lapse_cohort,
    build_rfm,
    load_clean_data,
    run_customer_intelligence,
    segment_customers,
)


def test_rfm_and_segmentation_are_complete() -> None:
    _, _, orders = load_clean_data()
    rfm = build_rfm(orders)
    segments, diagnostics, profile, pca_variance = segment_customers(orders)

    assert len(rfm) == orders.loc[orders["status"].eq("completed"), "customer_id"].nunique()
    assert len(segments) == len(rfm)
    assert diagnostics["k"].tolist() == list(range(2, 9))
    assert set(profile["segment_name"]) == {
        "At Risk",
        "Loyal",
        "Premium",
        "Promising",
    }
    assert profile["customers"].sum() == len(segments)
    assert len(pca_variance) == 2
    assert 0 < pca_variance.sum() <= 1


def test_lapse_features_do_not_use_label_window() -> None:
    customers, products, orders = load_clean_data()
    cohort = build_lapse_cohort(customers, products, orders)

    assert cohort["customer_id"].is_unique
    assert set(cohort["lapsed"]) == {0, 1}
    assert cohort["recency"].ge(0).all()
    assert cohort["tenure_days"].ge(0).all()
    assert not {"last_order", "total_orders", "cancelled_orders"} & set(cohort.columns)


def test_customer_intelligence_writes_model_and_evidence(tmp_path: Path) -> None:
    result = run_customer_intelligence(
        output_dir=tmp_path / "outputs", model_dir=tmp_path / "models"
    )

    assert all(path.is_file() for path in result["paths"].values())
    assert result["evaluation"]["test_f1"].between(0, 1).all()
    assert result["metadata"]["feature_cutoff"] == "2025-09-01"
    assert len(result["retention"]) <= 10
    assert hasattr(joblib.load(result["paths"]["model"]), "predict_proba")
