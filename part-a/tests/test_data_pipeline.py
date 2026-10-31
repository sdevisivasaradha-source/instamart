import sqlite3
from pathlib import Path

import pandas as pd

from src.data_pipeline import (
    clean_data,
    cleaning_metrics,
    load_source_data,
    pandas_sql_comparison,
    run_pipeline,
    validate_clean_data,
    write_sqlite,
)


def test_cleaning_produces_expected_shapes_and_categories() -> None:
    clean = clean_data(load_source_data())

    assert validate_clean_data(clean) == {
        "customers": 1200,
        "products": 40,
        "orders": 12309,
        "tickets": 164,
    }
    assert set(clean["customers"]["city"]) == {
        "Bengaluru",
        "Chennai",
        "Delhi",
        "Hyderabad",
        "Mumbai",
        "Pune",
    }
    assert set(clean["customers"]["gender"]) == {"Female", "Male", "Unknown"}
    assert clean["customers"]["age"].between(18, 60).all()


def test_order_cleaning_resolves_duplicates_and_invalid_numbers() -> None:
    clean = clean_data(load_source_data())
    orders = clean["orders"]

    assert orders["order_id"].is_unique
    assert not orders["status"].eq("pending").any()
    assert orders["order_amount"].notna().all()
    assert pd.api.types.is_numeric_dtype(orders["order_amount"])
    completed = orders["status"].eq("completed")
    assert orders.loc[completed, "delivery_minutes"].notna().all()
    assert orders.loc[completed, "delivery_minutes"].ge(0).all()
    assert orders.loc[~completed, "delivery_minutes"].isna().all()


def test_ticket_guardrail_cases_are_intentionally_retained() -> None:
    clean = clean_data(load_source_data())
    tickets = clean["tickets"]

    assert tickets["order_id"].eq("").sum() == 8
    assert tickets["text"].eq("").sum() == 4
    assert tickets["text"].str.len().gt(1000).sum() == 4


def test_cleaning_metrics_document_before_and_after_counts() -> None:
    raw = load_source_data()
    clean = clean_data(raw)
    metrics = cleaning_metrics(raw, clean).set_index(["dataset", "metric"])

    assert metrics.loc[("customers", "city variants"), ["before", "after"]].tolist() == [
        24,
        6,
    ]
    assert metrics.loc[
        ("customers", "invalid or blank ages"), ["before", "after"]
    ].tolist() == [67, 0]
    assert metrics.loc[
        ("customers", "gender values including blanks"), ["before", "after"]
    ].tolist() == [11, 3]
    assert metrics.loc[
        ("orders", "repeated order ID excess"), ["before", "after"]
    ].tolist() == [775, 0]
    assert metrics.loc[
        ("orders", "nonnumeric amounts"), ["before", "after"]
    ].tolist() == [120, 0]


def test_sqlite_has_constraints_and_matching_metrics(tmp_path: Path) -> None:
    clean = clean_data(load_source_data())
    db_path = write_sqlite(clean, tmp_path / "test.db")
    comparison = pandas_sql_comparison(clean, db_path)

    assert comparison["matches"].all()
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert tables == {"customers", "orders", "products", "tickets"}
        assert connection.execute(
            "SELECT COUNT(*) FROM orders WHERE delivery_minutes < 0"
        ).fetchone()[0] == 0


def test_pipeline_writes_all_submission_evidence(tmp_path: Path) -> None:
    result = run_pipeline(
        processed_dir=tmp_path / "processed",
        db_path=tmp_path / "instamart.db",
        output_dir=tmp_path / "outputs",
    )

    assert all(path.is_file() for path in result["csv_paths"].values())
    assert all(path.is_file() for path in result["output_paths"].values())
    assert result["db_path"].is_file()
    summary = (tmp_path / "outputs" / "cleaning_validation.json").read_text()
    assert '"all_pandas_sql_metrics_match": true' in summary
