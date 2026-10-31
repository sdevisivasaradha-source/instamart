from pathlib import Path

import pandas as pd

from src.eda import (
    category_variants,
    column_profile,
    duplicate_summary,
    load_raw_data,
    quality_findings,
    shape_summary,
    value_count_summary,
    write_eda_outputs,
)


def test_loads_expected_assignment_datasets() -> None:
    datasets = load_raw_data()

    summary = shape_summary(datasets).set_index("dataset")
    assert summary.to_dict("index") == {
        "customers": {"rows": 1200, "columns": 5},
        "orders": {"rows": 13084, "columns": 9},
        "tickets": {"rows": 164, "columns": 5},
    }


def test_profiles_preserve_and_count_blanks() -> None:
    profile = column_profile(load_raw_data()).set_index(["dataset", "column"])

    assert profile.loc[("customers", "age"), "blank_count"] == 61
    assert profile.loc[("customers", "gender"), "blank_count"] == 30
    assert profile.loc[("orders", "delivery_minutes"), "blank_count"] == 2541
    assert profile.loc[("tickets", "text"), "blank_count"] == 4


def test_value_count_rows_cover_each_column() -> None:
    datasets = load_raw_data()
    counts = value_count_summary(datasets)

    for (dataset, column), rows in counts.groupby(["dataset", "column"]):
        assert rows["count"].sum() == len(datasets[dataset]), column
    status = counts[
        (counts["dataset"] == "orders") & (counts["column"] == "status")
    ].set_index("value")
    assert status.loc["completed", "count"] == 11217
    assert status.loc["completed", "percentage"] == 85.73


def test_duplicate_summary_separates_exact_and_key_duplicates() -> None:
    summary = duplicate_summary(load_raw_data()).set_index("dataset")

    assert summary.loc["customers", "exact_duplicate_rows"] == 0
    assert summary.loc["orders", "exact_duplicate_rows"] == 119
    assert summary.loc["orders", "unique_keys"] == 12309
    assert summary.loc["orders", "duplicate_key_excess_rows"] == 775
    assert summary.loc["orders", "rows_with_repeated_key"] == 1547
    assert summary.loc["tickets", "duplicate_key_excess_rows"] == 0


def test_quality_findings_match_generated_anomalies() -> None:
    findings = quality_findings(load_raw_data()).set_index(
        ["dataset", "column", "finding"]
    )

    expected = {
        ("customers", "city", "raw spelling variants"): 24,
        ("customers", "city", "normalized candidate values"): 6,
        ("customers", "age", "domain outliers"): 6,
        ("orders", "order_amount", "nonnumeric values"): 120,
        ("orders", "delivery_minutes", "negative values"): 106,
        ("tickets", "text", "over 1000 characters"): 4,
    }
    for key, count in expected.items():
        assert findings.loc[key, "count"] == count


def test_category_variant_percentages_cover_customers() -> None:
    variants = category_variants(load_raw_data())

    counts = variants.groupby("column")["count"].sum()
    assert counts.to_dict() == {"city": 1200, "gender": 1200}
    assert variants[variants["column"] == "city"]["raw_value"].nunique() == 24
    assert variants[variants["column"] == "gender"]["raw_value"].nunique() == 11


def test_writes_assignment_evidence(tmp_path: Path) -> None:
    paths = write_eda_outputs(load_raw_data(), tmp_path)

    assert set(paths) == {
        "column_profile",
        "value_counts",
        "duplicate_summary",
        "quality_findings",
        "category_variants",
        "report",
    }
    assert all(path.is_file() for path in paths.values())
    report = paths["report"].read_text(encoding="utf-8")
    assert "customers | 1200 | 5" in report
    assert "cleaning stage" in report
    profile = pd.read_csv(paths["column_profile"])
    assert len(profile) == 19
