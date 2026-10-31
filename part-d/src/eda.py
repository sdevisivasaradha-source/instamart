"""Stage 3 exploratory data analysis for the raw synthetic datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
DATASET_FILES = {
    "customers": "customers.csv",
    "orders": "orders.csv",
    "tickets": "tickets.csv",
}
DATASET_KEYS = {
    "customers": "customer_id",
    "orders": "order_id",
    "tickets": "ticket_id",
}


def load_raw_data(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Load assignment datasets without converting intentional blanks to NaN."""
    datasets = {}
    for name, filename in DATASET_FILES.items():
        path = raw_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing raw dataset: {path}. Run `python generate_data.py` first."
            )
        datasets[name] = pd.read_csv(path, keep_default_na=False)
    return datasets


def shape_summary(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Return row and column counts for each dataset."""
    return pd.DataFrame(
        [
            {"dataset": name, "rows": len(frame), "columns": len(frame.columns)}
            for name, frame in datasets.items()
        ]
    )


def _blank_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype(str).str.strip().eq("")


def column_profile(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Profile inferred types, blank values, and cardinality by column."""
    records = []
    for dataset, frame in datasets.items():
        for column in frame.columns:
            blanks = _blank_mask(frame[column])
            records.append(
                {
                    "dataset": dataset,
                    "column": column,
                    "dtype": str(frame[column].dtype),
                    "blank_count": int(blanks.sum()),
                    "blank_pct": round(float(blanks.mean() * 100), 2),
                    "unique_non_blank": int(frame.loc[~blanks, column].nunique()),
                }
            )
    return pd.DataFrame(records)


def value_count_summary(
    datasets: Mapping[str, pd.DataFrame], top_n: int = 10
) -> pd.DataFrame:
    """Return value counts and percentages, grouping high-cardinality tails."""
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    records = []
    for dataset, frame in datasets.items():
        for column in frame.columns:
            values = frame[column].astype(object).where(~_blank_mask(frame[column]), "<BLANK>")
            counts = values.value_counts(dropna=False)
            shown = counts if len(counts) <= top_n else counts.iloc[:top_n]
            for value, count in shown.items():
                records.append(
                    {
                        "dataset": dataset,
                        "column": column,
                        "value": str(value),
                        "count": int(count),
                        "percentage": round(float(count / len(frame) * 100), 2),
                    }
                )
            if len(counts) > top_n:
                other_count = int(counts.iloc[top_n:].sum())
                records.append(
                    {
                        "dataset": dataset,
                        "column": column,
                        "value": f"<OTHER: {len(counts) - top_n} values>",
                        "count": other_count,
                        "percentage": round(float(other_count / len(frame) * 100), 2),
                    }
                )
    return pd.DataFrame(records)


def duplicate_summary(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Distinguish exact duplicate rows from repeated business keys."""
    records = []
    for dataset, frame in datasets.items():
        key = DATASET_KEYS[dataset]
        records.append(
            {
                "dataset": dataset,
                "key": key,
                "exact_duplicate_rows": int(frame.duplicated().sum()),
                "unique_keys": int(frame[key].nunique()),
                "duplicate_key_excess_rows": int(len(frame) - frame[key].nunique()),
                "rows_with_repeated_key": int(frame[key].duplicated(keep=False).sum()),
            }
        )
    return pd.DataFrame(records)


def quality_findings(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Identify raw-data issues without modifying the source datasets."""
    customers = datasets["customers"]
    orders = datasets["orders"]
    tickets = datasets["tickets"]

    age = pd.to_numeric(customers["age"], errors="coerce")
    amount = pd.to_numeric(orders["order_amount"], errors="coerce")
    delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    city = customers["city"].astype(str)
    gender = customers["gender"].astype(str)

    findings = [
        (
            "customers",
            "city",
            "raw spelling variants",
            customers["city"].nunique(),
            "Whitespace and letter case produce 24 raw values for six cities.",
        ),
        (
            "customers",
            "city",
            "normalized candidate values",
            city.str.strip().str.title().nunique(),
            "A strip-and-title preview yields six cities; no values are changed in EDA.",
        ),
        (
            "customers",
            "gender",
            "nonblank spelling variants",
            gender[gender.str.strip().ne("")].nunique(),
            "Male and female labels appear with inconsistent case and abbreviations.",
        ),
        (
            "customers",
            "gender",
            "blank values",
            gender.str.strip().eq("").sum(),
            "Blank values require an explicit Unknown mapping during cleaning.",
        ),
        (
            "customers",
            "age",
            "blank or nonnumeric values",
            age.isna().sum(),
            "Blank ages are preserved during EDA.",
        ),
        (
            "customers",
            "age",
            "domain outliers",
            ((age < 10) | (age > 100)).sum(),
            "Ages below 10 or above 100 violate the instructor audit rule.",
        ),
        (
            "orders",
            "order_id",
            "exact duplicate rows",
            orders.duplicated().sum(),
            "These rows are identical across every column.",
        ),
        (
            "orders",
            "order_id",
            "duplicate-key excess rows",
            len(orders) - orders["order_id"].nunique(),
            "Repeated IDs include exact duplicates and earlier pending states.",
        ),
        (
            "orders",
            "status",
            "pending rows",
            orders["status"].eq("pending").sum(),
            "Pending records may be earlier states of a final order.",
        ),
        (
            "orders",
            "order_amount",
            "nonnumeric values",
            amount.isna().sum(),
            "The raw amount column contains N/A and error strings.",
        ),
        (
            "orders",
            "delivery_minutes",
            "negative values",
            delivery.lt(0).sum(),
            "Negative delivery durations are impossible.",
        ),
        (
            "orders",
            "delivery_minutes",
            "blank values",
            _blank_mask(orders["delivery_minutes"]).sum(),
            "Cancelled orders and some completed orders have blank durations.",
        ),
        (
            "tickets",
            "order_id",
            "blank values",
            _blank_mask(tickets["order_id"]).sum(),
            "Some support tickets intentionally omit an order identifier.",
        ),
        (
            "tickets",
            "text",
            "blank values",
            _blank_mask(tickets["text"]).sum(),
            "Empty tickets need a guardrail before later model calls.",
        ),
        (
            "tickets",
            "text",
            "over 1000 characters",
            tickets["text"].astype(str).str.len().gt(1000).sum(),
            "Four intentionally overlong tickets require a later guardrail.",
        ),
    ]
    return pd.DataFrame(
        findings, columns=["dataset", "column", "finding", "count", "interpretation"]
    ).assign(count=lambda frame: frame["count"].astype(int))


def category_variants(datasets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Show raw city and gender labels with counts and percentages."""
    customers = datasets["customers"]
    records = []
    for column in ("city", "gender"):
        values = customers[column].where(~_blank_mask(customers[column]), "<BLANK>")
        counts = values.value_counts()
        for value, count in counts.items():
            normalized = (
                str(value).strip().title()
                if column == "city"
                else {
                    "m": "Male",
                    "male": "Male",
                    "f": "Female",
                    "female": "Female",
                }.get(str(value).strip().lower(), "Unknown")
            )
            records.append(
                {
                    "column": column,
                    "raw_value": str(value),
                    "normalized_preview": normalized,
                    "count": int(count),
                    "percentage": round(float(count / len(customers) * 100), 2),
                }
            )
    return pd.DataFrame(records)


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.astype(str)
    headers = [str(column) for column in display.columns]
    rows = [[value.replace("|", "\\|") for value in row] for row in display.values.tolist()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_eda_outputs(
    datasets: Mapping[str, pd.DataFrame], output_dir: Path = OUTPUT_DIR
) -> dict[str, Path]:
    """Write reproducible Stage 3 evidence without changing raw data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "column_profile": column_profile(datasets),
        "value_counts": value_count_summary(datasets),
        "duplicate_summary": duplicate_summary(datasets),
        "quality_findings": quality_findings(datasets),
        "category_variants": category_variants(datasets),
    }
    paths = {}
    for name, frame in tables.items():
        path = output_dir / f"eda_{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path

    report_path = output_dir / "eda_report.md"
    sections = [
        "# Stage 3 Raw-Data EDA Report",
        "",
        "This report audits the three assignment datasets without cleaning or "
        "modifying the generated source files.",
        "",
        "## Dataset shapes",
        "",
        markdown_table(shape_summary(datasets)),
        "",
        "## Duplicate analysis",
        "",
        markdown_table(tables["duplicate_summary"]),
        "",
        "## Data-quality findings",
        "",
        markdown_table(tables["quality_findings"]),
        "",
        "## Scope boundary",
        "",
        "Standardization, imputation, order-state resolution, SQLite loading, and "
        "Pandas/SQL validation belong to the next cleaning stage.",
        "",
    ]
    report_path.write_text("\n".join(sections), encoding="utf-8")
    paths["report"] = report_path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    datasets = load_raw_data(args.raw_dir)
    paths = write_eda_outputs(datasets, args.output_dir)
    print(shape_summary(datasets).to_string(index=False))
    print(f"\nWrote {len(paths)} Stage 3 evidence files to {args.output_dir}")


if __name__ == "__main__":
    main()
