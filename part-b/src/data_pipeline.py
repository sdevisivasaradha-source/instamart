"""Clean the synthetic datasets, persist SQLite tables, and validate results."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Mapping

import pandas as pd

from src.eda import RAW_DIR, _blank_mask, load_raw_data, markdown_table

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
DB_PATH = ROOT / "data" / "instamart.db"
OUTPUT_DIR = ROOT / "outputs"
SUPPORTED_CITIES = {"Bengaluru", "Chennai", "Delhi", "Hyderabad", "Mumbai", "Pune"}
GENDER_MAP = {
    "m": "Male",
    "male": "Male",
    "f": "Female",
    "female": "Female",
}


def load_source_data(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Load the three assignment datasets and the product lookup table."""
    datasets = load_raw_data(raw_dir)
    product_path = raw_dir / "products.csv"
    if not product_path.is_file():
        raise FileNotFoundError(
            f"Missing product lookup: {product_path}. Run `python generate_data.py` first."
        )
    products = pd.read_csv(product_path, keep_default_na=False)
    return {
        "customers": datasets["customers"],
        "products": products,
        "orders": datasets["orders"],
        "tickets": datasets["tickets"],
    }


def clean_customers(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardize customer labels and impute invalid ages by city median."""
    customers = raw.drop_duplicates().copy()
    customers["customer_id"] = customers["customer_id"].astype(str).str.strip()
    if customers["customer_id"].duplicated().any():
        raise ValueError("Customer IDs remain duplicated after exact-row deduplication")

    customers["city"] = customers["city"].astype(str).str.strip().str.title()
    unexpected_cities = set(customers["city"]) - SUPPORTED_CITIES
    if unexpected_cities:
        raise ValueError(f"Unsupported customer cities: {sorted(unexpected_cities)}")

    normalized_gender = customers["gender"].astype(str).str.strip().str.lower()
    customers["gender"] = normalized_gender.map(GENDER_MAP).fillna("Unknown")

    age = pd.to_numeric(customers["age"], errors="coerce")
    age = age.mask(~age.between(10, 100))
    city_median = age.groupby(customers["city"]).transform("median")
    customers["age"] = age.fillna(city_median).round().astype("int64")
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], format="%Y-%m-%d", errors="raise"
    ).dt.strftime("%Y-%m-%d")
    return customers.reset_index(drop=True)


def clean_products(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and type the product lookup used by order cleaning."""
    products = raw.drop_duplicates().copy()
    products["product_id"] = products["product_id"].astype(str).str.strip()
    products["product_name"] = products["product_name"].astype(str).str.strip()
    products["category"] = products["category"].astype(str).str.strip()
    products["price_inr"] = pd.to_numeric(products["price_inr"], errors="raise")
    if products["product_id"].duplicated().any():
        raise ValueError("Product IDs remain duplicated after exact-row deduplication")
    if products["price_inr"].le(0).any():
        raise ValueError("Product prices must be positive")
    return products.reset_index(drop=True)


def clean_orders(raw: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Resolve order histories and repair numeric fields using explicit rules."""
    orders = raw.drop_duplicates().copy()
    for column in ("order_id", "customer_id", "product_id"):
        orders[column] = orders[column].astype(str).str.strip()
    orders["status"] = orders["status"].astype(str).str.strip().str.lower()
    orders["order_ts"] = pd.to_datetime(
        orders["order_ts"], format="%Y-%m-%d %H:%M:%S", errors="raise"
    )

    status_rank = orders["status"].map({"pending": 0, "cancelled": 1, "completed": 1})
    if status_rank.isna().any():
        invalid = sorted(orders.loc[status_rank.isna(), "status"].unique())
        raise ValueError(f"Unsupported order statuses: {invalid}")
    orders = (
        orders.assign(_status_rank=status_rank)
        .sort_values(["order_id", "order_ts", "_status_rank"], kind="stable")
        .drop_duplicates("order_id", keep="last")
        .drop(columns="_status_rank")
        .reset_index(drop=True)
    )
    if orders["status"].eq("pending").any():
        pending = int(orders["status"].eq("pending").sum())
        raise ValueError(f"{pending} pending orders have no later final state")

    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="raise").astype("int64")
    orders["discount_pct"] = pd.to_numeric(
        orders["discount_pct"], errors="raise"
    ).astype("int64")
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    orders["delivery_minutes"] = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )
    orders.loc[orders["delivery_minutes"].lt(0), "delivery_minutes"] = pd.NA

    product_lookup = products.set_index("product_id")
    missing_products = set(orders["product_id"]) - set(product_lookup.index)
    if missing_products:
        raise ValueError(f"Orders reference unknown products: {sorted(missing_products)}")

    missing_amount = orders["order_amount"].isna()
    calculated_amount = (
        orders["product_id"].map(product_lookup["price_inr"])
        * orders["quantity"]
        * (1 - orders["discount_pct"] / 100)
    ).round(2)
    orders.loc[missing_amount, "order_amount"] = calculated_amount[missing_amount]
    orders["order_amount"] = orders["order_amount"].round(2)

    category = orders["product_id"].map(product_lookup["category"])
    completed = orders["status"].eq("completed")
    category_median = orders.loc[completed].groupby(category[completed])[
        "delivery_minutes"
    ].median()
    missing_completed_delivery = completed & orders["delivery_minutes"].isna()
    orders.loc[missing_completed_delivery, "delivery_minutes"] = category[
        missing_completed_delivery
    ].map(category_median)
    orders.loc[orders["status"].eq("cancelled"), "delivery_minutes"] = pd.NA
    orders["delivery_minutes"] = orders["delivery_minutes"].round().astype("Int64")
    orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return orders


def clean_tickets(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize ticket identifiers while retaining guardrail test cases."""
    tickets = raw.drop_duplicates().copy()
    for column in ("ticket_id", "customer_id", "order_id"):
        tickets[column] = tickets[column].astype(str).str.strip()
    tickets["text"] = tickets["text"].astype(str).str.strip()
    tickets["true_category"] = (
        tickets["true_category"].astype(str).str.strip().str.lower()
    )
    if tickets["ticket_id"].duplicated().any():
        raise ValueError("Ticket IDs remain duplicated after exact-row deduplication")
    return tickets.reset_index(drop=True)


def clean_data(raw: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Apply all cleaning rules without mutating caller-owned frames."""
    products = clean_products(raw["products"])
    return {
        "customers": clean_customers(raw["customers"]),
        "products": products,
        "orders": clean_orders(raw["orders"], products),
        "tickets": clean_tickets(raw["tickets"]),
    }


def validate_clean_data(clean: Mapping[str, pd.DataFrame]) -> dict[str, int]:
    """Assert cleaned-frame invariants and return concise checkpoints."""
    customers = clean["customers"]
    products = clean["products"]
    orders = clean["orders"]
    tickets = clean["tickets"]

    if not customers["customer_id"].is_unique:
        raise AssertionError("customer_id must be unique")
    if set(customers["city"]) != SUPPORTED_CITIES:
        raise AssertionError("customers must contain exactly the six supported cities")
    if not set(customers["gender"]) <= {"Male", "Female", "Unknown"}:
        raise AssertionError("gender contains nonstandard values")
    if customers["age"].isna().any() or not customers["age"].between(18, 60).all():
        raise AssertionError("customer ages must be populated and between 18 and 60")
    if not products["product_id"].is_unique:
        raise AssertionError("product_id must be unique")
    if not orders["order_id"].is_unique:
        raise AssertionError("order_id must be unique")
    if not set(orders["status"]) <= {"completed", "cancelled"}:
        raise AssertionError("orders must contain only final statuses")
    if orders["order_amount"].isna().any() or orders["order_amount"].lt(0).any():
        raise AssertionError("order amounts must be valid nonnegative numbers")
    completed_delivery = orders.loc[
        orders["status"].eq("completed"), "delivery_minutes"
    ]
    cancelled_delivery = orders.loc[
        orders["status"].eq("cancelled"), "delivery_minutes"
    ]
    if completed_delivery.isna().any() or completed_delivery.lt(0).any():
        raise AssertionError("completed orders need nonnegative delivery durations")
    if cancelled_delivery.notna().any():
        raise AssertionError("cancelled orders must not have delivery durations")
    if not tickets["ticket_id"].is_unique:
        raise AssertionError("ticket_id must be unique")

    customer_ids = set(customers["customer_id"])
    product_ids = set(products["product_id"])
    order_ids = set(orders["order_id"])
    if not set(orders["customer_id"]) <= customer_ids:
        raise AssertionError("orders contain unknown customer IDs")
    if not set(orders["product_id"]) <= product_ids:
        raise AssertionError("orders contain unknown product IDs")
    if not set(tickets["customer_id"]) <= customer_ids:
        raise AssertionError("tickets contain unknown customer IDs")
    ticket_order_ids = set(tickets.loc[tickets["order_id"].ne(""), "order_id"])
    if not ticket_order_ids <= order_ids:
        raise AssertionError("tickets contain unknown order IDs")

    return {
        "customers": len(customers),
        "products": len(products),
        "orders": len(orders),
        "tickets": len(tickets),
    }


def write_processed_csvs(
    clean: Mapping[str, pd.DataFrame], processed_dir: Path = PROCESSED_DIR
) -> dict[str, Path]:
    """Write clean tables as inspectable CSV files."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, frame in clean.items():
        path = processed_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths


def _sqlite_value(value: object) -> object:
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _records(frame: pd.DataFrame) -> list[tuple[object, ...]]:
    return [
        tuple(_sqlite_value(value) for value in row)
        for row in frame.itertuples(index=False, name=None)
    ]


def write_sqlite(
    clean: Mapping[str, pd.DataFrame], db_path: Path = DB_PATH
) -> Path:
    """Replace the local database with constrained, foreign-keyed clean tables."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    schema = """
    DROP TABLE IF EXISTS tickets;
    DROP TABLE IF EXISTS orders;
    DROP TABLE IF EXISTS products;
    DROP TABLE IF EXISTS customers;

    CREATE TABLE customers (
        customer_id TEXT PRIMARY KEY,
        city TEXT NOT NULL,
        signup_date TEXT NOT NULL,
        age INTEGER NOT NULL CHECK (age BETWEEN 18 AND 100),
        gender TEXT NOT NULL CHECK (gender IN ('Male', 'Female', 'Unknown'))
    );
    CREATE TABLE products (
        product_id TEXT PRIMARY KEY,
        product_name TEXT NOT NULL,
        category TEXT NOT NULL,
        price_inr REAL NOT NULL CHECK (price_inr > 0)
    );
    CREATE TABLE orders (
        order_id TEXT PRIMARY KEY,
        customer_id TEXT NOT NULL REFERENCES customers(customer_id),
        product_id TEXT NOT NULL REFERENCES products(product_id),
        quantity INTEGER NOT NULL CHECK (quantity > 0),
        order_ts TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('completed', 'cancelled')),
        discount_pct INTEGER NOT NULL CHECK (discount_pct BETWEEN 0 AND 100),
        order_amount REAL NOT NULL CHECK (order_amount >= 0),
        delivery_minutes INTEGER CHECK (delivery_minutes >= 0)
    );
    CREATE TABLE tickets (
        ticket_id TEXT PRIMARY KEY,
        customer_id TEXT NOT NULL REFERENCES customers(customer_id),
        order_id TEXT REFERENCES orders(order_id),
        text TEXT NOT NULL,
        true_category TEXT NOT NULL
    );
    """
    inserts = {
        "customers": "INSERT INTO customers VALUES (?, ?, ?, ?, ?)",
        "products": "INSERT INTO products VALUES (?, ?, ?, ?)",
        "orders": "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        "tickets": "INSERT INTO tickets VALUES (?, ?, ?, ?, ?)",
    }

    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(schema)
        for table in ("customers", "products", "orders"):
            connection.executemany(inserts[table], _records(clean[table]))
        ticket_rows = clean["tickets"].copy()
        ticket_rows["order_id"] = ticket_rows["order_id"].replace("", pd.NA)
        connection.executemany(inserts["tickets"], _records(ticket_rows))
        foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_errors:
            raise AssertionError(f"SQLite foreign-key violations: {foreign_key_errors}")
    return db_path


def pandas_sql_comparison(
    clean: Mapping[str, pd.DataFrame], db_path: Path = DB_PATH
) -> pd.DataFrame:
    """Compare core Pandas metrics with independently queried SQLite values."""
    customers = clean["customers"]
    products = clean["products"]
    orders = clean["orders"]
    tickets = clean["tickets"]
    pandas_metrics = {
        "customers_rows": len(customers),
        "products_rows": len(products),
        "orders_rows": len(orders),
        "tickets_rows": len(tickets),
        "distinct_cities": customers["city"].nunique(),
        "completed_orders": int(orders["status"].eq("completed").sum()),
        "cancelled_orders": int(orders["status"].eq("cancelled").sum()),
        "completed_revenue_inr": round(
            float(orders.loc[orders["status"].eq("completed"), "order_amount"].sum()), 2
        ),
        "tickets_without_order_id": int(tickets["order_id"].eq("").sum()),
    }
    sql_queries = {
        "customers_rows": "SELECT COUNT(*) FROM customers",
        "products_rows": "SELECT COUNT(*) FROM products",
        "orders_rows": "SELECT COUNT(*) FROM orders",
        "tickets_rows": "SELECT COUNT(*) FROM tickets",
        "distinct_cities": "SELECT COUNT(DISTINCT city) FROM customers",
        "completed_orders": "SELECT COUNT(*) FROM orders WHERE status='completed'",
        "cancelled_orders": "SELECT COUNT(*) FROM orders WHERE status='cancelled'",
        "completed_revenue_inr": (
            "SELECT ROUND(SUM(order_amount), 2) FROM orders WHERE status='completed'"
        ),
        "tickets_without_order_id": (
            "SELECT COUNT(*) FROM tickets WHERE order_id IS NULL"
        ),
    }
    with sqlite3.connect(db_path) as connection:
        sql_metrics = {
            metric: connection.execute(query).fetchone()[0]
            for metric, query in sql_queries.items()
        }

    rows = []
    for metric, pandas_value in pandas_metrics.items():
        sql_value = sql_metrics[metric]
        if isinstance(pandas_value, float):
            matches = abs(pandas_value - float(sql_value)) < 0.005
        else:
            matches = pandas_value == sql_value
        rows.append(
            {
                "metric": metric,
                "pandas_value": (
                    f"{pandas_value:.2f}"
                    if metric == "completed_revenue_inr"
                    else str(pandas_value)
                ),
                "sql_value": (
                    f"{float(sql_value):.2f}"
                    if metric == "completed_revenue_inr"
                    else str(sql_value)
                ),
                "matches": matches,
            }
        )
    return pd.DataFrame(rows)


def cleaning_metrics(
    raw: Mapping[str, pd.DataFrame], clean: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """Summarize each cleaning decision with before/after evidence."""
    raw_customers = raw["customers"]
    raw_orders = raw["orders"]
    clean_customers_frame = clean["customers"]
    clean_orders_frame = clean["orders"]
    raw_age = pd.to_numeric(raw_customers["age"], errors="coerce")
    raw_amount = pd.to_numeric(raw_orders["order_amount"], errors="coerce")
    raw_delivery = pd.to_numeric(raw_orders["delivery_minutes"], errors="coerce")

    rows = [
        ("customers", "rows", len(raw_customers), len(clean_customers_frame), "preserved"),
        (
            "customers",
            "city variants",
            raw_customers["city"].nunique(),
            clean_customers_frame["city"].nunique(),
            "strip whitespace and title-case",
        ),
        (
            "customers",
            "gender values including blanks",
            raw_customers["gender"].nunique(),
            clean_customers_frame["gender"].nunique(),
            "map to Male, Female, or Unknown",
        ),
        (
            "customers",
            "invalid or blank ages",
            int((raw_age.isna() | (raw_age < 10) | (raw_age > 100)).sum()),
            int(clean_customers_frame["age"].isna().sum()),
            "replace with city median",
        ),
        ("orders", "rows", len(raw_orders), len(clean_orders_frame), "keep latest final state"),
        (
            "orders",
            "exact duplicate rows",
            int(raw_orders.duplicated().sum()),
            int(clean_orders_frame.duplicated().sum()),
            "remove exact duplicates",
        ),
        (
            "orders",
            "repeated order ID excess",
            len(raw_orders) - raw_orders["order_id"].nunique(),
            len(clean_orders_frame) - clean_orders_frame["order_id"].nunique(),
            "keep latest timestamp per order",
        ),
        (
            "orders",
            "pending rows",
            int(raw_orders["status"].eq("pending").sum()),
            int(clean_orders_frame["status"].eq("pending").sum()),
            "resolve to later final state",
        ),
        (
            "orders",
            "nonnumeric amounts",
            int(raw_amount.isna().sum()),
            int(clean_orders_frame["order_amount"].isna().sum()),
            "recalculate from product price, quantity, and discount",
        ),
        (
            "orders",
            "negative delivery durations",
            int(raw_delivery.lt(0).sum()),
            int(clean_orders_frame["delivery_minutes"].lt(0).sum()),
            "replace and impute completed orders by category median",
        ),
        (
            "tickets",
            "blank ticket text",
            int(_blank_mask(raw["tickets"]["text"]).sum()),
            int(_blank_mask(clean["tickets"]["text"]).sum()),
            "retain for later guardrail handling",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["dataset", "metric", "before", "after", "cleaning_rule"]
    )


def write_validation_outputs(
    raw: Mapping[str, pd.DataFrame],
    clean: Mapping[str, pd.DataFrame],
    comparison: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Path]:
    """Write cleaning decisions and Pandas/SQL validation evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = cleaning_metrics(raw, clean)
    metrics_path = output_dir / "cleaning_metrics.csv"
    comparison_path = output_dir / "pandas_sql_validation.csv"
    summary_path = output_dir / "cleaning_validation.json"
    report_path = output_dir / "cleaning_report.md"
    metrics.to_csv(metrics_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    summary = {
        "all_pandas_sql_metrics_match": bool(comparison["matches"].all()),
        "clean_rows": {name: len(frame) for name, frame in clean.items()},
        "customer_cities": sorted(clean["customers"]["city"].unique()),
        "customer_genders": sorted(clean["customers"]["gender"].unique()),
        "database_tables": sorted(clean),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = "\n".join(
        [
            "# Data Cleaning and SQLite Validation Report",
            "",
            "## Cleaning evidence",
            "",
            markdown_table(metrics),
            "",
            "## Pandas and SQL comparison",
            "",
            markdown_table(comparison),
            "",
            "All comparison rows must show `True` before the pipeline succeeds.",
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")
    return {
        "metrics": metrics_path,
        "comparison": comparison_path,
        "summary": summary_path,
        "report": report_path,
    }


def run_pipeline(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    db_path: Path = DB_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, object]:
    """Run cleaning, persistence, and cross-system validation end to end."""
    raw = load_source_data(raw_dir)
    clean = clean_data(raw)
    checkpoints = validate_clean_data(clean)
    csv_paths = write_processed_csvs(clean, processed_dir)
    write_sqlite(clean, db_path)
    comparison = pandas_sql_comparison(clean, db_path)
    if not comparison["matches"].all():
        failures = comparison.loc[~comparison["matches"]].to_dict("records")
        raise AssertionError(f"Pandas/SQL validation failed: {failures}")
    output_paths = write_validation_outputs(raw, clean, comparison, output_dir)
    return {
        "raw": raw,
        "clean": clean,
        "checkpoints": checkpoints,
        "comparison": comparison,
        "csv_paths": csv_paths,
        "output_paths": output_paths,
        "db_path": db_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    result = run_pipeline(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        db_path=args.db_path,
        output_dir=args.output_dir,
    )
    print("Clean rows:", result["checkpoints"])
    print(result["comparison"].to_string(index=False))
    print(f"\nSQLite database: {args.db_path}")


if __name__ == "__main__":
    main()
