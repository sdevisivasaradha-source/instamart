# Instamart Customer Copilot - Part A

## Overview

Part A establishes the analytical data foundation for the Instamart customer-support capstone. It profiles the synthetic source data, cleans quality problems deterministically, loads validated tables into SQLite, and proves key metrics in both Pandas and SQL.

The instructor teaching kit is preserved unchanged under `instructor_reference/`. The root `generate_data.py` is a byte-identical copy of the instructor generator.

## Methodology

1. Generate reproducible customers, products, orders, tickets, and policy evaluation data.
2. Profile shapes, types, missing values, categories, duplicates, and outliers.
3. Standardize city and gender values and impute invalid or missing ages.
4. resolve duplicate order states, repair amounts, and handle delivery-time anomalies.
5. Write clean CSV files, load constrained SQLite tables, and compare Pandas and SQL results.

## Data and results

- Raw customers: 1,200 rows.
- Raw orders: 13,084 rows and 12,309 distinct order IDs.
- Clean orders: 12,309 rows.
- Completed orders: 11,110.
- Cancelled orders: 1,199.
- Completed revenue: INR 2,998,668.80.
- All nine Pandas/SQL cross-check metrics agree.

Executed evidence is available in `notebooks/` and `outputs/`.

## Run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python generate_data.py
python -m src.eda
python -m src.data_pipeline
pytest -q
```

## Conclusion

Part A produces a reproducible, quality-checked dataset and SQLite foundation for the later customer intelligence and support-copilot parts.
