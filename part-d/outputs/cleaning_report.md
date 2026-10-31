# Data Cleaning and SQLite Validation Report

## Cleaning evidence

| dataset | metric | before | after | cleaning_rule |
| --- | --- | --- | --- | --- |
| customers | rows | 1200 | 1200 | preserved |
| customers | city variants | 24 | 6 | strip whitespace and title-case |
| customers | gender values including blanks | 11 | 3 | map to Male, Female, or Unknown |
| customers | invalid or blank ages | 67 | 0 | replace with city median |
| orders | rows | 13084 | 12309 | keep latest final state |
| orders | exact duplicate rows | 119 | 0 | remove exact duplicates |
| orders | repeated order ID excess | 775 | 0 | keep latest timestamp per order |
| orders | pending rows | 656 | 0 | resolve to later final state |
| orders | nonnumeric amounts | 120 | 0 | recalculate from product price, quantity, and discount |
| orders | negative delivery durations | 106 | 0 | replace and impute completed orders by category median |
| tickets | blank ticket text | 4 | 4 | retain for later guardrail handling |

## Pandas and SQL comparison

| metric | pandas_value | sql_value | matches |
| --- | --- | --- | --- |
| customers_rows | 1200 | 1200 | True |
| products_rows | 40 | 40 | True |
| orders_rows | 12309 | 12309 | True |
| tickets_rows | 164 | 164 | True |
| distinct_cities | 6 | 6 | True |
| completed_orders | 11110 | 11110 | True |
| cancelled_orders | 1199 | 1199 | True |
| completed_revenue_inr | 2998668.80 | 2998668.80 | True |
| tickets_without_order_id | 8 | 8 | True |

All comparison rows must show `True` before the pipeline succeeds.
