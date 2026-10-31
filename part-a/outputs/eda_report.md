# Stage 3 Raw-Data EDA Report

This report audits the three assignment datasets without cleaning or modifying the generated source files.

## Dataset shapes

| dataset | rows | columns |
| --- | --- | --- |
| customers | 1200 | 5 |
| orders | 13084 | 9 |
| tickets | 164 | 5 |

## Duplicate analysis

| dataset | key | exact_duplicate_rows | unique_keys | duplicate_key_excess_rows | rows_with_repeated_key |
| --- | --- | --- | --- | --- | --- |
| customers | customer_id | 0 | 1200 | 0 | 0 |
| orders | order_id | 119 | 12309 | 775 | 1547 |
| tickets | ticket_id | 0 | 164 | 0 | 0 |

## Data-quality findings

| dataset | column | finding | count | interpretation |
| --- | --- | --- | --- | --- |
| customers | city | raw spelling variants | 24 | Whitespace and letter case produce 24 raw values for six cities. |
| customers | city | normalized candidate values | 6 | A strip-and-title preview yields six cities; no values are changed in EDA. |
| customers | gender | nonblank spelling variants | 10 | Male and female labels appear with inconsistent case and abbreviations. |
| customers | gender | blank values | 30 | Blank values require an explicit Unknown mapping during cleaning. |
| customers | age | blank or nonnumeric values | 61 | Blank ages are preserved during EDA. |
| customers | age | domain outliers | 6 | Ages below 10 or above 100 violate the instructor audit rule. |
| orders | order_id | exact duplicate rows | 119 | These rows are identical across every column. |
| orders | order_id | duplicate-key excess rows | 775 | Repeated IDs include exact duplicates and earlier pending states. |
| orders | status | pending rows | 656 | Pending records may be earlier states of a final order. |
| orders | order_amount | nonnumeric values | 120 | The raw amount column contains N/A and error strings. |
| orders | delivery_minutes | negative values | 106 | Negative delivery durations are impossible. |
| orders | delivery_minutes | blank values | 2541 | Cancelled orders and some completed orders have blank durations. |
| tickets | order_id | blank values | 8 | Some support tickets intentionally omit an order identifier. |
| tickets | text | blank values | 4 | Empty tickets need a guardrail before later model calls. |
| tickets | text | over 1000 characters | 4 | Four intentionally overlong tickets require a later guardrail. |

## Scope boundary

Standardization, imputation, order-state resolution, SQLite loading, and Pandas/SQL validation belong to the next cleaning stage.
