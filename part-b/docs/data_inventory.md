# Generated data inventory

`python generate_data.py` was run from the repository root using the unchanged instructor generator. All records are fictional and deterministic (`random.Random(42)`).

## Files

| Generated file | Rows / lines | Columns | SHA-256 |
| --- | ---: | --- | --- |
| `data/raw/customers.csv` | 1,200 rows | `customer_id`, `city`, `signup_date`, `age`, `gender` | `f7eb19ce7de23837cc6a0bc6f19bca3cae0f5c9f47401ddacf02cd60eb1dd8bf` |
| `data/raw/products.csv` | 40 rows | `product_id`, `product_name`, `category`, `price_inr` | `348177a4ea8a90752b8c0e1738550a6d202c7fe167f4754be129d6878ef0658a` |
| `data/raw/orders.csv` | 13,084 rows | `order_id`, `customer_id`, `product_id`, `quantity`, `order_ts`, `status`, `discount_pct`, `order_amount`, `delivery_minutes` | `ba593093c80e72b7a8ed9d6232c4fd949bd4c0436af8365130f22f3b0c616509` |
| `data/raw/tickets.csv` | 164 rows | `ticket_id`, `customer_id`, `order_id`, `text`, `true_category` | `89e054c5187de098f8587206671b0a4742580620db8dcb897288cf48a0696808` |
| `data/raw/policy_eval.csv` | 14 rows | `question`, `expected_doc`, `expected_fact` | `d238548a527380a7bae422efa997e43b7b9b76fb6aeec7867e9087287589d825` |
| `data/policies/delivery_policy.txt` | 1 line | N/A (text document) | `41903fdaf4d86a362b5be13dd7ed549a586d083bf66ebaeed88a3dba59714644` |
| `data/policies/escalation_policy.txt` | 1 line | N/A (text document) | `0fffd9c8d182ece27da5eb1f4349ed9a8e995119a8dc725236ef43ebd36edd54` |
| `data/policies/payment_policy.txt` | 1 line | N/A (text document) | `3ba59421d806d3d488e13b9421da0bf3b04742814d85e8043cc37847ec07bb4a` |
| `data/policies/privacy_policy.txt` | 1 line | N/A (text document) | `c11e960ef7ebd4c82f5f1e504459f3417a4fef8f23e35b13ff8ccf5f8ee37ac5` |
| `data/policies/refund_policy.txt` | 1 line | N/A (text document) | `458efedd1dc05f60386f4198536330f5f16d89dc445aef0fc67af6a8a20c9c91` |

The generator reports `customers=1200 products=40 orders_raw=13084 tickets=164`.

## Relationships

- `customers.customer_id` is unique and is referenced by every `orders.customer_id` and `tickets.customer_id`; no generated foreign-key values are missing.
- `products.product_id` is unique and is referenced by every `orders.product_id`; no generated foreign-key values are missing.
- `orders.order_id` has 12,309 distinct values across 13,084 raw rows. The 775 additional rows intentionally represent duplicate or earlier-status records for later cleaning exercises.
- `tickets.ticket_id` is unique. All 156 nonblank `tickets.order_id` values resolve to `orders.order_id`; eight tickets intentionally have a blank order identifier.
- `policy_eval.expected_doc` references one of the five generated policy filenames or uses `NONE` for unanswerable questions. Every non-`NONE` reference resolves to a generated document.

## Policy documents

1. `delivery_policy.txt` — promised delivery times, supported cities, fees, weather exceptions, contact attempts, and electronics OTP delivery.
2. `escalation_policy.txt` — fraud priority, response targets, six-step human handoff, and callback requirements.
3. `payment_policy.txt` — accepted methods, cash-on-delivery limit, failed and duplicate payments, wallet use, and coupon restrictions.
4. `privacy_policy.txt` — contact-data use, PII masking, address retention, deletion requests, and payment-card handling.
5. `refund_policy.txt` — eligible statuses, late-delivery threshold, automatic and human approval limits, payment timing, and request window.
