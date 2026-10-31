-- Row-count checks: these must match the corresponding Pandas DataFrame lengths.
SELECT COUNT(*) AS customers_rows FROM customers;
SELECT COUNT(*) AS products_rows FROM products;
SELECT COUNT(*) AS orders_rows FROM orders;
SELECT COUNT(*) AS tickets_rows FROM tickets;

-- Standardized customer categories.
SELECT city, COUNT(*) AS customers
FROM customers
GROUP BY city
ORDER BY city;

SELECT gender, COUNT(*) AS customers
FROM customers
GROUP BY gender
ORDER BY gender;

-- Final order status distribution and completed revenue.
SELECT status, COUNT(*) AS orders, ROUND(SUM(order_amount), 2) AS revenue_inr
FROM orders
GROUP BY status
ORDER BY status;

-- Referential-integrity checks: every query must return zero.
SELECT COUNT(*) AS orders_without_customer
FROM orders o
LEFT JOIN customers c ON c.customer_id = o.customer_id
WHERE c.customer_id IS NULL;

SELECT COUNT(*) AS orders_without_product
FROM orders o
LEFT JOIN products p ON p.product_id = o.product_id
WHERE p.product_id IS NULL;

SELECT COUNT(*) AS tickets_with_unknown_order
FROM tickets t
LEFT JOIN orders o ON o.order_id = t.order_id
WHERE t.order_id IS NOT NULL AND o.order_id IS NULL;
