# Ops Knowledge KB (v0.2)

## 1) Metric definitions (口径)

[P1] Scope for lateness metrics: delivered orders only, with both `order_delivered_customer_date` and `order_estimated_delivery_date` present.
[P2] `late_rate` = `n_late` / `n_orders`.
[P3] `avg_days_late` is computed on late orders only (where delivered date > estimated date). It is not a causal attribution.
[P4] Review scope: `n_low_ratings` counts orders with `review_score` <= 2 (missing reviews reduce confidence).
[P5] `low_rating_rate` = `n_low_ratings` / `n_orders` (within the query scope used to build the table).

## 2) Stage clues (诊断线索，不是归因)

[P6] Stage gaps are diagnostic clues, not root cause statements.
[P7] Larger `approved_to_carrier` (approved → carrier) suggests pre-carrier friction.
[P8] Larger `carrier_to_customer` (carrier → customer) suggests in-transit friction.

## 3) Dimensions & slicing (维度/切分)

[D1] Category uses `v_items_en.category_en`.
[D2] Lane uses `seller_state` → `customer_state` (use "NA" when missing).
[D3] Weekly trend uses purchase-time week buckets (`week_start` derived from `order_purchase_timestamp`).

## 4) Execution SOP (下一步怎么做)

[S1] Start with impact: prioritize categories by `n_late` (volume of late orders), then check `late_rate` and `avg_days_late`.
[S2] If one or two lanes dominate by `n_late`, drill down to seller ranking within those lanes.
[S3] If lanes are dispersed, run weekly trend to identify spike weeks vs stable deterioration.
[S4] If results are near the minimum sample threshold, widen time window or adjust `min_orders` before escalation.

## 5) Guardrails (边界)

[G1] Do not claim causality. Use "clue / likely / possible" language only.
[G2] Do not introduce numbers that are not present in the preview rows.
[G3] Any "highest/lowest" claim must match the sorted table; otherwise say "among the top".
[G4] If a field is missing in the table output, do not reference it.

## 6) Output style template (更像 Ops 的表达)

[T1] Always include: (a) Impact leader, (b) Volume vs rate note, (c) Uncertainty note if close to threshold, (d) Next action aligned to SOP.
[T2] When recommending drill-down: state the target dimension explicitly (lane, seller, week) and why it helps isolate the issue.