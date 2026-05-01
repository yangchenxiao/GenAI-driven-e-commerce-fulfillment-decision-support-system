# Fulfillment Triage Policy (Stage 3 RAG)

## [P1] Metrics definitions (must come from SQL)
- late_rate: share of delivered orders where delivered_customer_date > estimated_delivery_date.
- avg_days_late: average days late among late delivered orders only (null for on-time).
- n_orders / n_delivered: sample size used for each aggregation.
- stage gaps (hours):
  - purchase_to_approved = approved_at - purchase_timestamp
  - approved_to_carrier = delivered_carrier_date - approved_at
  - carrier_to_customer = delivered_customer_date - delivered_carrier_date

## [P2] Interpretation guardrails (no over-claiming)
- Stage gaps provide *clues* (pre-carrier vs in-transit) but do not prove root cause.
- Do not claim causality unless the dataset contains explicit causal fields (carrier SLA, route, warehouse scan logs, etc.).

## [P3] Small-sample & uncertainty rules
- If n_orders is small, results are unstable. Use a minimum threshold (e.g., min_orders) and clearly state uncertainty when near the threshold.
- Always prefer high-volume categories first when prioritizing.

## [P4] Review-risk rules (service quality vs delivery)
- low_rating_rate: share of reviews with score <= 2 among orders with a review_score present.
- Review signals may correlate with delays, but correlation is not causation. Use side-by-side metrics and propose further drill-down.

## [P5] Recommended drill-down queries
- Drill down by seller_state -> customer_state lanes to localize bottlenecks.
- Drill down to top sellers within a category to find concentration risk.
- Compare late vs on-time cohorts if available, and check missing timestamps (e.g., carrier date missing).

## [P6] Output requirements (what the assistant must do)
- Never invent numbers not shown in the SQL table.
- Provide 2–4 bullet insights + 1 next executable query suggestion.
- Include citations [P#] for any metric definitions / rules referenced.
