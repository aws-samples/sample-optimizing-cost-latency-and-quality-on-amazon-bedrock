---
name: billing-review
description: Decide the next review step for a reported duplicate charge using the case record.
---

# Billing review

1. Call `get_billing_case` with the supplied case reference before deciding.
2. An alleged duplicate charge requires verification. When the record includes
   an order reference, use `decision=billing_review` and
   `next_step=verify_duplicate`.
3. Without an order reference, use `decision=missing_evidence` and
   `next_step=request_order_reference`.
4. Copy `evidence_id` from the case record and keep `approved=false`.

Return only the JSON fields required by the shared instructions. Never request
card numbers or claim that a refund has been issued.
