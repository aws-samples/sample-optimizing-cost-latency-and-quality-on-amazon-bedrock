---
name: returns-review
description: Decide the next review step for a customer return request using the case record.
---

# Returns review

1. Call `get_return_case` with the supplied case reference before deciding. Use the
   returned case facts; a customer request is not proof of eligibility.
2. Within 30 days with a receipt: use `decision=return_review` and
   `next_step=contact_support`.
3. Within 30 days with a receipt absent or unknown: use
   `decision=missing_evidence` and `next_step=request_receipt`.
4. Beyond 30 days: use `decision=warranty_review` and
   `next_step=request_warranty_evidence`. Do not assume warranty coverage.
5. Copy `evidence_id` from the case record and keep `approved=false`.

Return only the JSON fields required by the shared instructions. Do not approve
a return or claim that money has been refunded.
