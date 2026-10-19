# Northstar Devices support handbook

Synthetic workshop document. These are fictional company policies, not AWS policies.
Section IDs identify evidence that can be cited in an answer.

## P01 — Intake and ownership

Open one support case for each affected device. Record the customer's description
before deciding whether the issue concerns a product return, technical support or
a possible warranty claim. Several departments may contribute to the same case,
but the person preparing a handoff remains responsible for distinguishing observed
facts from requests and proposed actions. Keep unrelated orders in their own
cases. An available shipping address, receipt or account preference does not by
itself authorize a transaction. If a customer corrects an identifier, use the
correction and preserve enough history to explain the change to the next agent.

## P02 — Standard product returns

The standard product-return window is 30 days from purchase. A receipt and original
packaging are required. An item purchased more than 30 days ago is outside this
standard window even when the customer still has both. A support assistant may
explain the requirements and refer an eligible request for review, but cannot
approve a return or promise payment. A technical fault does not automatically
extend the standard window. For a reported fault outside the window, move to
warranty review under P03. Do not describe warranty review as an already-approved
return, repair or replacement.

## P03 — Warranty evidence and approval

For a reported device fault outside the standard product-return window, the next
route is `warranty_review`. Before a coverage decision, collect the device serial
number, the applicable warranty terms and the approved hardware diagnostic result.
A receipt confirms purchase but does not establish warranty coverage. A cable swap
or test on another device is useful troubleshooting evidence; neither is the
approved hardware diagnostic. Missing warranty evidence must remain marked as
missing. Only the authorized warranty team can approve a remedy. Until that team
records approval, the assistant must keep `approved=false` and avoid promising a
repair, replacement or refund.

## P04 — Troubleshooting boundaries

Ask which checks have already been completed and what each one showed. Distinguish
a symptom reproduced on another device from a hardware diagnostic designed for the
specific model. An employee using managed equipment may be unable to install
software or reset a device. Respect those restrictions when proposing the next
step. If the next documented check requires an action the customer has prohibited,
refer it to the equipment owner or support specialist. Do not reinterpret a
meeting time, a willingness to continue chatting or access to another computer as
permission to change device settings.

## P05 — Receipts and attachments

Accept a readable receipt or order confirmation through the case's attachment
control. Check that the device and order identifiers belong to the current issue.
A photograph can establish which label the customer read, but an unreadable
photograph should not be treated as a verified serial number. Keep attachment
descriptions separate from what their contents establish. A record saying that a
customer will upload a file is a future action, not proof that the file arrived.
Do not request payment-card details as a substitute for an order reference. Use the
authenticated case's document controls for anything containing customer data.

## P06 — Standard shipping

After an authorized transaction, standard delivery normally takes three to five
business days from dispatch. Dispatch and delivery are different events. A label
being prepared does not mean a carrier has collected a parcel. Warehouse cutoffs
and public holidays can change the dispatch date; support should quote the
available estimate rather than turn it into a guarantee. Shipping preferences
may be stored before any remedy is approved. Do not use those preferences as
evidence of approval. An unrelated order arriving at the same address must not be
merged into the affected device's technical history.

## P07 — Packaging and collection

When a collection has been authorized, send the instructions for that product
category. Customers should remove accessories that are not requested and keep a
record of what they place in the parcel. Packaging recommendations for safe
transport are separate from the original-packaging requirement for a standard
product return. If the company provides replacement transport packaging, that
does not retrospectively change the customer's return eligibility. Do not book a
collection merely because a customer has a box available. Collection details
belong after a documented decision by the team responsible for the remedy.

## P08 — Business-account contacts

A business account may have separate purchasing, equipment-management and billing
contacts. Confirm which contact can supply missing information before asking a
user to obtain it. The person experiencing a fault may be able to describe a
symptom but unable to authorize a reset, install software or approve a purchase.
The company's contact hierarchy does not change the standard product-return
window in P02. Record who owns the next action and which evidence is still needed.
Avoid sending the customer through several teams when one specific request for
information would move the case forward.

## P09 — Communication and accessibility

Use the communication channel requested for the current case when it is available.
Written instructions can help a customer follow a check between meetings or share
it with an equipment owner. Do not infer technical restrictions solely from a
preferred communication channel. Keep a case summary brief enough for another
agent to read, while retaining corrections, explicit restrictions, completed
checks and unresolved work. A polite acknowledgement should not be phrased as a
decision on eligibility. State when the next step depends on information that has
not yet been supplied.

## P10 — Escalation handoffs

An escalation package should identify the current device, describe the symptom,
list the checks already completed and distinguish those observations from future
steps. Include any restriction that could make a standard instruction unsuitable.
Use source references so the receiving team can inspect the original policy or
case note when something is unclear. Old identifiers may remain in historical
notes; the summary must make the current identifier unambiguous. Avoid copying an
entire account history into every handoff when only a few entries affect the next
decision. Never omit an unresolved requirement merely to shorten the summary.

## P11 — Case closure

A case can be closed after the customer confirms resolution or the responsible
team completes its documented process. A pause in correspondence is not proof that
a technical problem has been fixed. Record whether a diagnostic was completed,
whether the customer accepted a remedy and whether any action remains open.
Closure notes should refer to the decision that authorized any transaction.
Do not convert an unanswered question into an affirmative customer response.
If the customer returns with the same unresolved symptom, consult the existing
evidence before asking them to repeat checks already recorded as unsuccessful.

## P12 — Reporting and quality review

Quality reviews examine whether the advice followed policy, relied on available
evidence and gave a useful next step. Short answers are valuable when they retain
the conditions that matter. A detailed answer can still be wrong if it merges
unrelated orders, uses a superseded identifier or describes a proposed check as
completed. Operational reports count handling time and completed actions
separately. They should not treat a newly assigned case as a resolved case.
When reviewing automation, include the work performed before the final response,
such as document reading and specialist analysis.
