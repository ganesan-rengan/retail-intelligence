# Returns Policy

## 1. Return Eligibility
Customers may request a return only for orders with the status `delivered`.
A return request must be initiated within 30 days of the order date (`orders.order_date`), regardless of how long delivery took.
Orders with the status `pending` or `shipped` are not eligible for a return because they have not yet been delivered.
Orders with the status `cancelled` are not eligible for a return.

## 2. Return Conditions
Items must be returned in their original condition, including original packaging and accessories where applicable.
Items that have been damaged, used improperly, or altered by the customer may not be eligible for a return.
The return request must correspond to the order being returned. A customer may not initiate a return for an order belonging to another customer.

## 3. Initiating a Return
Customers who meet the eligibility requirements may request a return through the return process.
The system will verify that the order belongs to the requesting customer and that the order has a `delivered` status.
The system will also verify that the return request is within 30 days of the order date (`orders.order_date`).
If these requirements are not satisfied, the return request will not be initiated.

## 4. Return Review
A return request may be reviewed before final approval.
Approval of a return does not automatically guarantee that the returned item will be accepted if the item does not satisfy the applicable return conditions.

## 5. Exceptions
Exceptions to this policy may be considered only when specifically authorized under applicable customer-service procedures.
The standard return eligibility requirements remain the default rules.