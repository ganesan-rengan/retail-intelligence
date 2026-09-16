# Shipping Policy

## 1. Order Status
Orders use four supported statuses:

* `pending` — The order has been placed but has not yet been shipped.
* `shipped` — The order has been shipped and is in transit.
* `delivered` — The order has been delivered.
* `cancelled` — The order has been cancelled and will not proceed to delivery.

## 2. Pending Orders
An order with status `pending` has not yet been shipped.
Customers may contact customer support regarding a pending order if they need information about its progress.
A pending order may be cancelled when it satisfies the applicable cancellation rules.

## 3. Shipped Orders
An order with status `shipped` has already been shipped.
Once an order is shipped, cancellation may no longer be available through the standard cancellation process.
Customers should use the order status information provided by the system to determine whether an order has been shipped.

## 4. Delivered Orders
An order with status `delivered` has been delivered to the customer.
Once an order has been delivered, the customer should follow the Returns Policy if they want to return an eligible item.

## 5. Cancelled Orders
An order with status `cancelled` will not be delivered.
A cancelled order cannot be treated as a shipped or delivered order.
Customers with questions about cancelled orders should refer to the applicable Refund Policy.

## 6. Order Status Information
The system's `get_order_status` function is the source of truth for the current order status.
Customer-service responses about an order's current status must be based on the status returned by the system rather than an assumed or inferred status.