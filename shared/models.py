"""SQLAlchemy models for the shared retail database.

Both the forecasting service and the support agent import from here,
so this module is the single source of truth for the schema.
"""

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    """Base class all models inherit from. Alembic reads its metadata."""


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    country: Mapped[str | None] = mapped_column(String(80))

    orders: Mapped[list["Order"]] = relationship(back_populates="customer")


class Product(Base):
    __tablename__ = "products"

    product_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str | None] = mapped_column(String(80))
    unit_price: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)

    order_items: Mapped[list["OrderItem"]] = relationship(back_populates="product")


class Order(Base):
    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.customer_id"), nullable=False, index=True
    )
    order_date: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)

    customer: Mapped["Customer"] = relationship(back_populates="orders")
    items: Mapped[list["OrderItem"]] = relationship(back_populates="order")
    returns: Mapped[list["Return"]] = relationship(back_populates="order")

    __table_args__ = (
        CheckConstraint(
            "status IN ('delivered', 'shipped', 'pending', 'cancelled')",
            name="ck_orders_status_valid",
        ),
    )


class OrderItem(Base):
    __tablename__ = "order_items"

    order_item_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(
        ForeignKey("orders.order_id"), nullable=False, index=True
    )
    product_id: Mapped[str] = mapped_column(
        ForeignKey("products.product_id"), nullable=False, index=True
    )
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    price: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False)

    order: Mapped["Order"] = relationship(back_populates="items")
    product: Mapped["Product"] = relationship(back_populates="order_items")

    __table_args__ = (
        Index("ix_order_items_product_date", "product_id", "order_id"),
        CheckConstraint("price > 0", name="ck_order_items_price_positive"),
        CheckConstraint("quantity > 0", name="ck_order_items_quantity_positive"),
    )


class Return(Base):
    __tablename__ = "returns"

    return_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(
        ForeignKey("orders.order_id"), nullable=False, index=True
    )
    reason: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="requested")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    idempotency_key: Mapped[str | None] = mapped_column(String(64))

    order: Mapped["Order"] = relationship(back_populates="returns")

    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_returns_idempotency_key"),
        CheckConstraint(
            "status IN ('requested', 'completed', 'approved', 'rejected')",
            name="ck_returns_status_valid",
        ),
    )


class AgentAction(Base):
    __tablename__ = "agent_actions"

    action_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[int | None] = mapped_column(Integer, index=True)
    tool_name: Mapped[str] = mapped_column(String(60), nullable=False)
    arguments: Mapped[str | None] = mapped_column(Text)
    outcome: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), index=True
    )


class PolicyChunk(Base):
    __tablename__ = "policy_chunks"

    chunk_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document: Mapped[str] = mapped_column(String(60), nullable=False)
    section: Mapped[str] = mapped_column(String(120), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Nullable as a safety margin only. In normal operation this is
    # always populated in the same insert as `content` -- the loading
    # script never writes a chunk without its embedding.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384))

    __table_args__ = (
        CheckConstraint(
            "document IN ('returns_policy', 'shipping_policy', 'refund_policy')",
            name="ck_policy_chunks_document_valid",
        ),
    )