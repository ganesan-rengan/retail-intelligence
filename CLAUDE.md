# Retail Intelligence Platform

## What this is
Two services over one shared PostgreSQL database:
- `forecasting-service/` — weekly demand forecasting API (FastAPI + LightGBM)
- `support-agent/` — LLM agent with tool calling, including a guarded write operation
- `shared/` — SQLAlchemy models and database session, imported by both

## Stack
Python 3.12 · uv · FastAPI · SQLAlchemy 2.0 · Alembic · PostgreSQL 16 + pgvector · Docker

## How I work
I am learning this stack while building. Before writing code:
1. Explain your approach and wait for my confirmation.
2. After writing, explain what each new function does and why.
3. Never add a dependency without asking first.
4. Prefer simple, readable code over clever code.

## Conventions
- Run commands with `uv run`, never bare `python`
- Config comes from environment variables via `python-dotenv`. Never hardcode secrets
- `.env` is gitignored. `.env.example` documents variable names only
- Schema changes go through Alembic migrations, never `create_all()`
- Type hints on function signatures
- Conventional Commits: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`

## Data decisions already made (do not change without asking)
- Orphan rows (243,007 with null Customer ID) map to customer_id=0 "Guest Checkout"
- `C`-prefixed invoices are cancellations → `returns` table, excluded from `order_items`
- Order status is randomised on seed 42: 70% delivered, 20% shipped, 8% pending
- `product_id` is String(20) — codes like POST, M, BANK CHARGES exist
- Money columns are Numeric(10,2), never Float