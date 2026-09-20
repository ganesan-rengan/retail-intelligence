# Retail Intelligence

Python 3.12, uv, PostgreSQL 16 + pgvector via docker compose, SQLAlchemy, Alembic.

## Structure
- shared/          — DB models and session, used by both services
- forecasting_service/src/  — Project 1 (standalone scripts, sys.path.insert pattern)
- support_agent/   — Project 2 (in progress: RAG + tool-calling agent, Gates 1-4)
- scripts/         — ETL and one-off utilities

## Working with me
I am learning. Before writing code, explain the approach and wait for confirmation.
After writing, explain what each new file does. Never add a dependency without asking.
Never modify .env. Metrics: WAPE is the headline, MAPE is secondary and unreliable
on sparse rows.