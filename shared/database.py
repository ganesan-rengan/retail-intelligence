"""Database connection and session management shared by both services."""

import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()


@lru_cache
def get_engine() -> Engine:
    """Built on first use, not at import time -- so importing this module
    never requires DATABASE_URL unless something actually opens a
    connection. Cached so every caller shares the same engine (and its
    connection pool)."""
    database_url = os.environ["DATABASE_URL"]
    return create_engine(database_url, pool_pre_ping=True)


def get_session() -> Session:
    """Return a new database session. Caller is responsible for closing it."""
    SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)
    return SessionLocal()
