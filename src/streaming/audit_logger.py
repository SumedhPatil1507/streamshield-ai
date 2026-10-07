"""Audit logging module for the StreamShield AI streaming pipeline.

Persists ``InferenceResult`` records to a PostgreSQL ``inference_audit`` table
using SQLAlchemy's async engine backed by asyncpg.  All inserts are performed
inside a single transaction per batch; any failure triggers a full rollback and
re-raises the exception so that the consumer can withhold the Kafka offset commit.
"""

from datetime import datetime, timezone
from typing import List

import structlog
from sqlalchemy import BigInteger, Column, DateTime, Float, MetaData, String, Table, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from .config import settings
from .schemas import InferenceResult

logger = structlog.get_logger(__name__)

# ── SQLAlchemy async engine (created at import time) ─────────────────────────
engine = create_async_engine(
    settings.POSTGRES_DSN,
    pool_pre_ping=True,
    echo=False,
)

# ── Table definition ──────────────────────────────────────────────────────────
metadata = MetaData()

inference_audit = Table(
    "inference_audit",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("payload_id", String, nullable=False),
    Column("label", String, nullable=False),
    Column("confidence", Float, nullable=False),
    Column("processed_at", DateTime(timezone=True), nullable=False),
    Column(
        "logged_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("NOW()"),
    ),
)

# ── Session factory ───────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ── Public API ────────────────────────────────────────────────────────────────

async def init_db() -> None:
    """Create the ``inference_audit`` table if it does not already exist.

    Safe to call on every startup — uses ``checkfirst=True`` semantics via
    ``CREATE TABLE IF NOT EXISTS`` equivalent from SQLAlchemy's ``create_all``.
    """
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    logger.info("audit_db_initialized", dsn=settings.POSTGRES_DSN)


async def log_batch(results: List[InferenceResult]) -> None:
    """Bulk-insert a batch of inference results into ``inference_audit``.

    All rows are inserted in a single transaction.  If the insert fails the
    transaction is rolled back and the exception is re-raised so that the
    calling consumer worker can skip the Kafka offset commit.

    Parameters
    ----------
    results:
        List of ``InferenceResult`` objects to persist.

    Raises
    ------
    Exception
        Re-raises any database exception after rolling back the transaction.
    """
    if not results:
        logger.warning("log_batch_called_with_empty_results")
        return

    rows = [
        {
            "payload_id": r.payload_id,
            "label": r.label,
            "confidence": r.confidence,
            "processed_at": r.processed_at,
            "logged_at": datetime.now(timezone.utc),
        }
        for r in results
    ]

    async with AsyncSessionLocal() as session:
        try:
            await session.execute(inference_audit.insert(), rows)
            await session.commit()
            logger.info("audit_batch_logged", count=len(rows))
        except Exception:
            await session.rollback()
            logger.exception("audit_batch_failed", count=len(rows))
            raise
