"""Configuration module for the StreamShield AI streaming pipeline.

All settings are read from environment variables with sensible defaults so that
the application can run locally without any additional setup and can be
reconfigured for production via env vars or a ``.env`` file.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables.

    All fields have defaults suitable for a local development environment.
    Override by exporting the corresponding environment variable or by placing
    a ``.env`` file in the working directory.
    """

    # ── Kafka ─────────────────────────────────────────────────────────────────
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC: str = "multimodal-stream"
    KAFKA_GROUP_ID: str = "streamshield-consumers"
    KAFKA_AUTO_OFFSET_RESET: str = "earliest"
    KAFKA_MAX_POLL_RECORDS: int = 16

    # ── Batching ──────────────────────────────────────────────────────────────
    BATCH_SIZE: int = 16
    BATCH_TIMEOUT_MS: int = 500

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    POSTGRES_DSN: str = "postgresql+asyncpg://user:password@localhost:5432/streamshield"

    # ── Producer ──────────────────────────────────────────────────────────────
    PRODUCER_RATE_HZ: float = 10.0
    PRODUCER_NUM_MESSAGES: int = 1000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Module-level singleton — import this everywhere instead of instantiating
# Settings() repeatedly.
settings = Settings()
