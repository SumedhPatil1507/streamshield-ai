"""Kafka producer for the StreamShield AI streaming pipeline.

Generates simulated multimodal payloads (text + image URL) and publishes them
to the configured Kafka topic at the rate specified by ``PRODUCER_RATE_HZ``.
The underlying ``confluent_kafka.Producer`` is synchronous; IO-blocking calls
(``produce`` and ``flush``) are dispatched via ``asyncio.get_event_loop().run_in_executor``
so the async event loop is never stalled.
"""

import asyncio
import random
import uuid
from typing import Optional

import structlog
from confluent_kafka import Producer, KafkaException

from .config import settings
from .schemas import MultimodalPayload

logger = structlog.get_logger(__name__)

# ── Predefined content pools ──────────────────────────────────────────────────

_TEXT_POOL = [
    "Suspicious activity detected near the main entrance at 02:14 UTC.",
    "Normal pedestrian traffic observed on platform B during peak hours.",
    "Unattended luggage identified at gate 7 — initiating review protocol.",
    "Crowd density exceeds safety threshold in sector 4; alert dispatched.",
    "False positive confirmed: item identified as a standard travel bag.",
    "High-velocity object trajectory detected; flagging for human review.",
    "Thermal anomaly recorded in server room 3 — potential overheating.",
    "No irregular patterns found in the overnight surveillance window.",
]


def _make_image_url() -> str:
    """Return a plausible HTTPS CDN URL for a dynamically generated image ID."""
    return f"https://cdn.streamshield.ai/img/{uuid.uuid4()}.jpg"


# ── Delivery callback ─────────────────────────────────────────────────────────

def _delivery_callback(err: Optional[Exception], msg: object) -> None:
    """Confluent-Kafka delivery report callback.

    Called by the producer for every message that has been acknowledged (or
    failed) by the broker.  Errors are logged at ERROR level; successes at
    DEBUG level.

    Parameters
    ----------
    err:
        Non-None if delivery failed; ``None`` on success.
    msg:
        The ``confluent_kafka.Message`` object returned by the broker.
    """
    if err is not None:
        logger.error(
            "producer_delivery_failed",
            error=str(err),
            topic=msg.topic() if msg else None,
        )
    else:
        logger.debug(
            "producer_delivery_success",
            topic=msg.topic(),
            partition=msg.partition(),
            offset=msg.offset(),
            key=msg.key().decode("utf-8") if msg.key() else None,
        )


# ── Producer factory ──────────────────────────────────────────────────────────

def _build_producer() -> Producer:
    """Instantiate and return a ``confluent_kafka.Producer``."""
    conf = {
        "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS,
        # Improve throughput while keeping latency acceptable.
        "linger.ms": 5,
        "acks": "all",
        "retries": 5,
        "retry.backoff.ms": 200,
    }
    return Producer(conf)


# ── Async wrapper helpers ─────────────────────────────────────────────────────

async def _produce_async(
    loop: asyncio.AbstractEventLoop,
    producer: Producer,
    topic: str,
    key: bytes,
    value: bytes,
) -> None:
    """Call ``producer.produce`` inside an executor thread."""
    await loop.run_in_executor(
        None,
        lambda: producer.produce(
            topic,
            key=key,
            value=value,
            on_delivery=_delivery_callback,
        ),
    )


async def _flush_async(
    loop: asyncio.AbstractEventLoop,
    producer: Producer,
) -> None:
    """Call ``producer.flush`` inside an executor thread."""
    await loop.run_in_executor(None, producer.flush)


# ── Main coroutine ────────────────────────────────────────────────────────────

async def main() -> None:
    """Produce ``settings.PRODUCER_NUM_MESSAGES`` multimodal payloads to Kafka.

    Each payload is throttled to ``settings.PRODUCER_RATE_HZ`` messages per
    second using ``asyncio.sleep``.  The producer is flushed after all messages
    have been enqueued to ensure delivery before the process exits.
    """
    loop = asyncio.get_event_loop()
    producer = _build_producer()

    logger.info(
        "producer_started",
        topic=settings.KAFKA_TOPIC,
        rate_hz=settings.PRODUCER_RATE_HZ,
        num_messages=settings.PRODUCER_NUM_MESSAGES,
    )

    interval = 1.0 / settings.PRODUCER_RATE_HZ

    for i in range(settings.PRODUCER_NUM_MESSAGES):
        payload = MultimodalPayload(
            text_content=random.choice(_TEXT_POOL),
            image_url=_make_image_url(),
        )

        value_bytes = payload.model_dump_json().encode("utf-8")
        key_bytes = payload.payload_id.encode("utf-8")

        try:
            await _produce_async(loop, producer, settings.KAFKA_TOPIC, key_bytes, value_bytes)
        except KafkaException as exc:
            logger.error(
                "producer_kafka_error",
                payload_id=payload.payload_id,
                error=str(exc),
            )
            raise

        logger.debug(
            "producer_message_enqueued",
            payload_id=payload.payload_id,
            topic=settings.KAFKA_TOPIC,
            index=i + 1,
        )

        # Throttle to the configured rate without blocking the event loop.
        await asyncio.sleep(interval)

    logger.info("producer_flushing")
    await _flush_async(loop, producer)
    logger.info("producer_done", total_messages=settings.PRODUCER_NUM_MESSAGES)


if __name__ == "__main__":
    asyncio.run(main())
