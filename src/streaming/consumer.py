"""Kafka consumer worker for the StreamShield AI streaming pipeline.

Consumes multimodal payloads from Kafka in micro-batches of up to
``settings.BATCH_SIZE`` messages (or ``settings.BATCH_TIMEOUT_MS`` milliseconds,
whichever comes first), runs them through the inference pipeline, and commits
offsets **only** after a successful audit log write to PostgreSQL.

Key design decisions
--------------------
* ``enable.auto.commit`` is ``False`` — offsets are committed manually and only
  after ``audit_logger.log_batch`` returns without raising.
* The synchronous Confluent-Kafka poll loop runs inside ``asyncio.to_thread``
  so it never blocks the event loop.
* Failed batches (inference or audit errors) are retried with exponential
  back-off (base 1 s, ±20 % jitter, max 3 attempts) via ``asyncio.sleep``.
* Graceful shutdown on SIGINT/SIGTERM: the current in-flight batch is drained,
  successfully-processed offsets are committed, then the consumer is closed.
"""

import asyncio
import random
import signal
import time
from typing import List

import structlog
from confluent_kafka import Consumer, KafkaException, Message

from .audit_logger import init_db, log_batch
from .config import settings
from .inference_pipeline import run_batch
from .schemas import MultimodalPayload

logger = structlog.get_logger(__name__)

# ── Shutdown coordination ─────────────────────────────────────────────────────
shutdown_event: asyncio.Event  # initialised inside main() after the loop starts


# ── Synchronous poll thread ───────────────────────────────────────────────────

def _poll_batch(consumer: Consumer, batch_size: int, timeout_ms: int) -> List[Message]:
    """Collect up to *batch_size* messages within *timeout_ms* milliseconds.

    Runs in a worker thread (via ``asyncio.to_thread``) so the async event loop
    is never blocked.

    Parameters
    ----------
    consumer:
        An already-subscribed ``confluent_kafka.Consumer`` instance.
    batch_size:
        Maximum number of messages to collect before returning.
    timeout_ms:
        Wall-clock deadline in milliseconds.  The function returns early with
        whatever it has collected once this elapses.

    Returns
    -------
    list[Message]
        Raw Kafka messages (may be empty if no messages arrived).

    Raises
    ------
    KafkaException
        Propagated immediately if the broker signals an error on a message.
    """
    batch: List[Message] = []
    deadline = time.monotonic() + timeout_ms / 1000.0

    while len(batch) < batch_size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        msg = consumer.poll(timeout=min(remaining, 0.1))
        if msg is None:
            continue
        if msg.error():
            raise KafkaException(msg.error())
        batch.append(msg)

    return batch


# ── Retry helper ──────────────────────────────────────────────────────────────

async def _process_batch_with_retry(
    payloads: List[MultimodalPayload],
    max_retries: int = 3,
    base_delay: float = 1.0,
    jitter_fraction: float = 0.2,
) -> bool:
    """Run inference + audit logging with exponential back-off retries.

    Parameters
    ----------
    payloads:
        Deserialised ``MultimodalPayload`` objects for this micro-batch.
    max_retries:
        Maximum number of attempts before giving up.
    base_delay:
        Base sleep duration (seconds) before the first retry.
    jitter_fraction:
        Fraction of *base_delay* added as random jitter (±).

    Returns
    -------
    bool
        ``True`` if the batch was processed and audited successfully;
        ``False`` if all retries were exhausted.
    """
    for attempt in range(1, max_retries + 1):
        try:
            results = await run_batch(payloads)
            await log_batch(results)
            return True
        except Exception as exc:
            logger.error(
                "batch_processing_failed",
                attempt=attempt,
                max_retries=max_retries,
                error=str(exc),
                exc_info=True,
            )
            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))
                jitter = delay * jitter_fraction * (2 * random.random() - 1)
                sleep_for = max(0.0, delay + jitter)
                logger.info(
                    "batch_retry_backoff",
                    attempt=attempt,
                    sleep_seconds=round(sleep_for, 3),
                )
                await asyncio.sleep(sleep_for)

    return False


# ── Main consumer coroutine ───────────────────────────────────────────────────

async def main() -> None:
    """Subscribe to Kafka, consume micro-batches, infer, audit, and commit.

    Lifecycle
    ---------
    1. ``init_db()`` — ensure the audit table exists.
    2. Build a ``Consumer`` with ``enable.auto.commit: False``.
    3. Subscribe to ``settings.KAFKA_TOPIC``.
    4. Loop: poll a micro-batch in a thread → deserialise → infer → audit →
       commit offsets.  Stop when ``shutdown_event`` is set.
    5. On shutdown: attempt to process any pending in-flight batch, then close.
    """
    global shutdown_event
    shutdown_event = asyncio.Event()

    loop = asyncio.get_running_loop()

    def _handle_signal(sig: signal.Signals) -> None:
        logger.info("shutdown_signal_received", signal=sig.name)
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal, sig)
        except (NotImplementedError, OSError):
            # Windows does not support add_signal_handler for all signals;
            # fall back to signal.signal for SIGINT at minimum.
            signal.signal(sig, lambda s, f: shutdown_event.set())

    logger.info("consumer_initialising_db")
    await init_db()

    consumer_conf = {
        "bootstrap.servers": settings.KAFKA_BOOTSTRAP_SERVERS,
        "group.id": settings.KAFKA_GROUP_ID,
        "auto.offset.reset": settings.KAFKA_AUTO_OFFSET_RESET,
        "enable.auto.commit": False,
        "max.poll.interval.ms": 300000,
    }

    consumer = Consumer(consumer_conf)
    consumer.subscribe([settings.KAFKA_TOPIC])

    logger.info(
        "consumer_started",
        topic=settings.KAFKA_TOPIC,
        group_id=settings.KAFKA_GROUP_ID,
        batch_size=settings.BATCH_SIZE,
        batch_timeout_ms=settings.BATCH_TIMEOUT_MS,
    )

    try:
        while not shutdown_event.is_set():
            # ── Poll a micro-batch in a worker thread ─────────────────────────
            try:
                raw_messages: List = await asyncio.to_thread(
                    _poll_batch,
                    consumer,
                    settings.BATCH_SIZE,
                    settings.BATCH_TIMEOUT_MS,
                )
            except KafkaException as exc:
                logger.error("consumer_poll_kafka_error", error=str(exc))
                # Brief pause before re-polling to avoid a tight error loop.
                await asyncio.sleep(1.0)
                continue

            if not raw_messages:
                continue

            logger.debug(
                "consumer_batch_received",
                raw_count=len(raw_messages),
            )

            # ── Deserialise raw bytes → MultimodalPayload ─────────────────────
            payloads: List[MultimodalPayload] = []
            for msg in raw_messages:
                try:
                    payload = MultimodalPayload.model_validate_json(msg.value())
                    payloads.append(payload)
                except Exception as exc:
                    logger.error(
                        "consumer_deserialise_error",
                        error=str(exc),
                        raw_value=msg.value()[:200] if msg.value() else None,
                    )
                    # Skip poison-pill messages; they will NOT be retried and
                    # their offset will be committed with the batch.

            # ── Inference + audit (with retries) ──────────────────────────────
            if payloads:
                success = await _process_batch_with_retry(payloads)
            else:
                success = True  # All messages were unparseable; commit to advance.

            # ── Commit offsets ONLY on success ────────────────────────────────
            if success:
                try:
                    await asyncio.to_thread(consumer.commit, asynchronous=False)
                    logger.info(
                        "consumer_offsets_committed",
                        batch_size=len(raw_messages),
                    )
                except KafkaException as exc:
                    logger.error("consumer_commit_failed", error=str(exc))
            else:
                logger.error(
                    "consumer_batch_dropped_no_commit",
                    batch_size=len(raw_messages),
                )

    finally:
        logger.info("consumer_closing")
        consumer.close()
        logger.info("consumer_closed")


if __name__ == "__main__":
    asyncio.run(main())
