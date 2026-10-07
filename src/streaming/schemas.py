"""Pydantic v2 data models for the StreamShield AI streaming pipeline.

Defines ``MultimodalPayload`` (inbound messages) and ``InferenceResult``
(output from the inference pipeline) used throughout the system.
"""

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field, HttpUrl


class MultimodalPayload(BaseModel):
    """Represents a single multimodal message streamed through Kafka.

    Fields
    ------
    payload_id:
        Unique identifier for the payload (UUID4 string).
    text_content:
        The text component of the multimodal message.
    image_url:
        A publicly accessible HTTPS URL pointing to the image component.
    timestamp:
        UTC timestamp at which the payload was produced.
    metadata:
        Arbitrary key/value pairs for supplementary context.
    """

    payload_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="UUID4 unique identifier for this payload.",
    )
    text_content: str = Field(..., description="Text component of the payload.")
    image_url: str = Field(..., description="HTTPS URL for the image component.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp of payload creation.",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Optional supplementary key/value metadata.",
    )


class InferenceResult(BaseModel):
    """Represents the output of the inference pipeline for a single payload.

    Fields
    ------
    payload_id:
        Matches the ``payload_id`` of the originating ``MultimodalPayload``.
    label:
        Classification label produced by the model (e.g. 'safe', 'flagged').
    confidence:
        Model confidence score in the range [0, 1].
    processed_at:
        UTC timestamp at which inference was completed.
    """

    payload_id: str = Field(..., description="ID of the processed payload.")
    label: str = Field(..., description="Classification label from the model.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score.")
    processed_at: datetime = Field(..., description="UTC timestamp of inference completion.")
