"""StreamShield AI — standardized explainable moderation report builder.

Every moderation surface of the platform — WebSocket live alerts, the
``POST /moderate/upload`` file API, and the Streamlit cockpit — emits the
*same* four-field explainable JSON contract:

.. code-block:: json

    {
      "verdict": "CRITICAL TOXICITY VIOLATION",
      "reasoning": "Cross-modal Bayesian probability breakdown (Vision: 94.2%, Text/Audio: 91.8%)",
      "recommendation": "Instant Mute / Quarantine WebRTC Track",
      "next_steps": [
        "Terminate active WebRTC media track immediately",
        "Issue automated strike notification to the stream host",
        "Log frame timestamp and 128-d embedding into PostgreSQL audit trail"
      ]
    }

The builder is a pure function of the modality confidences so it can be
unit-tested, serialised identically across transports, and versioned with the
policy engine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Policy thresholds (fused confidence in [0, 1])
CRITICAL_THRESHOLD = 0.85
WARNING_THRESHOLD = 0.50

REPORT_FIELDS = ("verdict", "reasoning", "recommendation", "next_steps")

_LEVEL_VERDICTS = {
    "CRITICAL": "CRITICAL TOXICITY VIOLATION",
    "WARNING": "POLICY RISK DETECTED — REVIEW REQUIRED",
    "SAFE": "CLEAR — NO POLICY VIOLATION",
}

_LEVEL_RECOMMENDATIONS = {
    "CRITICAL": "Instant Mute / Quarantine WebRTC Track",
    "WARNING": "Diminish reach and queue for human moderator review",
    "SAFE": "Continue streaming — no enforcement action required",
}


def classify_alert_level(confidence: float) -> str:
    """Map a fused confidence score to ``SAFE`` / ``WARNING`` / ``CRITICAL``."""
    if confidence >= CRITICAL_THRESHOLD:
        return "CRITICAL"
    if confidence >= WARNING_THRESHOLD:
        return "WARNING"
    return "SAFE"


def build_moderation_report(
    label: str,
    confidence: float,
    text_confidence: float,
    image_confidence: float,
    *,
    alert_level: Optional[str] = None,
    stream_id: str = "",
    frame_ref: str = "",
    source: str = "live_stream",
) -> Dict[str, Any]:
    """Build the standardized explainable moderation JSON payload.

    Parameters
    ----------
    label:
        ``"Toxic"`` or ``"Non-Toxic"``.
    confidence:
        Fused cross-modal risk score in ``[0, 1]``.
    text_confidence / image_confidence:
        Per-modality sigmoid probabilities in ``[0, 1]``.
    alert_level:
        Optional pre-computed ``SAFE`` / ``WARNING`` / ``CRITICAL``; derived
        from *confidence* when omitted.
    stream_id / frame_ref:
        Provenance identifiers embedded into the reasoning/next-steps text so
        alerts are self-describing in downstream audit trails.
    source:
        ``live_stream``, ``webrtc``, ``rtsp`` or ``file_upload``.

    Returns
    -------
    dict
        Exactly the keys ``verdict``, ``reasoning``, ``recommendation``,
        ``next_steps`` (in that order).
    """
    level = alert_level or classify_alert_level(confidence)
    flagged = label == "Toxic" or level in ("WARNING", "CRITICAL")
    if not flagged:
        level = "SAFE"

    vision_pct = image_confidence * 100.0
    text_pct = text_confidence * 100.0
    fused_pct = confidence * 100.0

    verdict = _LEVEL_VERDICTS[level]
    if label == "Toxic" and level == "SAFE":  # label/level disagreement guard
        verdict = "TOXIC CONTENT DETECTED — LOW CONFIDENCE"

    reasoning = (
        f"Cross-modal Bayesian probability breakdown (Vision: {vision_pct:.1f}%, "
        f"Text/Audio: {text_pct:.1f}%); fused risk score {fused_pct:.1f}% "
        f"vs policy thresholds [WARNING≥{WARNING_THRESHOLD * 100:.0f}%, "
        f"CRITICAL≥{CRITICAL_THRESHOLD * 100:.0f}%]. "
        f"Verdict={verdict}."
    )

    recommendation = _LEVEL_RECOMMENDATIONS[level]

    provenance = f" [{source}{':' + stream_id if stream_id else ''}{':' + frame_ref if frame_ref else ''}]"
    if level == "CRITICAL":
        next_steps: List[str] = [
            f"Step 1: Terminate active WebRTC media track immediately{provenance}",
            "Step 2: Issue automated strike notification to the stream host",
            "Step 3: Log frame timestamp and 128-d embedding into PostgreSQL audit trail",
        ]
    elif level == "WARNING":
        next_steps = [
            f"Step 1: Throttle distribution and flag the frame for human review{provenance}",
            "Step 2: Notify the on-call moderation queue with reasoning payload",
            "Step 3: Log frame timestamp and 128-d embedding into PostgreSQL audit trail",
        ]
    else:
        next_steps = [
            f"Step 1: Continue ingest — frame passes all policy checks{provenance}",
            "Step 2: Record clearance decision in the streaming audit log",
            "Step 3: Retain 128-d embedding for tamper-evident audit trail",
        ]

    return {
        "verdict": verdict,
        "reasoning": reasoning,
        "recommendation": recommendation,
        "next_steps": next_steps,
    }
