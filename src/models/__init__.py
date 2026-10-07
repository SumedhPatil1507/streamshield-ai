"""
StreamShield AI — models package.

Exposes the high-level ``run_batch`` coroutine and ``get_engine`` singleton
accessor so that consumers of this package only need to import from here.
"""

from .inference import ModerationResult, get_engine, run_batch

__all__ = ["ModerationResult", "get_engine", "run_batch"]
