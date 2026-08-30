"""Clock abstraction for deterministic time in evaluation.

All data-path `datetime.now()` calls should use `clock.now()` instead.
When `FINABOT_EVAL_AS_OF` is set, all data-path time queries return that
fixed time, enabling frozen offline evaluation. Wall-clock operations
(session TTL, heartbeat, telemetry timestamps) stay on the real clock.
"""

from __future__ import annotations

import os
from datetime import datetime


def now() -> datetime:
    """Return the current time, respecting `FINABOT_EVAL_AS_OF` when set.

    Returns
    -------
    datetime
        If FINABOT_EVAL_AS_OF is set and parseable, that fixed datetime.
        Otherwise, ``datetime.now()`` (local system time).
    """
    eval_as_of = os.environ.get("FINABOT_EVAL_AS_OF", "").strip()
    if eval_as_of:
        try:
            return datetime.fromisoformat(eval_as_of)
        except ValueError:
            pass  # fall through to wall clock on unparseable value
    return datetime.now()