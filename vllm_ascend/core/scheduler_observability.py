"""Optional low-rate scheduler queue diagnostics."""

from __future__ import annotations

import os
import time
from typing import Any

from vllm.logger import logger


def log_scheduler_queues(scheduler: Any, scheduler_output: Any) -> None:
    """Log local queue state when ``VLLM_ASCEND_SCHEDULER_LOG_INTERVAL`` is set.

    The value is a minimum interval in seconds. Zero or an invalid value keeps
    the diagnostics disabled so production scheduling has no extra log work.
    """
    try:
        interval = float(os.environ.get("VLLM_ASCEND_SCHEDULER_LOG_INTERVAL", "0"))
    except ValueError:
        return
    if interval <= 0:
        return

    now = time.monotonic()
    last = getattr(scheduler, "_queue_log_timestamp", 0.0)
    if now - last < interval:
        return
    scheduler._queue_log_timestamp = now

    waiting = getattr(scheduler, "waiting", ())
    skipped = getattr(scheduler, "skipped_waiting", ())
    scheduled = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
    parallel_config = getattr(getattr(scheduler, "vllm_config", None),
                              "parallel_config", None)
    dp_rank = getattr(parallel_config, "data_parallel_rank", None)
    dp_index = getattr(parallel_config, "data_parallel_index", None)
    dp_size = getattr(parallel_config, "data_parallel_size", 1)
    logger.info(
        "scheduler queues | dp=%s/%s index=%s running=%d waiting=%d "
        "skipped_waiting=%d scheduled_reqs=%d scheduled_tokens=%d",
        dp_rank if dp_rank is not None else "?",
        dp_size,
        dp_index if dp_index is not None else "?",
        len(getattr(scheduler, "running", ())),
        len(waiting),
        len(skipped),
        len(scheduled),
        sum(scheduled.values()),
    )
