import logging
import os
import time
from contextlib import contextmanager
from typing import List, Tuple

logger = logging.getLogger("regression_studio.perf")

# Opt-in only — disabled by default so production doesn't pay logging overhead
# or leak internal timing/shape data. Enable locally with PERF_LOG=true.
PERF_LOGGING_ENABLED = os.getenv("PERF_LOG", "false").lower() in ("1", "true", "yes")

# In addition to the logger, append each report to a plain file. Request-time
# stdout from this process isn't reliably observable through every log
# viewer/host, so a file gives a guaranteed-readable trail during local
# profiling. Never enabled unless PERF_LOG is set.
_PERF_LOG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "perf.log")
)


class PerfTimer:
    """Collects (stage, milliseconds) timings for one request and prints a
    summary table. No-ops (near-zero overhead) unless PERF_LOG=true."""

    def __init__(self, label: str):
        self.label = label
        self.stages: List[Tuple[str, float]] = []
        self._enabled = PERF_LOGGING_ENABLED

    @contextmanager
    def stage(self, name: str):
        if not self._enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages.append((name, (time.perf_counter() - start) * 1000))

    def report(self):
        if not self._enabled or not self.stages:
            return
        total = sum(ms for _, ms in self.stages)
        width = max(len(name) for name, _ in self.stages)
        lines = [f"[PERF] {self.label}"]
        for name, ms in self.stages:
            lines.append(f"  {name.ljust(width)}  {ms:8.1f}ms")
        lines.append(f"  {'-' * (width + 12)}")
        lines.append(f"  {'TOTAL'.ljust(width)}  {total:8.1f}ms")
        text = "\n".join(lines)
        logger.info("\n" + text)
        with open(_PERF_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(text + "\n\n")
