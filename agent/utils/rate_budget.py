import time
from collections import deque

class RateBudget:
    """Rolling 60s window tracking (requests, tokens) against per-minute limits."""

    def __init__(self, tpm: int, rpm: int, safety: float = 0.90):
        self.tpm = tpm * safety
        self.rpm = rpm * safety
        self.events = deque()

    def _prune(self, now: float):
        while self.events and now - self.events[0][0] >= 60.0:
            self.events.popleft()

    def wait(self, est_tokens: int):
        """Block until sending est_tokens keeps us under both limits."""
        while True:
            now = time.time()
            self._prune(now)
            tok = sum(t for _, t in self.events)
            if tok + est_tokens <= self.tpm and len(self.events) + 1 <= self.rpm:
                return
            sleep_for = (
                60.0 - (now - self.events[0][0]) + 0.5
                if self.events else 1.0
            )
            time.sleep(max(0.5, sleep_for))

    def record(self, tokens: int):
        self.events.append((time.time(), tokens))