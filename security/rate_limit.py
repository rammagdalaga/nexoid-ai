import time
from collections import defaultdict, deque
from dataclasses import dataclass

@dataclass(frozen=True)
class RateLimitRule:
    limit: int
    window_seconds: int


class InMemoryRateLimiter:
    def __init__(self):
        self._events = defaultdict(deque)

    def allow(self, key: str, rule: RateLimitRule) -> bool:
        now = time.time()
        q = self._events[key]
        cutoff = now - rule.window_seconds
        while q and q[0] < cutoff:
            q.popleft()
        if len(q) >= rule.limit:
            return False
        q.append(now)
        return True
