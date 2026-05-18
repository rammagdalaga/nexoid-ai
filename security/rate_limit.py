import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Deque


@dataclass(frozen=True)
class RateLimitRule:
    limit: int
    window_seconds: int


class RateLimiterBackend(ABC):
    @abstractmethod
    def allow(self, key: str, rule: RateLimitRule) -> bool:
        raise NotImplementedError


class InMemoryRateLimiter(RateLimiterBackend):
    def __init__(self):
        self._events: Dict[str, Deque[float]] = defaultdict(deque)

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


class SimulatedRedisRateLimiter(RateLimiterBackend):
    """
    Redis-like distributed limiter simulation.
    Uses process-local map but key format and API mimic centralized backend.
    """
    _shared_events: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, key: str, rule: RateLimitRule) -> bool:
        now = time.time()
        q = self._shared_events[key]
        cutoff = now - rule.window_seconds
        while q and q[0] < cutoff:
            q.popleft()
        if len(q) >= rule.limit:
            return False
        q.append(now)
        return True


class RedisRateLimiter(RateLimiterBackend):
    """
    Optional redis backend. Falls back to simulation if redis package/server unavailable.
    """
    def __init__(self, redis_url: str):
        self._redis_url = redis_url
        self._client = None
        self._fallback = SimulatedRedisRateLimiter()
        try:
            import redis  # type: ignore
            self._client = redis.Redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
        except Exception:
            self._client = None

    def allow(self, key: str, rule: RateLimitRule) -> bool:
        if self._client is None:
            return self._fallback.allow(key, rule)
        now = int(time.time())
        window_key = f"rl:{key}:{now // rule.window_seconds}"
        count = self._client.incr(window_key)
        if count == 1:
            self._client.expire(window_key, rule.window_seconds + 1)
        return count <= rule.limit


def create_rate_limiter() -> RateLimiterBackend:
    mode = os.environ.get("APEXAI_RATE_LIMIT_BACKEND", "memory").lower().strip()
    if mode == "redis":
        redis_url = os.environ.get("APEXAI_REDIS_URL", "redis://localhost:6379/0")
        return RedisRateLimiter(redis_url)
    if mode == "simulated_redis":
        return SimulatedRedisRateLimiter()
    return InMemoryRateLimiter()
