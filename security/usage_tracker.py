"""In-memory API usage tracking for per-key observability."""
import threading
from collections import defaultdict
from typing import Dict, Any


class UsageTracker:
    def __init__(self):
        self._lock = threading.RLock()
        self._requests = defaultdict(int)
        self._tokens = defaultdict(int)
        self._endpoint = defaultdict(lambda: defaultdict(int))
        self._violations = defaultdict(int)

    def record_request(self, key: str, endpoint: str):
        with self._lock:
            self._requests[key] += 1
            self._endpoint[key][endpoint] += 1

    def record_tokens(self, key: str, tokens: int):
        with self._lock:
            self._tokens[key] += max(0, int(tokens))

    def record_violation(self, key: str):
        with self._lock:
            self._violations[key] += 1

    def snapshot(self, key: str) -> Dict[str, Any]:
        with self._lock:
            return {
                "requests": self._requests[key],
                "tokens": self._tokens[key],
                "endpoint_usage": dict(self._endpoint[key]),
                "rate_limit_violations": self._violations[key],
            }

"""
CHANGELOG:
- API Platform stabilization pass completed
- Flow consistency verified
- Security enforcement unified
- Streaming + routing integration hardened
- Production readiness improved
"""
