"""API key authentication and per-key throttling for ApexAI gateway."""
import os
import hmac
from typing import Dict

from security.rate_limit import RateLimitRule, create_rate_limiter


class APIKeyAuth:
    def __init__(self):
        raw = os.environ.get("NEXOID_API_KEY", "")
        self._valid_keys = {k.strip() for k in raw.split(",") if k.strip()}
        self._limiter = create_rate_limiter()
        self._rule = RateLimitRule(limit=120, window_seconds=60)

    def validate(self, key: str) -> bool:
        if not key or not self._valid_keys:
            return False
        for valid in self._valid_keys:
            if hmac.compare_digest(key, valid):
                return True
        return False

    def enforce_rate_limit(self, key: str, endpoint: str) -> bool:
        bucket = f"apikey:{key[:6]}:endpoint:{endpoint}"
        return self._limiter.allow(bucket, self._rule)

    @staticmethod
    def extract_key(headers: Dict[str, str]) -> str:
        return headers.get("X-API-Key", "")
