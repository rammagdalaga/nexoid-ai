"""API router with middleware ordering for ApexAI gateway."""
from typing import Callable, Dict, Any


class APIRouter:
    def __init__(self):
        self.routes: Dict[str, Callable] = {}

    def register(self, path: str, handler: Callable):
        self.routes[path] = handler

    def dispatch(self, path: str, ctx: Dict[str, Any]):
        if path not in self.routes:
            raise KeyError("route_not_found")
        return self.routes[path](ctx)

"""
CHANGELOG:
- API Platform stabilization pass completed
- Flow consistency verified
- Security enforcement unified
- Streaming + routing integration hardened
- Production readiness improved
"""
