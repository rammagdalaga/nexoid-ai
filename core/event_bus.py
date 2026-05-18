import threading
from collections import defaultdict
from typing import Any, Callable, Dict, List


class EventBus:
    def __init__(self):
        self._subs: Dict[str, List[Callable[[Dict[str, Any]], None]]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, event_name: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            self._subs[event_name].append(handler)

    def publish(self, event_name: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            handlers = list(self._subs.get(event_name, []))
        for handler in handlers:
            try:
                handler(payload)
            except Exception:
                # failure isolation: event handlers must not crash publisher
                pass


def default_events() -> List[str]:
    return [
        "training_started",
        "training_failed",
        "checkpoint_saved",
        "inference_batch_processed",
        "security_violation_detected",
    ]
