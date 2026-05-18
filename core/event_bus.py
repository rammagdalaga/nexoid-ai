import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional


class EventBus:
    def __init__(self):
        self._subs: Dict[str, List[Callable[[Dict[str, Any]], None]]] = defaultdict(list)
        self._lock = threading.RLock()
        self._event_seq = 0
        self._history: List[Dict[str, Any]] = []
        self._error_handler: Optional[Callable[[Dict[str, Any]], None]] = None

    def set_error_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            self._error_handler = handler

    def subscribe(self, event_name: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        with self._lock:
            self._subs[event_name].append(handler)

    def publish(self, event_name: str, payload: Dict[str, Any]) -> int:
        with self._lock:
            self._event_seq += 1
            event_id = self._event_seq
            handlers = list(self._subs.get(event_name, []))
            record = {
                "event_id": event_id,
                "event": event_name,
                "ts": time.time(),
                "payload": dict(payload),
                "handlers": len(handlers),
            }
            self._history.append(record)
            if len(self._history) > 5000:
                self._history = self._history[-5000:]

        for handler in handlers:
            try:
                handler(payload)
            except Exception as e:
                err = {
                    "event_id": event_id,
                    "event": event_name,
                    "handler_error": str(e),
                }
                with self._lock:
                    error_handler = self._error_handler
                if error_handler:
                    error_handler(err)
        return event_id

    def recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._history[-max(1, limit):])


def default_events() -> List[str]:
    return [
        "training_started",
        "training_failed",
        "checkpoint_saved",
        "inference_batch_processed",
        "security_violation_detected",
        "system_error",
        "state_reconciled",
    ]
