"""
APEXAI MODULE STATUS
Phase: 2 HARDENING COMPLETE
Role: Structured logging and observability routing
Dependencies: stdlib json/os/time, router implementations
System Integration: ACTIVE
Thread Safety: ENFORCED

Purpose:
- Provide consistent JSON logging across security, inference, training, and evaluation paths.
- Support release-readiness observability with optional trace logging mode.
"""

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any

TRACE_MODE = os.environ.get("APEXAI_TRACE_MODE", "0") == "1"


@dataclass
class LogEvent:
    category: str
    event: str
    payload: Dict[str, Any]


class LogRouter(ABC):
    @abstractmethod
    def write(self, event: LogEvent) -> None:
        raise NotImplementedError


class ConsoleRouter(LogRouter):
    def write(self, event: LogEvent) -> None:
        print(json.dumps({
            "ts": int(time.time()),
            "category": event.category,
            "event": event.event,
            **event.payload,
        }, ensure_ascii=False))


class FileRouter(LogRouter):
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def write(self, event: LogEvent) -> None:
        row = {
            "ts": int(time.time()),
            "category": event.category,
            "event": event.event,
            **event.payload,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class CloudStubRouter(LogRouter):
    def write(self, event: LogEvent) -> None:
        return None


class StructuredLogger:
    def __init__(self, routers):
        self.routers = routers

    def log(self, category: str, event: str, **payload):
        entry = LogEvent(category=category, event=event, payload=payload)
        for router in self.routers:
            router.write(entry)

    def trace(self, event: str, **payload):
        if TRACE_MODE:
            self.log("trace", event, **payload)


def create_logger() -> StructuredLogger:
    routers = [ConsoleRouter()]
    file_path = os.environ.get("APEXAI_LOG_FILE", "logs/events.jsonl")
    routers.append(FileRouter(file_path))
    if os.environ.get("APEXAI_ENABLE_CLOUD_LOG_STUB", "0") == "1":
        routers.append(CloudStubRouter())
    return StructuredLogger(routers)
