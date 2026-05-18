import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any


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
        # Stub adapter intentionally local-only for future integration.
        return None


class StructuredLogger:
    def __init__(self, routers):
        self.routers = routers

    def log(self, category: str, event: str, **payload):
        entry = LogEvent(category=category, event=event, payload=payload)
        for router in self.routers:
            router.write(entry)


def create_logger() -> StructuredLogger:
    routers = [ConsoleRouter()]
    file_path = os.environ.get("APEXAI_LOG_FILE", "logs/events.jsonl")
    routers.append(FileRouter(file_path))
    if os.environ.get("APEXAI_ENABLE_CLOUD_LOG_STUB", "0") == "1":
        routers.append(CloudStubRouter())
    return StructuredLogger(routers)
