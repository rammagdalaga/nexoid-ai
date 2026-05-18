"""Streaming adapters for incremental token emission via EventBus."""
from typing import Iterable, Dict, Any


def stream_chunks(tokens: Iterable[str], event_bus, request_id: str):
    for i, token in enumerate(tokens):
        chunk = {"id": request_id, "index": i, "token": token}
        event_bus.publish("inference_batch_processed", {"stream": True, "request_id": request_id, "index": i})
        yield chunk


def simple_tokenize(text: str):
    for t in text.split(" "):
        yield t + " "
