"""Standardized API response helpers."""
import time
import uuid
from typing import Dict, Any


API_VERSION = "0.2.0"
MODEL_NAME = "apexai"


def success(data: Dict[str, Any], start_time: float, request_id: str = "") -> Dict[str, Any]:
    rid = request_id or f"req_{uuid.uuid4().hex[:12]}"
    return {
        "id": rid,
        "status": "success",
        "data": data,
        "meta": {
            "latency_ms": int((time.time() - start_time) * 1000),
            "model": MODEL_NAME,
            "version": API_VERSION,
        },
    }


def error(message: str, start_time: float, request_id: str = "") -> Dict[str, Any]:
    rid = request_id or f"req_{uuid.uuid4().hex[:12]}"
    return {
        "id": rid,
        "status": "error",
        "data": {"message": message},
        "meta": {
            "latency_ms": int((time.time() - start_time) * 1000),
            "model": MODEL_NAME,
            "version": API_VERSION,
        },
    }
