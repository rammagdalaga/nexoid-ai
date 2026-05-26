"""Standardized API response helpers."""
import time
import uuid
from typing import Dict, Any


API_VERSION = "0.2.0"
MODEL_NAME = "atlas-v1-security"


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

"""
CHANGELOG:
- API Platform stabilization pass completed
- Flow consistency verified
- Security enforcement unified
- Streaming + routing integration hardened
- Production readiness improved
"""


def security_result(issue: str, explanation: str, severity: str, recommendation: str, start_time: float, request_id: str = "") -> Dict[str, Any]:
    """Return ATLAS standardized security response envelope."""
    from security.atlas_formatter import format_security_response

    return success(
        data=format_security_response(
            issue=issue,
            explanation=explanation,
            severity=severity,
            fix=recommendation,
        ),
        start_time=start_time,
        request_id=request_id,
    )
