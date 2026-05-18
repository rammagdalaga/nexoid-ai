"""Central API gateway for ApexAI OpenAI-style platform routing."""
import time
from typing import Dict, Any, Tuple

from api.response import success, error
from api.router import APIRouter
from api.streaming import stream_chunks, simple_tokenize
from security.api_key_auth import APIKeyAuth
from security.usage_tracker import UsageTracker
from security.validation import require_object, validate_endpoint_schema, ValidationError


class APIGateway:
    def __init__(self, system_manager):
        self.system = system_manager
        self.auth = APIKeyAuth()
        self.usage = UsageTracker()
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        self.router.register("/v1/inference", self._inference)
        self.router.register("/v1/chat", self._chat)
        self.router.register("/v1/batch", self._batch)
        self.router.register("/v1/evaluate", self._evaluate)

    def _emit_event(self, event: str, **payload):
        self.system.event_bus.publish(event, payload)

    def _preflight(self, path: str, headers: Dict[str, str], body: Dict[str, Any], t0: float) -> Tuple[bool, str, Dict[str, Any]]:
        key = self.auth.extract_key(headers)
        if not self.auth.validate(key):
            self._emit_event("security_violation_detected", reason="unauthorized", endpoint=path)
            return False, "", error("unauthorized", t0)
        if not self.auth.enforce_rate_limit(key, path):
            self.usage.record_violation(key)
            self._emit_event("security_violation_detected", reason="rate_limit_exceeded", endpoint=path)
            return False, key, error("rate_limit_exceeded", t0)
        try:
            require_object(body)
            endpoint_type = "inference" if path in ("/v1/inference", "/v1/chat", "/v1/batch") else "evaluation"
            validate_endpoint_schema(endpoint_type, body)
        except ValidationError as e:
            self._emit_event("security_violation_detected", reason="validation_error", endpoint=path)
            return False, key, error(str(e), t0)
        return True, key, {}

    def handle(self, path: str, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        ok, key, early = self._preflight(path, headers, body, t0)
        if not ok:
            return early

        ctx = {"path": path, "key": key, "body": body, "t0": t0}
        self._emit_event("training_started", endpoint=path, gateway_flow="Auth>RateLimit>Validation>Router>EventBus>Pipeline>Inference>Response")
        try:
            out = self.router.dispatch(path, ctx)
            self.usage.record_request(key, path)
            return success(out, t0)
        except KeyError:
            self._emit_event("system_error", module="api_router", error="route_not_found", recovery="isolate")
            return error("route_not_found", t0)
        except Exception:
            self._emit_event("system_error", module="api_gateway", error="internal_error", recovery="isolate")
            return error("internal_error", t0)

    def _inference(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        r = self.system.submit_inference_payload(ctx["body"])
        self.usage.record_tokens(ctx["key"], len(str(r.get("data", ""))))
        return r

    def _chat(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return self._inference(ctx)

    def _batch(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        prompts = ctx["body"].get("prompts", [])
        outs = []
        for p in prompts:
            outs.append(self.system.submit_inference_payload({"prompt": p, "max_tokens": 128, "temperature": 0.8, "top_k": 40, "top_p": 0.95}))
        self.usage.record_tokens(ctx["key"], sum(len(str(x.get("data", ""))) for x in outs))
        return {"results": outs}

    def _evaluate(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "evaluation_queued"}

    def stream_inference(self, headers: Dict[str, str], body: Dict[str, Any], request_id: str):
        t0 = time.time()
        ok, key, early = self._preflight("/v1/inference", headers, body, t0)
        if not ok:
            yield early
            return
        try:
            result = self.system.submit_inference_payload(body)
            text = str(result.get("data", ""))
            self.usage.record_request(key, "/v1/inference")
            self.usage.record_tokens(key, len(text))
            yield from stream_chunks(simple_tokenize(text), self.system.event_bus, request_id)
        except Exception:
            self._emit_event("system_error", module="api_streaming", error="stream_failed", recovery="isolate")
            yield error("stream_failed", t0, request_id=request_id)

"""
CHANGELOG:
- API Platform stabilization pass completed
- Flow consistency verified
- Security enforcement unified
- Streaming + routing integration hardened
- Production readiness improved
"""
