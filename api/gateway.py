"""Central API gateway for ApexAI OpenAI-style platform routing."""
import time
from typing import Dict, Any

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

    def handle(self, path: str, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.time()
        key = self.auth.extract_key(headers)
        if not self.auth.validate(key):
            return error("unauthorized", t0)
        if not self.auth.enforce_rate_limit(key, path):
            self.usage.record_violation(key)
            return error("rate_limit_exceeded", t0)

        try:
            require_object(body)
            validate_endpoint_schema("inference" if path in ("/v1/inference", "/v1/chat", "/v1/batch") else "evaluation", body)
        except ValidationError as e:
            return error(str(e), t0)

        ctx = {"path": path, "key": key, "body": body, "t0": t0}
        try:
            out = self.router.dispatch(path, ctx)
            self.usage.record_request(key, path)
            return success(out, t0)
        except Exception:
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
        return {"results": outs}

    def _evaluate(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "evaluation_queued"}

    def stream_inference(self, headers: Dict[str, str], body: Dict[str, Any], request_id: str):
        key = self.auth.extract_key(headers)
        if not self.auth.validate(key):
            yield {"id": request_id, "status": "error", "data": {"message": "unauthorized"}}
            return
        result = self.system.submit_inference_payload(body)
        text = str(result.get("data", ""))
        yield from stream_chunks(simple_tokenize(text), self.system.event_bus, request_id)
