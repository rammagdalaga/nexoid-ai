"""
APEXAI — REST Inference Server
Provides a scalable REST API for model inference.
Designed for cloud deployment with optional streaming support.

RULES:
  - Never auto-deploy — user must explicitly start the server
  - Cloud-first: designed for cloud inference endpoints
  - Lightweight single-file implementation
"""

import os
import sys
import json
import time
import torch
from typing import Optional, Dict, Any
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.config import ModelConfig
from models.transformer import GPT
from tokenizer.tokenizer import CodeTokenizer
from inference.generate import find_latest_checkpoint, load_model, generate
from inference.streaming import TokenStreamer
from security.rate_limit import create_rate_limiter, RateLimitRule
from security.validation import (
    MAX_BODY_BYTES,
    ValidationError,
    bounded_float,
    bounded_int,
    extract_chat_prompt,
    require_object,
    sanitize_prompt,
    validate_endpoint_schema,
)
from security.logging import create_logger


# ── Global model instance (lazy loaded) ─────

_model = None
_tokenizer = None
_cfg = None
_device = None
_RATE_LIMITER = create_rate_limiter()
_LOGGER = create_logger()
GLOBAL_RULE = RateLimitRule(limit=120, window_seconds=60)
INFER_RULE = RateLimitRule(limit=30, window_seconds=60)
AUTH_RULE = RateLimitRule(limit=5, window_seconds=900)


def get_model(ckpt_path: Optional[str] = None):
    """Lazy-load the model and tokenizer."""
    global _model, _tokenizer, _cfg, _device

    if _model is not None:
        return _model, _tokenizer, _cfg

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = os.path.join(ROOT, "checkpoints")
    ckpt = ckpt_path or find_latest_checkpoint(ckpt_dir)

    _model, _cfg = load_model(ckpt, _device)
    _tokenizer = CodeTokenizer.load(os.path.join(ROOT, "tokenizer", "tokenizer.json"))

    return _model, _tokenizer, _cfg


# ── Request Handler ──────────────────────────

class InferenceHandler(BaseHTTPRequestHandler):
    """HTTP handler for inference requests."""

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/health":
            self._send_json({"status": "ok", "model_loaded": _model is not None})
        elif parsed.path == "/v1/models":
            self._send_json({
                "models": [{
                    "id": "apexai",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "apexai",
                }]
            })
        else:
            self._send_error(404, "Not found")

    def _client_ip(self) -> str:
        xff = self.headers.get("X-Forwarded-For", "")
        if xff:
            return xff.split(",")[0].strip()
        return self.client_address[0]

    def _client_user(self) -> str:
        return self.headers.get("X-User-Id", "anonymous")

    def _rate_limit_or_reject(self, rule: RateLimitRule, bucket: str) -> bool:
        ip = self._client_ip()
        user = self._client_user()
        endpoint = urlparse(self.path).path
        keys = [
            f"{bucket}:ip:{ip}:endpoint:{endpoint}",
            f"{bucket}:user:{user}:endpoint:{endpoint}",
            f"{bucket}:global:endpoint:{endpoint}",
        ]
        for k in keys:
            if not _RATE_LIMITER.allow(k, rule):
                _LOGGER.log("security", "rate_limit_block", ip=ip, user=user, bucket=bucket, endpoint=endpoint)
                self._send_error(429, "Rate limit exceeded")
                return False
        return True

    def do_POST(self):
        parsed = urlparse(self.path)
        if not self._rate_limit_or_reject(GLOBAL_RULE, "global"):
            return

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > MAX_BODY_BYTES:
            self._send_error(413, "Payload too large")
            return

        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            data = require_object(json.loads(body))
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON")
            return
        except ValidationError as e:
            self._send_error(400, str(e))
            return

        try:
            if parsed.path == "/v1/completions":
                validate_endpoint_schema("inference", data)
                if not self._rate_limit_or_reject(INFER_RULE, "inference"):
                    return
                self._handle_completion(data)
            elif parsed.path == "/v1/chat/completions":
                validate_endpoint_schema("inference", data)
                if not self._rate_limit_or_reject(INFER_RULE, "inference"):
                    return
                self._handle_chat(data)
            elif parsed.path == "/v1/generate":
                validate_endpoint_schema("inference", data)
                if not self._rate_limit_or_reject(INFER_RULE, "inference"):
                    return
                self._handle_generate(data)
            elif parsed.path == "/v1/auth/login":
                if not self._rate_limit_or_reject(AUTH_RULE, "auth"):
                    return
                self._send_error(501, "Auth endpoint not implemented")
            else:
                self._send_error(404, "Not found")
        except ValidationError as e:
            self._send_error(400, str(e))

    def _handle_completion(self, data: Dict):
        """Handle OpenAI-compatible completion requests."""
        prompt = sanitize_prompt(data.get("prompt", ""))
        max_tokens = bounded_int(data.get("max_tokens"), "max_tokens", 256, 1, 2048)
        temperature = bounded_float(data.get("temperature"), "temperature", 0.8, 0.0, 2.0)
        top_k = bounded_int(data.get("top_k"), "top_k", 40, 1, 200)
        top_p = bounded_float(data.get("top_p"), "top_p", 0.95, 0.0, 1.0)
        stream = bool(data.get("stream", False))

        model, tokenizer, cfg = get_model()

        if stream:
            self._start_stream()
            streamer = TokenStreamer(model, tokenizer, cfg, _device)
            for token in streamer.generate_stream(
                prompt, max_new_tokens=max_tokens,
                temperature=temperature, top_k=top_k, top_p=top_p
            ):
                self._send_stream_event({"choices": [{"text": token}]})
            self._end_stream()
        else:
            result = generate(
                prompt, model, tokenizer, cfg, _device,
                max_new_tokens=max_tokens,
                temperature=temperature, top_k=top_k, top_p=top_p,
            )
            self._send_json({
                "choices": [{"text": result}],
                "usage": {"prompt_tokens": len(tokenizer.encode(prompt))},
            })

    def _handle_chat(self, data: Dict):
        """Handle chat completion requests (simplified)."""
        prompt = extract_chat_prompt(data.get("messages", []))
        max_tokens = bounded_int(data.get("max_tokens"), "max_tokens", 256, 1, 2048)
        temperature = bounded_float(data.get("temperature"), "temperature", 0.8, 0.0, 2.0)

        model, tokenizer, cfg = get_model()
        result = generate(prompt, model, tokenizer, cfg, _device,
                          max_new_tokens=max_tokens, temperature=temperature)

        self._send_json({
            "choices": [{
                "message": {"role": "assistant", "content": result}
            }]
        })

    def _handle_generate(self, data: Dict):
        """Handle code generation requests."""
        prompt = sanitize_prompt(data.get("prompt", ""))
        max_tokens = bounded_int(data.get("max_tokens"), "max_tokens", 256, 1, 2048)
        temperature = bounded_float(data.get("temperature"), "temperature", 0.8, 0.0, 2.0)
        top_k = bounded_int(data.get("top_k"), "top_k", 40, 1, 200)
        top_p = bounded_float(data.get("top_p"), "top_p", 0.95, 0.0, 1.0)

        model, tokenizer, cfg = get_model()
        result = generate(prompt, model, tokenizer, cfg, _device,
                          max_new_tokens=max_tokens,
                          temperature=temperature, top_k=top_k, top_p=top_p)

        self._send_json({
            "generated_text": prompt + result,
            "generated_code": result,
        })

    def _send_json(self, data: Dict, status: int = 200):
        response = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def _send_error(self, status: int, message: str):
        self._send_json({"error": {"message": message}}, status)

    def _start_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

    def _send_stream_event(self, data: Dict):
        try:
            self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()
        except BrokenPipeError:
            pass

    def _end_stream(self):
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except BrokenPipeError:
            pass

    def log_message(self, format, *args):
        """Suppress default HTTP server logs."""
        if os.environ.get("APEXAI_VERBOSE"):
            super().log_message(format, *args)


def start_server(host: str = "0.0.0.0", port: int = 8080,
                 checkpoint: Optional[str] = None):
    """
    Start the inference server.

    Args:
        host: Host address to bind
        port: Port to listen on
        checkpoint: Path to model checkpoint (None = latest)
    """
    print(f"\n[APEXAI] Starting inference server on {host}:{port}")

    # Pre-load model
    print("[APEXAI] Loading model...")
    _model, _tokenizer, _cfg = get_model(checkpoint)
    print(f"[APEXAI] Model loaded: {_model.num_params()/1e6:.1f}M params")

    print(f"[APEXAI] Server ready at http://{host}:{port}")
    _LOGGER.log("security", "server_start", host=host, port=port, stateless_mode=True)
    print(f"  Endpoints:")
    print(f"    GET  /health          — Health check")
    print(f"    GET  /v1/models        — List models")
    print(f"    POST /v1/completions   — Text completion")
    print(f"    POST /v1/chat/completions — Chat completion")
    print(f"    POST /v1/generate      — Code generation")
    print()

    server = HTTPServer((host, port), InferenceHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[APEXAI] Server shutting down...")
        server.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="APEXAI Inference Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    start_server(args.host, args.port, args.checkpoint)