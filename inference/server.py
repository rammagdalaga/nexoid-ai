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


# ── Global model instance (lazy loaded) ─────

_model = None
_tokenizer = None
_cfg = None
_device = None


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

    def do_POST(self):
        parsed = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON")
            return

        if parsed.path == "/v1/completions":
            self._handle_completion(data)
        elif parsed.path == "/v1/chat/completions":
            self._handle_chat(data)
        elif parsed.path == "/v1/generate":
            self._handle_generate(data)
        else:
            self._send_error(404, "Not found")

    def _handle_completion(self, data: Dict):
        """Handle OpenAI-compatible completion requests."""
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 256)
        temperature = data.get("temperature", 0.8)
        top_k = data.get("top_k", 40)
        top_p = data.get("top_p", 0.95)
        stream = data.get("stream", False)

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
        messages = data.get("messages", [])
        max_tokens = data.get("max_tokens", 256)
        temperature = data.get("temperature", 0.8)

        # Extract last user message as prompt
        prompt = ""
        for msg in messages:
            if msg.get("role") == "user":
                prompt = msg.get("content", "")

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
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 256)
        temperature = data.get("temperature", 0.8)
        top_k = data.get("top_k", 40)
        top_p = data.get("top_p", 0.95)

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