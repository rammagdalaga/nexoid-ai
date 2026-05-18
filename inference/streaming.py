"""
APEXAI — Streaming Generation
Supports token-by-token streaming output for interactive generation.
"""

import asyncio
import torch
from typing import Optional, AsyncGenerator, Callable
from models.transformer import GPT
from tokenizer.tokenizer import CodeTokenizer


class TokenStreamer:
    """
    Streams generated tokens one by one.
    Useful for real-time display during generation.
    """

    def __init__(self, model: GPT, tokenizer: CodeTokenizer, cfg,
                 device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        self.generated_tokens = []
        self._stop_requested = False

    def stop(self):
        """Request early stopping of generation."""
        self._stop_requested = True

    @torch.no_grad()
    def generate_stream(self, prompt: str, max_new_tokens: int = 256,
                        temperature: float = 0.8, top_k: int = 40,
                        top_p: float = 0.95) -> list:
        """
        Generate tokens and yield them one at a time.

        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Yields:
            Generated tokens as decoded strings
        """
        ids = self.tokenizer.encode(prompt)
        if ids and ids[-1] == self.tokenizer.eos_id:
            ids = ids[:-1]

        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        self.model.eval()

        # Enable KV cache
        self.model.enable_kv_cache(batch_size=1, device=self.device)

        try:
            for _ in range(max_new_tokens):
                if self._stop_requested:
                    break

                idx_cond = idx[:, -min(self.cfg.block_size, self.cfg.max_seq_len):]

                if self.model.kv_cache and self.model.kv_cache.seen_tokens > 0:
                    idx_cond = idx[:, -1:]

                logits, _ = self.model(idx_cond)
                logits = logits[:, -1, :] / max(temperature, 1e-6)

                # top-k
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                # top-p
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits[remove] = float("-inf")
                    logits = torch.zeros_like(logits).scatter_(
                        1, sorted_idx, sorted_logits)

                probs = F.softmax(logits, dim=-1)
                next_t = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_t], dim=1)

                decoded = self.tokenizer.decode(next_t[0].tolist())
                self.generated_tokens.append(next_t.item())
                yield decoded

        finally:
            self.model.disable_kv_cache()

    def get_full_text(self) -> str:
        """Get the full generated text."""
        return self.tokenizer.decode(self.generated_tokens)


class AsyncTokenStreamer:
    """
    Async version of TokenStreamer for web/server use.
    """

    def __init__(self, model: GPT, tokenizer: CodeTokenizer, cfg,
                 device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        self.generated_tokens = []
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    async def generate_stream(self, prompt: str, max_new_tokens: int = 256,
                               temperature: float = 0.8, top_k: int = 40,
                               top_p: float = 0.95) -> AsyncGenerator[str, None]:
        """
        Async generator that yields tokens as they're generated.
        """
        loop = asyncio.get_event_loop()

        def _sync_generate():
            streamer = TokenStreamer(
                self.model, self.tokenizer, self.cfg, self.device
            )
            for token in streamer.generate_stream(
                prompt, max_new_tokens, temperature, top_k, top_p
            ):
                if self._stop_requested:
                    streamer.stop()
                    break
                self.generated_tokens.extend(streamer.generated_tokens)
                yield token

        gen = _sync_generate()
        for token in gen:
            yield token
            await asyncio.sleep(0)


# To avoid circular imports
import torch.nn.functional as F