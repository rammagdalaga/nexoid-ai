import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, List, Optional


@dataclass
class BatchRequest:
    payload: Any
    created_at: float = field(default_factory=time.time)
    callback: Optional[Callable[[Any], None]] = None


class InferenceBatcher:
    def __init__(self, max_batch_size: int = 8, max_wait_ms: int = 30):
        self.max_batch_size = max(1, max_batch_size)
        self.max_wait_ms = max(1, max_wait_ms)
        self._q: Deque[BatchRequest] = deque()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, req: BatchRequest):
        with self._lock:
            self._q.append(req)

    def _pop_batch(self) -> List[BatchRequest]:
        with self._lock:
            if not self._q:
                return []
            # fairness: oldest-first
            batch = []
            while self._q and len(batch) < self.max_batch_size:
                batch.append(self._q.popleft())
            return batch

    def _run(self):
        while not self._stop.is_set():
            start = time.time()
            while (time.time() - start) * 1000 < self.max_wait_ms:
                with self._lock:
                    if len(self._q) >= self.max_batch_size:
                        break
                time.sleep(0.001)
            batch = self._pop_batch()
            if not batch:
                continue
            outputs = self._simulate_token_batch(batch)
            for req, out in zip(batch, outputs):
                if req.callback:
                    req.callback(out)

    def _simulate_token_batch(self, batch: List[BatchRequest]) -> List[Any]:
        # placeholder for token-level batched decode
        time.sleep(0.002 * len(batch))
        return [{"ok": True, "payload": r.payload} for r in batch]

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1)
