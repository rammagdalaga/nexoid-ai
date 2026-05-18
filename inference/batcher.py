import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional


@dataclass
class BatchRequest:
    payload: Any
    created_at: float = field(default_factory=time.time)
    callback: Optional[Callable[[Any], None]] = None
    request_id: str = ""


class InferenceBatcher:
    def __init__(self, max_batch_size: int = 8, max_wait_ms: int = 30, max_queue_size: int = 4096):
        self.max_batch_size = max(1, max_batch_size)
        self.max_wait_ms = max(1, max_wait_ms)
        self.max_queue_size = max(1, max_queue_size)
        self._q: Deque[BatchRequest] = deque()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._req_seq = 0
        self._inflight: Dict[str, BatchRequest] = {}
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, req: BatchRequest) -> str:
        with self._lock:
            if len(self._q) >= self.max_queue_size:
                raise RuntimeError("inference batch queue full")
            self._req_seq += 1
            req.request_id = req.request_id or f"req_{self._req_seq}"
            self._q.append(req)
            self._inflight[req.request_id] = req
            return req.request_id

    def _pop_batch(self) -> List[BatchRequest]:
        with self._lock:
            if not self._q:
                return []
            batch = []
            while self._q and len(batch) < self.max_batch_size:
                batch.append(self._q.popleft())
            return batch

    def _run(self):
        while not self._stop.is_set():
            batch = self._wait_and_collect_batch()
            if not batch:
                continue
            outputs = self._simulate_token_batch(batch)
            for req, out in zip(batch, outputs):
                try:
                    if req.callback:
                        req.callback(out)
                finally:
                    with self._lock:
                        self._inflight.pop(req.request_id, None)
            del batch
            del outputs

    def _wait_and_collect_batch(self) -> List[BatchRequest]:
        start = time.time()
        while not self._stop.is_set() and (time.time() - start) * 1000 < self.max_wait_ms:
            with self._lock:
                qlen = len(self._q)
            if qlen >= self.max_batch_size:
                break
            if qlen > 0 and (time.time() - start) * 1000 >= self.max_wait_ms * 0.5:
                break
            time.sleep(0.001)
        return self._pop_batch()

    def _simulate_token_batch(self, batch: List[BatchRequest]) -> List[Any]:
        time.sleep(0.0015 * len(batch))
        return [{"ok": True, "request_id": r.request_id, "payload": r.payload} for r in batch]

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)
        with self._lock:
            while self._q:
                req = self._q.popleft()
                if req.callback:
                    req.callback({"ok": False, "request_id": req.request_id, "error": "batcher_shutdown"})
            self._inflight.clear()
