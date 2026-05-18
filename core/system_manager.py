import os
import threading
import time
from typing import Any, Dict, Optional

from core.event_bus import EventBus, default_events
from inference.batcher import InferenceBatcher, BatchRequest
from security.validation import ValidationError, require_object, validate_endpoint_schema
from system.health_monitor import HealthMonitor
from training.checkpoint_manager import CheckpointManager
from training.orchestrator import TrainingOrchestrator, JOB_FAILED


class SystemManager:
    STATE_BOOT = "boot"
    STATE_TRAINING = "training"
    STATE_INFERENCE = "inference"
    STATE_EVALUATION = "evaluation"
    STATE_SHUTDOWN = "shutdown"

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.state = self.STATE_BOOT
        self.event_bus = EventBus()
        self.training_orchestrator = TrainingOrchestrator(num_workers=1, gpus_per_worker=1)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir, keep_last_n=5)
        self.inference_batcher = InferenceBatcher(max_batch_size=8, max_wait_ms=30)
        self.health_monitor = HealthMonitor(window=100)
        self._lock = threading.Lock()
        self._setup_subscriptions()
        self._register_event_contract()

    def _register_event_contract(self):
        self._known_events = set(default_events())

    def _setup_subscriptions(self):
        self.event_bus.subscribe("training_failed", self._on_training_failed)
        self.event_bus.subscribe("inference_batch_processed", self._on_batch_processed)

    def _on_training_failed(self, payload: Dict[str, Any]):
        self.health_monitor.record_error(True)

    def _on_batch_processed(self, payload: Dict[str, Any]):
        latency = float(payload.get("latency_s", 0.0))
        self.health_monitor.record_inference_latency(latency)

    def boot(self):
        with self._lock:
            self.state = self.STATE_BOOT

    def shutdown(self):
        with self._lock:
            self.state = self.STATE_SHUTDOWN
        self.training_orchestrator.stop()
        self.inference_batcher.stop()

    def submit_training_job(self, config: Dict[str, Any]) -> str:
        with self._lock:
            self.state = self.STATE_TRAINING
        job_id = self.training_orchestrator.submit_job(config)
        self.event_bus.publish("training_started", {"job_id": job_id, "config": dict(config)})
        return job_id

    def check_training_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self.training_orchestrator.get_job(job_id)
        if not job:
            return None
        if job.status == JOB_FAILED:
            self.event_bus.publish("training_failed", {"job_id": job_id, "error": job.error})
        return {
            "job_id": job.job_id,
            "status": job.status,
            "started_at": job.started_at,
            "ended_at": job.ended_at,
            "error": job.error,
        }

    def save_checkpoint_from_training(self, step: int, model_state: Dict[str, Any], optimizer_state: Dict[str, Any], meta: Dict[str, Any]) -> str:
        path = self.checkpoint_manager.save(step, model_state, optimizer_state, meta)
        self.event_bus.publish("checkpoint_saved", {"step": step, "path": path})
        return path

    def submit_inference_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self.state = self.STATE_INFERENCE
        start = time.time()
        try:
            obj = require_object(payload)
            validate_endpoint_schema("inference", obj)
        except ValidationError as e:
            self.event_bus.publish("security_violation_detected", {"error": str(e), "module": "inference"})
            self.health_monitor.record_error(True)
            return {"stage": "inference", "status": "rejected", "data": None, "meta": {}, "errors": [str(e)]}

        out = {"result": None}

        def _cb(x):
            out["result"] = x

        self.inference_batcher.submit(BatchRequest(payload=obj, callback=_cb))
        deadline = time.time() + 2.0
        while out["result"] is None and time.time() < deadline:
            time.sleep(0.005)

        latency = time.time() - start
        self.event_bus.publish("inference_batch_processed", {"latency_s": latency})
        return {"stage": "inference", "status": "ok", "data": out["result"], "meta": {"latency_s": latency}, "errors": []}

    def rollback_to_latest_checkpoint(self) -> Dict[str, Any]:
        step, payload = self.checkpoint_manager.resume_latest(verify_integrity=True)
        if step is None or payload is None:
            return {"rolled_back": False, "reason": "no_checkpoint"}
        return {"rolled_back": True, "step": step, "meta": payload.get("meta", {})}

    def health(self) -> Dict[str, Any]:
        return {"state": self.state, "health": self.health_monitor.snapshot(), "stateless": True}
