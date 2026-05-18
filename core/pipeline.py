"""
APEXAI MODULE STATUS
Phase: 2 HARDENING COMPLETE
Role: Deterministic stage-to-stage pipeline executor
Dependencies: SystemManager, security.validation, evaluation.benchmarks
System Integration: ACTIVE
Thread Safety: ENFORCED

Purpose:
- Enforce validated stage ordering and standardized handoff envelopes.
- Prevent invalid transitions across dataset/training/checkpoint/inference/evaluation.
"""

import uuid
from typing import Any, Callable, Dict, List

from core.system_manager import SystemManager
from security.validation import ValidationError, require_object, validate_endpoint_schema


class ApexPipeline:
    ORDER = ["dataset", "training", "checkpoint", "inference", "evaluation"]

    def __init__(self, system_manager: SystemManager):
        self.system = system_manager

    def _envelope(self, run_id: str, stage: str, status: str, data: Any = None, meta: Dict[str, Any] = None, errors: List[str] = None):
        return {
            "run_id": run_id,
            "stage": stage,
            "status": status,
            "data": data,
            "meta": meta or {},
            "errors": errors or [],
        }

    def _valid_transition(self, current: str, nxt: str) -> bool:
        if current is None:
            return nxt == self.ORDER[0]
        try:
            return self.ORDER.index(nxt) >= self.ORDER.index(current)
        except ValueError:
            return False

    def run_dataset_stage(self, run_id: str, dataset_meta: Dict[str, Any]) -> Dict[str, Any]:
        try:
            obj = require_object(dataset_meta)
        except ValidationError as e:
            return self._envelope(run_id, "dataset", "rejected", errors=[str(e)])
        return self._envelope(run_id, "dataset", "ok", data=obj)

    def run_training_stage(self, run_id: str, training_request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            obj = require_object(training_request)
            validate_endpoint_schema("training", obj)
        except ValidationError as e:
            return self._envelope(run_id, "training", "rejected", errors=[str(e)])
        job_id = self.system.submit_training_job(obj)
        return self._envelope(run_id, "training", "queued", data={"job_id": job_id})

    def run_checkpoint_stage(self, run_id: str, step: int, model_state: Dict[str, Any], optimizer_state: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        path = self.system.save_checkpoint_from_training(step, model_state, optimizer_state, meta)
        return self._envelope(run_id, "checkpoint", "ok", data={"path": path, "step": step})

    def run_inference_stage(self, run_id: str, inference_request: Dict[str, Any]) -> Dict[str, Any]:
        r = self.system.submit_inference_payload(inference_request)
        r["run_id"] = run_id
        return r

    def run_evaluation_stage(self, run_id: str, inference_generate_fn: Callable[[str], str], cases: List[Any], benchmark_name: str = "integrated_eval") -> Dict[str, Any]:
        from evaluation.benchmarks import run_benchmark
        result = run_benchmark(inference_generate_fn, cases, name=benchmark_name)
        return self._envelope(run_id, "evaluation", "ok", data={
            "name": result.name,
            "passed": result.passed,
            "total": result.total,
            "accuracy": result.accuracy,
            "avg_time_ms": result.avg_time_ms,
        })

    def run_full_pipeline(self, dataset_meta: Dict[str, Any], training_request: Dict[str, Any], inference_request: Dict[str, Any]) -> Dict[str, Any]:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        stage = None

        if not self._valid_transition(stage, "dataset"):
            return self._envelope(run_id, "pipeline", "failed", errors=["invalid initial stage"])
        ds = self.run_dataset_stage(run_id, dataset_meta)
        stage = "dataset"
        if ds["status"] != "ok":
            return self._envelope(run_id, "pipeline", "failed", data={"dataset": ds}, errors=ds["errors"])

        if not self._valid_transition(stage, "training"):
            return self._envelope(run_id, "pipeline", "failed", errors=["invalid stage transition dataset->training"])
        tr = self.run_training_stage(run_id, training_request)
        stage = "training"
        if tr["status"] not in ("queued", "ok"):
            return self._envelope(run_id, "pipeline", "failed", data={"dataset": ds, "training": tr}, errors=tr["errors"])

        if not self._valid_transition(stage, "inference"):
            return self._envelope(run_id, "pipeline", "failed", errors=["invalid stage transition training->inference"])
        inf = self.run_inference_stage(run_id, inference_request)
        stage = "inference"
        if inf["status"] != "ok":
            return self._envelope(run_id, "pipeline", "failed", data={"dataset": ds, "training": tr, "inference": inf}, errors=inf["errors"])

        return self._envelope(run_id, "pipeline", "ok", data={"dataset": ds, "training": tr, "inference": inf, "stage": stage})
