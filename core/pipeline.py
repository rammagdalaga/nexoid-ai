from typing import Any, Callable, Dict, List

from core.system_manager import SystemManager
from security.validation import ValidationError, require_object, validate_endpoint_schema


class ApexPipeline:
    def __init__(self, system_manager: SystemManager):
        self.system = system_manager

    def _envelope(self, stage: str, status: str, data: Any = None, meta: Dict[str, Any] = None, errors: List[str] = None):
        return {
            "stage": stage,
            "status": status,
            "data": data,
            "meta": meta or {},
            "errors": errors or [],
        }

    def run_dataset_stage(self, dataset_meta: Dict[str, Any]) -> Dict[str, Any]:
        try:
            obj = require_object(dataset_meta)
        except ValidationError as e:
            return self._envelope("dataset", "rejected", errors=[str(e)])
        return self._envelope("dataset", "ok", data=obj)

    def run_training_stage(self, training_request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            obj = require_object(training_request)
            validate_endpoint_schema("training", obj)
        except ValidationError as e:
            return self._envelope("training", "rejected", errors=[str(e)])
        job_id = self.system.submit_training_job(obj)
        return self._envelope("training", "queued", data={"job_id": job_id})

    def run_checkpoint_stage(self, step: int, model_state: Dict[str, Any], optimizer_state: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        path = self.system.save_checkpoint_from_training(step, model_state, optimizer_state, meta)
        return self._envelope("checkpoint", "ok", data={"path": path, "step": step})

    def run_inference_stage(self, inference_request: Dict[str, Any]) -> Dict[str, Any]:
        # mandatory validation path is inside system manager
        return self.system.submit_inference_payload(inference_request)

    def run_evaluation_stage(self, inference_generate_fn: Callable[[str], str], cases: List[Any], benchmark_name: str = "integrated_eval") -> Dict[str, Any]:
        from evaluation.benchmarks import run_benchmark

        result = run_benchmark(inference_generate_fn, cases, name=benchmark_name)
        return self._envelope("evaluation", "ok", data={
            "name": result.name,
            "passed": result.passed,
            "total": result.total,
            "accuracy": result.accuracy,
            "avg_time_ms": result.avg_time_ms,
        })

    def run_full_pipeline(self, dataset_meta: Dict[str, Any], training_request: Dict[str, Any], inference_request: Dict[str, Any]) -> Dict[str, Any]:
        ds = self.run_dataset_stage(dataset_meta)
        if ds["status"] != "ok":
            return self._envelope("pipeline", "failed", errors=ds["errors"])

        tr = self.run_training_stage(training_request)
        if tr["status"] not in ("queued", "ok"):
            return self._envelope("pipeline", "failed", errors=tr["errors"])

        inf = self.run_inference_stage(inference_request)
        if inf["status"] != "ok":
            return self._envelope("pipeline", "failed", errors=inf["errors"])

        return self._envelope("pipeline", "ok", data={"dataset": ds, "training": tr, "inference": inf})
