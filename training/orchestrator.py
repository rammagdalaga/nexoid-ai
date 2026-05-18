import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_PAUSED = "paused"
JOB_FAILED = "failed"
JOB_COMPLETED = "completed"


@dataclass
class TrainingJob:
    config: Dict[str, Any]
    job_id: str = field(default_factory=lambda: f"job_{uuid.uuid4().hex[:12]}")
    status: str = JOB_QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    error: Optional[str] = None


class Worker:
    def __init__(self, worker_id: str, simulated_gpus: int = 1):
        self.worker_id = worker_id
        self.simulated_gpus = max(1, simulated_gpus)
        self.is_busy = False

    def run(self, job: TrainingJob):
        self.is_busy = True
        job.status = JOB_RUNNING
        job.started_at = time.time()
        try:
            # Simulated distributed execution runtime
            simulated_steps = int(job.config.get("simulated_steps", 5))
            for _ in range(simulated_steps):
                if job.status == JOB_PAUSED:
                    while job.status == JOB_PAUSED:
                        time.sleep(0.1)
                time.sleep(0.05 / self.simulated_gpus)
            job.status = JOB_COMPLETED
        except Exception as e:
            job.status = JOB_FAILED
            job.error = str(e)
        finally:
            job.ended_at = time.time()
            self.is_busy = False


class TrainingOrchestrator:
    def __init__(self, num_workers: int = 1, gpus_per_worker: int = 1):
        self.job_queue: "queue.Queue[TrainingJob]" = queue.Queue()
        self.jobs: Dict[str, TrainingJob] = {}
        self.workers: List[Worker] = [
            Worker(worker_id=f"worker_{i}", simulated_gpus=gpus_per_worker)
            for i in range(max(1, num_workers))
        ]
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._dispatcher = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._dispatcher.start()

    def submit_job(self, config: Dict[str, Any]) -> str:
        job = TrainingJob(config=config)
        with self._lock:
            self.jobs[job.job_id] = job
        self.job_queue.put(job)
        return job.job_id

    def pause_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job.status != JOB_RUNNING:
            return False
        job.status = JOB_PAUSED
        return True

    def resume_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if not job or job.status != JOB_PAUSED:
            return False
        job.status = JOB_RUNNING
        return True

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        return self.jobs.get(job_id)

    def _dispatch_loop(self):
        while not self._stop.is_set():
            try:
                job = self.job_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            worker = self._wait_for_available_worker()
            threading.Thread(target=worker.run, args=(job,), daemon=True).start()

    def _wait_for_available_worker(self) -> Worker:
        while True:
            for w in self.workers:
                if not w.is_busy:
                    return w
            time.sleep(0.05)

    def stop(self):
        self._stop.set()
        self._dispatcher.join(timeout=1.0)
