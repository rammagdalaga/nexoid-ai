SCHEMA = {
    "inference_requests": {
        "indexes": ["request_id", "user_id", "ip", "endpoint", "ts"],
    },
    "training_jobs": {
        "indexes": ["job_id", "status", "created_at"],
    },
    "evaluation_jobs": {
        "indexes": ["eval_id", "benchmark", "created_at"],
    },
}
