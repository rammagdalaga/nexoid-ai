import json
import time

def log_event(event_type: str, **fields):
    payload = {
        "ts": int(time.time()),
        "event": event_type,
        **fields,
    }
    print(json.dumps(payload, ensure_ascii=False))
