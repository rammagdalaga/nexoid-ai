import json
import urllib.request
import urllib.error


class NexoidClient:
    def __init__(self, config: dict):
        self.base = config["api_base_url"].rstrip("/")
        self.key = config["api_key"]

    def _post(self, endpoint: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base + endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "X-API-Key": self.key},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError:
            return {"status": "error", "data": {"message": "request_failed"}}
        except Exception:
            return {"status": "error", "data": {"message": "connection_failed"}}

    def chat(self, prompt: str, temperature: float = 0.7) -> dict:
        return self._post("/v1/chat", {"prompt": prompt, "temperature": temperature, "max_tokens": 256, "top_k": 40, "top_p": 0.95})

    def inference(self, prompt: str, temperature: float = 0.7) -> dict:
        return self._post("/v1/inference", {"prompt": prompt, "temperature": temperature, "max_tokens": 256, "top_k": 40, "top_p": 0.95})

    def stream(self, prompt: str, temperature: float = 0.7):
        # Uses standard endpoint and streams locally by chunking returned text if stream endpoint unavailable.
        res = self.inference(prompt, temperature)
        text = str(res.get("data", res))
        for tok in text.split(" "):
            yield tok + " "
