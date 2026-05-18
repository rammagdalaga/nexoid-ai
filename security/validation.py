from typing import Any, Dict

MAX_BODY_BYTES = 1024 * 1024
MAX_PROMPT_CHARS = 20_000
MAX_MESSAGES = 100
MAX_JSON_DEPTH = 20
MAX_ARRAY_ITEMS = 5000
MAX_OBJECT_KEYS = 2000


class ValidationError(ValueError):
    pass


def ensure_type(value: Any, expected, field: str):
    if not isinstance(value, expected):
        raise ValidationError(f"'{field}' has invalid type")


def bounded_int(value: Any, field: str, default: int, min_v: int, max_v: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int):
        raise ValidationError(f"'{field}' must be an integer")
    if value < min_v or value > max_v:
        raise ValidationError(f"'{field}' must be between {min_v} and {max_v}")
    return value


def bounded_float(value: Any, field: str, default: float, min_v: float, max_v: float) -> float:
    if value is None:
        return default
    if not isinstance(value, (int, float)):
        raise ValidationError(f"'{field}' must be a number")
    value = float(value)
    if value < min_v or value > max_v:
        raise ValidationError(f"'{field}' must be between {min_v} and {max_v}")
    return value


def sanitize_prompt(prompt: Any) -> str:
    if prompt is None:
        return ""
    if not isinstance(prompt, str):
        raise ValidationError("'prompt' must be a string")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValidationError("'prompt' is too long")
    return prompt.replace("\x00", "")


def validate_json_shape(obj: Any, depth: int = 0, seen=None):
    if seen is None:
        seen = set()
    if id(obj) in seen:
        raise ValidationError("recursive object detected")
    if depth > MAX_JSON_DEPTH:
        raise ValidationError("payload nesting too deep")

    if isinstance(obj, (dict, list)):
        seen.add(id(obj))

    if isinstance(obj, dict):
        if len(obj) > MAX_OBJECT_KEYS:
            raise ValidationError("object has too many keys")
        for k, v in obj.items():
            if not isinstance(k, str):
                raise ValidationError("object keys must be strings")
            validate_json_shape(v, depth + 1, seen)
    elif isinstance(obj, list):
        if len(obj) > MAX_ARRAY_ITEMS:
            raise ValidationError("array has too many items")
        for item in obj:
            validate_json_shape(item, depth + 1, seen)


def extract_chat_prompt(messages: Any) -> str:
    if messages is None:
        return ""
    ensure_type(messages, list, "messages")
    if len(messages) > MAX_MESSAGES:
        raise ValidationError("'messages' exceeds maximum length")
    last = ""
    for m in messages:
        ensure_type(m, dict, "messages[]")
        role = m.get("role")
        content = m.get("content", "")
        if role == "user":
            if not isinstance(content, str):
                raise ValidationError("message content must be a string")
            last = content
    return sanitize_prompt(last)


def require_object(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValidationError("request body must be a JSON object")
    validate_json_shape(data)
    return data


def validate_endpoint_schema(endpoint_type: str, data: Dict[str, Any]):
    if endpoint_type == "inference":
        allowed = {"prompt", "max_tokens", "temperature", "top_k", "top_p", "stream", "messages"}
    elif endpoint_type == "training":
        allowed = {"config", "max_iters", "resume", "deepspeed_config"}
    elif endpoint_type == "evaluation":
        allowed = {"benchmark", "checkpoint", "max_tokens"}
    else:
        raise ValidationError("unknown endpoint type")

    extra = set(data.keys()) - allowed
    if extra:
        raise ValidationError(f"unexpected fields: {sorted(extra)}")
