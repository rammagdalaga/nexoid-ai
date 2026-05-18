from typing import Any, Dict, List

MAX_BODY_BYTES = 1024 * 1024
MAX_PROMPT_CHARS = 20_000
MAX_MESSAGES = 100


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
    return data
