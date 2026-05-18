# utils/patterns.py
import time
import logging
from contextlib import contextmanager
from typing import Iterator


# ── Custom Exceptions ───────────────────────

class AppError(Exception):
    """Base application error."""
    def __init__(self, message: str, code: int = 500):
        super().__init__(message)
        self.code    = code
        self.message = message

    def __repr__(self):
        return f"{self.__class__.__name__}(code={self.code}, message={self.message!r})"


class NotFoundError(AppError):
    def __init__(self, resource: str, id=None):
        msg = f"{resource} not found" + (f": {id}" if id else "")
        super().__init__(msg, 404)


class ValidationError(AppError):
    def __init__(self, field: str, reason: str):
        super().__init__(f"Validation failed on '{field}': {reason}", 422)
        self.field = field


class AuthError(AppError):
    def __init__(self, message="Unauthorized"):
        super().__init__(message, 401)


# ── Context Managers ────────────────────────

@contextmanager
def timer_ctx(label: str = "block") -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[timer] {label}: {elapsed:.4f}s")


@contextmanager
def suppress_errors(*exc_types) -> Iterator[None]:
    try:
        yield
    except exc_types:
        pass


@contextmanager
def temp_file(suffix=".tmp") -> Iterator[str]:
    import tempfile
    import os
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        os.close(fd)
        yield path
    finally:
        if os.path.exists(path):
            os.remove(path)


class ManagedResource:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        print(f"[resource] opening {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"[resource] closing {self.name}")
        return False   # don't suppress exceptions

    def use(self):
        return f"using {self.name}"


# ── Iterators ───────────────────────────────

class Counter:
    def __init__(self, start: int = 0, step: int = 1, stop: int = None):
        self.current = start
        self.step    = step
        self.stop    = stop

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop is not None and self.current >= self.stop:
            raise StopIteration
        val           = self.current
        self.current += self.step
        return val


class Fibonacci:
    def __init__(self, limit: int):
        self.limit = limit
        self.a     = 0
        self.b     = 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.a > self.limit:
            raise StopIteration
        val       = self.a
        self.a, self.b = self.b, self.a + self.b
        return val


class InfiniteRange:
    def __init__(self, start: int = 0, step: int = 1):
        self.current = start
        self.step    = step

    def __iter__(self):
        return self

    def __next__(self):
        val           = self.current
        self.current += self.step
        return val

    def take(self, n: int):
        result = []
        for _ in range(n):
            result.append(next(self))
        return result


# ── Logging Setup ───────────────────────────

def setup_logger(name: str, level=logging.DEBUG,
                 fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s") -> logging.Logger:
    logger    = logging.getLogger(name)
    logger.setLevel(level)
    handler   = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


# ── Data Validation ─────────────────────────

def validate_email(email: str) -> bool:
    import re
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, email))


def validate_url(url: str) -> bool:
    import re
    pattern = r"^https?://[^\s/$.?#].[^\s]*$"
    return bool(re.match(pattern, url))


def validate_phone(phone: str) -> bool:
    import re
    cleaned = re.sub(r"[\s\-\(\)]", "", phone)
    return bool(re.match(r"^\+?\d{7,15}$", cleaned))


def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))


def chunks(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep))
        else:
            items[new_key] = v
    return items