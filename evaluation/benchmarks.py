"""
APEXAI — Core Benchmark Suite
Standardized evaluation benchmarks for coding intelligence.
Includes support for HumanEval, MBPP, and custom benchmarks.
"""

import ast
import re
import sys
import json
import time
import subprocess
import tempfile
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field


@dataclass
class BenchmarkResult:
    """Results from a single benchmark evaluation."""
    name: str
    passed: int
    total: int
    accuracy: float
    avg_time_ms: float
    errors: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)


@dataclass
class EvalCase:
    """A single evaluation test case."""
    prompt: str
    reference_code: str
    test_code: str
    language: str = "python"
    timeout_s: int = 10


# ── Syntax Validation ────────────────────────

def validate_syntax(code: str, language: str = "python") -> Dict:
    """
    Validate code syntax.

    Args:
        code: Source code to validate
        language: Programming language

    Returns:
        Dict with 'valid' bool and 'error' string if invalid
    """
    result = {"valid": True, "error": None}

    try:
        if language == "python":
            ast.parse(code)
        elif language in ("javascript", "typescript", "jsx", "tsx"):
            # Basic bracket matching for JS/TS
            stack = []
            pairs = {"(": ")", "{": "}", "[": "]"}
            for char in code:
                if char in pairs:
                    stack.append(pairs[char])
                elif char in (")", "}", "]") and stack:
                    expected = stack.pop()
                    if char != expected:
                        result["valid"] = False
                        result["error"] = f"Mismatched bracket: expected {expected}, got {char}"
                        return result
        # HTML validation
        elif language == "html":
            # Check for balanced tags (basic)
            open_tags = []
            import re
            for match in re.finditer(r'</?(\w+)[^>]*>', code):
                tag = match.group(1)
                if tag in ("br", "hr", "img", "input", "meta", "link"):
                    continue
                if match.group(0).startswith("</"):
                    if open_tags and open_tags[-1] == tag:
                        open_tags.pop()
                    else:
                        result["valid"] = False
                        result["error"] = f"Mismatched closing tag: {tag}"
                        return result
                else:
                    open_tags.append(tag)
    except SyntaxError as e:
        result["valid"] = False
        result["error"] = str(e)
    except Exception as e:
        result["valid"] = False
        result["error"] = f"Validation error: {e}"

    return result


# ── Execution-Based Testing ──────────────────

def run_code_test(code: str, test_code: str, timeout_s: int = 10) -> Dict:
    """
    Run generated code against test cases in a subprocess.

    Args:
        code: The generated code
        test_code: Test code to execute against
        timeout_s: Timeout in seconds

    Returns:
        Dict with 'passed' bool, 'output', and 'error'
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code + "\n\n# Tests\n" + test_code)
        temp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        import os
        os.unlink(temp_path)

        return {
            "passed": result.returncode == 0,
            "output": result.stdout,
            "error": result.stderr if result.returncode != 0 else None,
        }
    except subprocess.TimeoutExpired:
        import os
        os.unlink(temp_path)
        return {"passed": False, "output": "", "error": "Timeout exceeded"}
    except Exception as e:
        import os
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return {"passed": False, "output": "", "error": str(e)}


# ── HumanEval-style Benchmarks ───────────────

HUMANEVAL_SAMPLES = [
    EvalCase(
        prompt="Write a function that returns the sum of two numbers.",
        reference_code="""def add(a: int, b: int) -> int:
    return a + b
""",
        test_code="""assert add(1, 2) == 3
assert add(-1, 1) == 0
assert add(0, 0) == 0
assert add(100, -50) == 50
print("All tests passed!")
""",
    ),
    EvalCase(
        prompt="Write a function that checks if a string is a palindrome.",
        reference_code="""def is_palindrome(s: str) -> bool:
    s = s.lower().replace(' ', '')
    return s == s[::-1]
""",
        test_code="""assert is_palindrome("racecar") == True
assert is_palindrome("hello") == False
assert is_palindrome("A man a plan a canal Panama") == True
assert is_palindrome("") == True
print("All tests passed!")
""",
    ),
    EvalCase(
        prompt="Write a function that finds the maximum element in a list.",
        reference_code="""def find_max(arr: list) -> int:
    if not arr:
        return None
    return max(arr)
""",
        test_code="""assert find_max([1, 2, 3, 4, 5]) == 5
assert find_max([-10, -5, -1]) == -1
assert find_max([0]) == 0
assert find_max([]) == None
print("All tests passed!")
""",
    ),
]

MBPP_SAMPLES = [
    EvalCase(
        prompt="Write a Python function to remove duplicates from a list.",
        reference_code="""def remove_duplicates(lst):
    return list(set(lst))
""",
        test_code="""assert sorted(remove_duplicates([1, 2, 2, 3, 3, 3])) == [1, 2, 3]
assert remove_duplicates([]) == []
assert len(remove_duplicates([1, 1, 1])) == 1
print("All tests passed!")
""",
    ),
    EvalCase(
        prompt="Write a function that counts vowels in a string.",
        reference_code="""def count_vowels(s: str) -> int:
    vowels = 'aeiouAEIOU'
    return sum(1 for char in s if char in vowels)
""",
        test_code="""assert count_vowels("hello") == 2
assert count_vowels("xyz") == 0
assert count_vowels("AEIOU") == 5
assert count_vowels("") == 0
print("All tests passed!")
""",
    ),
]


# ── Benchmark Runner ─────────────────────────

def run_benchmark(
    generate_fn: Callable,
    cases: List[EvalCase],
    name: str = "benchmark",
    max_tokens: int = 256,
) -> BenchmarkResult:
    """
    Run a benchmark evaluation using a generation function.

    Args:
        generate_fn: Function that takes (prompt) and returns generated code
        cases: List of test cases
        name: Benchmark name
        max_tokens: Max tokens for generation

    Returns:
        BenchmarkResult with pass/fail statistics
    """
    passed = 0
    total = len(cases)
    times = []
    errors = []
    details = {}

    for i, case in enumerate(cases):
        try:
            start = time.time()
            generated = generate_fn(case.prompt)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)

            # Validate syntax
            syntax_ok = validate_syntax(generated, case.language)
            if not syntax_ok["valid"]:
                errors.append(f"Case {i}: Syntax error — {syntax_ok['error']}")
                details[case.prompt[:50]] = {"passed": False, "error": syntax_ok["error"]}
                continue

            # Run test
            test_result = run_code_test(generated, case.test_code, case.timeout_s)
            if test_result["passed"]:
                passed += 1
                details[case.prompt[:50]] = {"passed": True}
            else:
                errors.append(f"Case {i}: {test_result['error'][:100]}")
                details[case.prompt[:50]] = {"passed": False, "error": test_result["error"]}

        except Exception as e:
            errors.append(f"Case {i}: {str(e)}")
            details[case.prompt[:50]] = {"passed": False, "error": str(e)}

    accuracy = passed / max(total, 1)
    avg_time = sum(times) / max(len(times), 1)

    return BenchmarkResult(
        name=name,
        passed=passed,
        total=total,
        accuracy=accuracy,
        avg_time_ms=avg_time,
        errors=errors[:10],
        details=details,
    )


# ── Code Correctness Scoring ─────────────────

def score_code_correctness(code: str, language: str = "python") -> Dict:
    """
    Heuristic code quality scoring without execution.

    Checks:
      - Syntax validity
      - Function definitions
      - Return statements
      - Documentation presence
      - Error handling patterns
    """
    score = 0.0
    checks = []

    # Syntax
    syntax = validate_syntax(code, language)
    if syntax["valid"]:
        score += 0.2
        checks.append({"check": "syntax", "passed": True, "weight": 0.2})
    else:
        checks.append({"check": "syntax", "passed": False, "error": syntax["error"], "weight": 0.2})

    # Has function/class definitions
    has_def = bool(re.search(r'def |class |function |const \w+ =', code))
    if has_def:
        score += 0.15
        checks.append({"check": "definitions", "passed": True, "weight": 0.15})

    # Has return statements (for functions)
    has_return = bool(re.search(r'return |yield ', code))
    if has_return:
        score += 0.15
        checks.append({"check": "returns", "passed": True, "weight": 0.15})

    # Has docstrings/comments
    has_docs = bool(re.search(r'""".*?"""|\'\'\'.*?\'\'\'|#|//|/\*|\*\/', code))
    if has_docs:
        score += 0.15
        checks.append({"check": "documentation", "passed": True, "weight": 0.15})

    # Has error handling
    has_error_handling = bool(re.search(r'try:|except |if .* is None|if .* is not None|raise ', code))
    if has_error_handling:
        score += 0.15
        checks.append({"check": "error_handling", "passed": True, "weight": 0.15})

    # Type hints (Python) / Type annotations
    has_types = bool(re.search(r': \w+ = |-> \w+:|: int|: str|: list|: dict|: bool', code))
    if has_types:
        score += 0.1
        checks.append({"check": "type_hints", "passed": True, "weight": 0.1})

    return {
        "score": round(score, 3),
        "checks": checks,
        "syntax_ok": syntax["valid"],
        "has_definitions": has_def,
        "has_return": has_return,
        "has_documentation": has_docs,
    }

# ── Benchmark Presets ────────────────────────

BENCHMARK_PRESETS = {
    "humaneval": HUMANEVAL_SAMPLES,
    "mbpp": MBPP_SAMPLES,
    "all": HUMANEVAL_SAMPLES + MBPP_SAMPLES,
}


def get_benchmark(name: str = "all") -> List[EvalCase]:
    """Get a benchmark preset by name."""
    return BENCHMARK_PRESETS.get(name, BENCHMARK_PRESETS["all"])