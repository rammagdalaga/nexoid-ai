"""
APEXAI — HumanEval Evaluation Runner
Runs the full HumanEval benchmark against the model.
"""

from evaluation.benchmarks import (
    EvalCase, BenchmarkResult, run_code_test, validate_syntax
)
from typing import Dict, List, Callable


# ── HumanEval-style Problems ─────────────────

HUMANEVAL_PROBLEMS = [
    {
        "task_id": "HumanEval/0",
        "prompt": "Write a function that returns the sum of a and b.",
        "entry_point": "add",
        "test": "assert add(1, 2) == 3\nassert add(-1, 1) == 0",
    },
    {
        "task_id": "HumanEval/1",
        "prompt": "Write a function that checks if a number is even.",
        "entry_point": "is_even",
        "test": "assert is_even(2) == True\nassert is_even(3) == False\nassert is_even(0) == True",
    },
    {
        "task_id": "HumanEval/2",
        "prompt": "Write a function that returns the length of a string.",
        "entry_point": "string_length",
        "test": "assert string_length('hello') == 5\nassert string_length('') == 0",
    },
    {
        "task_id": "HumanEval/3",
        "prompt": "Write a function that returns the largest number in a list.",
        "entry_point": "largest",
        "test": "assert largest([1, 2, 3, 4, 5]) == 5\nassert largest([-1, -2, -3]) == -1",
    },
    {
        "task_id": "HumanEval/4",
        "prompt": "Write a function that reverses a string.",
        "entry_point": "reverse",
        "test": "assert reverse('hello') == 'olleh'\nassert reverse('') == ''",
    },
]


def create_eval_cases() -> List[EvalCase]:
    """Create EvalCase objects from HumanEval problems."""
    cases = []
    for problem in HUMANEVAL_PROBLEMS:
        cases.append(EvalCase(
            prompt=problem["prompt"],
            reference_code=problem.get("canonical_solution", ""),
            test_code=problem["test"],
            language="python",
        ))
    return cases


def run_humaneval(generate_fn: Callable, name: str = "humaneval") -> BenchmarkResult:
    """
    Run full HumanEval benchmark.

    Args:
        generate_fn: Function that takes (prompt) and returns generated code
        name: Benchmark name

    Returns:
        BenchmarkResult with pass@k metrics
    """
    from evaluation.benchmarks import run_benchmark
    cases = create_eval_cases()
    return run_benchmark(generate_fn, cases, name=name)


def compute_pass_at_k(results: List[bool], k: int = 1) -> float:
    """
    Compute pass@k metric.

    pass@k = 1 - (C(n - correct, k) / C(n, k))
    where n is total samples, correct is number passing.
    """
    from math import comb

    n = len(results)
    correct = sum(1 for r in results if r)

    if n == 0:
        return 0.0

    if n - correct < k:
        return 1.0

    return 1.0 - comb(n - correct, k) / comb(n, k)