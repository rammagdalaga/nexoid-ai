"""Security-focused dataset utilities for ATLAS Phase 1.

This module builds a balanced, filtered corpus for defensive security modeling.
It supports secure/insecure code pairs, OWASP defensive snippets, CVE-style
reports, and patched-vs-unpatched comparisons.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


SECURITY_CATEGORIES = [
    "secure_coding",
    "insecure_secure_pairs",
    "owasp_top10_defensive",
    "cve_reports",
    "patched_unpatched",
]

UNSAFE_MARKERS = (
    "weaponize",
    "exploit chain",
    "ransomware",
    "privilege escalation guide",
)


@dataclass
class SecurityRecord:
    category: str
    content: str
    metadata: Dict[str, str]


def is_defensive_record(text: str) -> bool:
    lowered = text.lower()
    return not any(marker in lowered for marker in UNSAFE_MARKERS)


def load_security_records(path: str | Path) -> List[SecurityRecord]:
    """Load JSONL records and keep only defensive entries."""
    src = Path(path)
    records: List[SecurityRecord] = []
    if not src.exists():
        return records

    with src.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            content = raw.get("content", "")
            category = raw.get("category", "secure_coding")
            if category not in SECURITY_CATEGORIES or not is_defensive_record(content):
                continue
            records.append(
                SecurityRecord(
                    category=category,
                    content=content,
                    metadata=raw.get("metadata", {}),
                )
            )
    return records


def balanced_security_sample(records: Iterable[SecurityRecord], per_category: int) -> List[SecurityRecord]:
    """Return balanced records sampled by category, preserving deterministic order."""
    grouped: Dict[str, List[SecurityRecord]] = {k: [] for k in SECURITY_CATEGORIES}
    for record in records:
        grouped.setdefault(record.category, []).append(record)

    sampled: List[SecurityRecord] = []
    for category in SECURITY_CATEGORIES:
        sampled.extend(grouped.get(category, [])[:per_category])
    return sampled


def summarize_security_dataset(records: Iterable[SecurityRecord]) -> Dict[str, int]:
    counts: Dict[str, int] = {k: 0 for k in SECURITY_CATEGORIES}
    for record in records:
        if record.category in counts:
            counts[record.category] += 1
    counts["total"] = sum(counts.values())
    return counts
