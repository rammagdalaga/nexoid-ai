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


OWASP_BALANCE_KEYS = [
    "a01_broken_access_control",
    "a02_crypto_failures",
    "a03_injection",
    "a04_insecure_design",
    "a05_security_misconfiguration",
]

CVE_TYPE_KEYS = [
    "cwe-79", "cwe-89", "cwe-287", "cwe-522", "cwe-798"
]


def detect_dataset_imbalance(records: Iterable[SecurityRecord], threshold_ratio: float = 0.35) -> Dict[str, object]:
    """Detect imbalance across categories and metadata classes.

    threshold_ratio: allowed deviation from mean class size.
    """
    recs = list(records)
    category_counts = summarize_security_dataset(recs)
    categories_only = [category_counts[c] for c in SECURITY_CATEGORIES]
    avg = sum(categories_only) / max(len(categories_only), 1)
    warnings = []

    for c in SECURITY_CATEGORIES:
        if avg == 0:
            continue
        ratio = abs(category_counts[c] - avg) / avg
        if ratio > threshold_ratio:
            warnings.append(f"category imbalance: {c} count={category_counts[c]} avg={avg:.2f}")

    owasp_counts = {k: 0 for k in OWASP_BALANCE_KEYS}
    cve_counts = {k: 0 for k in CVE_TYPE_KEYS}
    for r in recs:
        owasp = str(r.metadata.get("owasp", "")).lower()
        cwe = str(r.metadata.get("cwe", "")).lower()
        if owasp in owasp_counts:
            owasp_counts[owasp] += 1
        if cwe in cve_counts:
            cve_counts[cwe] += 1

    return {
        "ok": len(warnings) == 0,
        "warnings": warnings,
        "category_counts": category_counts,
        "owasp_counts": owasp_counts,
        "cve_type_counts": cve_counts,
    }
