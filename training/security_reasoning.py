"""ATLAS Phase 1 Security Reasoning Engine.

Provides a deterministic, defensive-only reasoning pipeline:
1) Detection
2) Classification
3) Risk Analysis
4) Secure Fix Recommendation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


STAGES = ("detection", "classification", "risk_analysis", "secure_fix_recommendation")

CATEGORY_PATTERNS = {
    "injection_risks": ["select *", "exec(", "eval(", "os.system(", "subprocess"],
    "auth_weaknesses": ["password ==", "hardcoded_password", "jwt.decode(", "md5(", "sha1("],
    "insecure_storage": ["localstorage", "pickle.loads", "private_key", "secret =", "plaintext"],
    "unsafe_apis": ["verify=False", "allow_origins=['*']", "debug=True", "yaml.load("],
    "dependency_risks": ["requirements.txt", "package-lock.json", "pip install", "npm install"],
}


@dataclass
class SecurityReasoningResult:
    issue_detection: str
    explanation: str
    severity_level: str
    risk_reasoning: str
    secure_fix_recommendation: str
    category: str
    stage_outputs: Dict[str, str]


def _detect_category(code: str) -> tuple[str, List[str]]:
    text = code.lower()
    for category, patterns in CATEGORY_PATTERNS.items():
        hits = [p for p in patterns if p.lower() in text]
        if hits:
            return category, hits
    return "secure_coding", []


def _severity_for(category: str, hits: List[str]) -> str:
    if category in {"injection_risks", "auth_weaknesses"} and len(hits) >= 2:
        return "High"
    if category in {"injection_risks", "auth_weaknesses", "unsafe_apis"} and hits:
        return "Medium"
    if category in {"insecure_storage", "dependency_risks"} and hits:
        return "Medium"
    return "Low"


def analyze_security_reasoning(code: str) -> SecurityReasoningResult:
    """Run defensive multi-stage reasoning on code text without execution."""
    category, hits = _detect_category(code)

    detection = (
        f"Detected potential {category.replace('_', ' ')} indicators: {', '.join(hits)}"
        if hits else "No explicit vulnerability indicator pattern detected."
    )
    classification = category
    severity = _severity_for(category, hits)

    risk_reasoning = {
        "injection_risks": "Untrusted input paths can alter queries/commands and break integrity.",
        "auth_weaknesses": "Weak credential/token validation can enable unauthorized access.",
        "insecure_storage": "Sensitive data exposure increases confidentiality and compliance risk.",
        "unsafe_apis": "Unsafe runtime/API flags reduce transport or execution safety guarantees.",
        "dependency_risks": "Unpinned/unreviewed dependencies can import known vulnerable components.",
        "secure_coding": "Current snippet does not match known insecure signatures but still needs review.",
    }[category]

    secure_fix = {
        "injection_risks": "Use parameterized queries, strict input validation, and context-aware escaping.",
        "auth_weaknesses": "Enforce strong hashing, secure token validation, and least-privilege auth checks.",
        "insecure_storage": "Encrypt sensitive data at rest, remove hardcoded secrets, and use secret managers.",
        "unsafe_apis": "Disable unsafe flags, enforce TLS verification, and restrict debug/broad CORS settings.",
        "dependency_risks": "Pin versions, run SCA scans, and patch dependencies with known CVEs.",
        "secure_coding": "Continue static/dynamic security review and add policy-based validation tests.",
    }[category]

    stage_outputs = {
        "detection": detection,
        "classification": classification,
        "risk_analysis": risk_reasoning,
        "secure_fix_recommendation": secure_fix,
    }

    return SecurityReasoningResult(
        issue_detection=detection,
        explanation=f"Pattern-based classification mapped to {classification}.",
        severity_level=severity,
        risk_reasoning=risk_reasoning,
        secure_fix_recommendation=secure_fix,
        category=category,
        stage_outputs=stage_outputs,
    )
