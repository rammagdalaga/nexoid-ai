"""ATLAS standardized defensive security response formatter."""

from typing import Dict


VALID_SEVERITIES = {"Low", "Medium", "High", "Critical"}


def format_security_response(issue: str, explanation: str, severity: str, fix: str) -> Dict[str, str]:
    normalized = severity.capitalize()
    if normalized not in VALID_SEVERITIES:
        normalized = "Medium"
    return {
        "vulnerability_or_issue": issue.strip(),
        "explanation": explanation.strip(),
        "severity_level": normalized,
        "secure_fix_or_recommendation": fix.strip(),
    }
