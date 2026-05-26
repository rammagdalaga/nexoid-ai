"""ATLAS standardized defensive security response formatter."""

from typing import Dict


VALID_SEVERITIES = {"Low", "Medium", "High", "Critical"}


def format_security_response(
    issue_detection: str,
    explanation: str,
    severity: str,
    risk_reasoning: str,
    fix: str,
) -> Dict[str, str]:
    normalized = severity.capitalize()
    if normalized not in VALID_SEVERITIES:
        normalized = "Medium"
    return {
        "issue_detection": issue_detection.strip(),
        "explanation": explanation.strip(),
        "severity_level": normalized,
        "risk_reasoning": risk_reasoning.strip(),
        "secure_fix_recommendation": fix.strip(),
    }
