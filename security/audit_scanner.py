import json
import re
from pathlib import Path
from typing import Dict, List

SECRET_PATTERNS = [
    ("critical", re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*=\s*['\"][^'\"]{8,}['\"]")),
    ("high", re.compile(r"(?i)bearer\s+[a-z0-9\-\._~\+\/]+=*")),
]
UNSAFE_IMPORTS = {
    "pickle": "medium",
    "subprocess": "low",
}


def scan_repo(root: str = ".") -> Dict:
    findings: List[Dict] = []
    base = Path(root)
    for path in base.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        txt = path.read_text(encoding="utf-8", errors="ignore")
        for severity, pat in SECRET_PATTERNS:
            for m in pat.finditer(txt):
                findings.append({
                    "file": str(path),
                    "severity": severity,
                    "type": "hardcoded_secret",
                    "match": m.group(0)[:120],
                    "fix": "Move secret material to environment variables and rotate credentials.",
                })
        for mod, severity in UNSAFE_IMPORTS.items():
            if re.search(rf"(^|\n)\s*(import|from)\s+{re.escape(mod)}\b", txt):
                findings.append({
                    "file": str(path),
                    "severity": severity,
                    "type": "unsafe_import",
                    "match": mod,
                    "fix": f"Review necessity of '{mod}' and constrain untrusted input paths.",
                })

    server_path = base / "inference" / "server.py"
    if server_path.exists():
        s = server_path.read_text(encoding="utf-8", errors="ignore")
        for endpoint in ["/v1/completions", "/v1/chat/completions", "/v1/generate"]:
            if endpoint in s and "validate_endpoint_schema" not in s:
                findings.append({
                    "file": str(server_path),
                    "severity": "high",
                    "type": "missing_validation_endpoint",
                    "match": endpoint,
                    "fix": "Apply strict endpoint schema validation before handler execution.",
                })

    return {
        "summary": {
            "total_findings": len(findings),
            "by_severity": {
                "critical": sum(1 for f in findings if f["severity"] == "critical"),
                "high": sum(1 for f in findings if f["severity"] == "high"),
                "medium": sum(1 for f in findings if f["severity"] == "medium"),
                "low": sum(1 for f in findings if f["severity"] == "low"),
            },
        },
        "findings": findings,
    }


if __name__ == "__main__":
    report = scan_repo(".")
    print(json.dumps(report, indent=2))
