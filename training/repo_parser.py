"""
APEXAI — Repository Parser for Full-Stack Intelligence
Teaches ApexAI repository-level understanding:
  - Multi-file training support
  - Dependency graph awareness
  - Project structure learning
  - AST-aware chunking
  - Frontend/backend linking

Supports: React + Vite, FastAPI, Django, Next.js, Node.js, TypeScript
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class ProjectFile:
    """Represents a file in a project."""
    path: str
    language: str
    size_bytes: int
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    ast_type: str = "unknown"  # 'component', 'page', 'api', 'util', 'config'


@dataclass
class ProjectStructure:
    """Represents the full structure of a parsed project."""
    root: str
    framework: str  # 'react-vite', 'fastapi', 'django', 'nextjs', 'node-express'
    files: List[ProjectFile] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    entry_points: List[str] = field(default_factory=list)
    routes: List[Dict] = field(default_factory=list)
    components: List[str] = field(default_factory=list)
    apis: List[str] = field(default_factory=list)


# ── Framework Detection ──────────────────────

FRAMEWORK_PATTERNS = {
    "react-vite": {
        "files": ["vite.config.ts", "vite.config.js"],
        "patterns": [r"from ['\"]react['\"]", r"import React", r"<[A-Z]"],
    },
    "nextjs": {
        "files": ["next.config.js", "next.config.ts"],
        "patterns": [r"from ['\"]next", r"next/link", r"getStaticProps"],
    },
    "fastapi": {
        "files": ["main.py"],
        "patterns": [r"from fastapi", r"@app\.", r"FastAPI\(\)"],
    },
    "django": {
        "files": ["manage.py", "settings.py"],
        "patterns": [r"from django", r"django.urls", r"django.db"],
    },
    "node-express": {
        "files": ["app.js", "app.ts", "server.js", "server.ts"],
        "patterns": [r"require\(['\"]express['\"]", r"from ['\"]express['\"]"],
    },
}


def detect_framework(project_root: str) -> str:
    """Detect the framework used in a project."""
    root_path = Path(project_root)
    scores = {}

    for framework, config in FRAMEWORK_PATTERNS.items():
        score = 0
        # Check for framework-specific files
        for f in config["files"]:
            if (root_path / f).exists():
                score += 2

        # Check for framework patterns in source files
        for py_file in root_path.rglob("*.py" if "django" in framework or "fastapi" in framework else "*.{js,jsx,ts,tsx}"):
            try:
                content = py_file.read_text(encoding="utf-8", errors="ignore")
                for pattern in config["patterns"]:
                    if re.search(pattern, content):
                        score += 1
            except Exception:
                continue

        if score > 0:
            scores[framework] = score

    if not scores:
        return "unknown"
    return max(scores, key=scores.get)


# ── File Type Detection ──────────────────────

LANGUAGE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".sql": "sql",
}

IMPORT_PATTERNS = {
    "python": r"^import\s+(\S+)|^from\s+(\S+)\s+import",
    "javascript": r"(?:import\s+.*?\s+from\s+['\"](.+?)['\"]|require\(['\"](.+?)['\"]\))",
    "typescript": r"(?:import\s+.*?\s+from\s+['\"](.+?)['\"]|require\(['\"](.+?)['\"]\))",
    "jsx": r"(?:import\s+.*?\s+from\s+['\"](.+?)['\"]|require\(['\"](.+?)['\"]\))",
    "tsx": r"(?:import\s+.*?\s+from\s+['\"](.+?)['\"]|require\(['\"](.+?)['\"]\))",
}


def detect_file_type(file_path: str) -> str:
    """Detect programming language from file extension."""
    ext = Path(file_path).suffix.lower()
    return LANGUAGE_EXTENSIONS.get(ext, "unknown")


def extract_imports(content: str, language: str) -> List[str]:
    """Extract import statements from source code."""
    pattern = IMPORT_PATTERNS.get(language, "")
    if not pattern:
        return []

    imports = []
    for match in re.finditer(pattern, content, re.MULTILINE):
        for group in match.groups():
            if group:
                # Clean up the import path
                clean = group.strip().strip("'\"")
                if clean and not clean.startswith("."):
                    imports.append(clean)
    return list(set(imports))


# ── Project Parsing ──────────────────────────

def parse_project(project_root: str, max_files: int = 1000) -> ProjectStructure:
    """
    Parse a full project directory into a structured representation.

    Args:
        project_root: Root directory of the project
        max_files: Maximum number of files to parse

    Returns:
        ProjectStructure with all parsed information
    """
    root_path = Path(project_root)
    if not root_path.exists():
        raise FileNotFoundError(f"Project root not found: {project_root}")

    framework = detect_framework(project_root)
    structure = ProjectStructure(root=project_root, framework=framework)

    # Directories to skip
    skip_dirs = {
        "node_modules", ".git", "__pycache__", "venv", "env",
        ".venv", "dist", "build", ".next", "coverage", ".cache",
    }

    # Collect all project files
    file_count = 0
    for file_path in root_path.rglob("*"):
        if file_count >= max_files:
            break

        # Skip directories and unwanted paths
        if file_path.is_dir():
            continue
        if any(part in skip_dirs for part in file_path.parts):
            continue

        rel_path = str(file_path.relative_to(root_path))
        language = detect_file_type(str(file_path))

        if language == "unknown":
            continue

        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        project_file = ProjectFile(
            path=rel_path,
            language=language,
            size_bytes=file_path.stat().st_size,
            imports=extract_imports(content, language),
        )

        # Determine AST type based on file patterns
        project_file.ast_type = classify_ast_type(rel_path, content, language)
        structure.files.append(project_file)

        # Track entry points
        if rel_path in ["main.py", "app.py", "index.js", "index.tsx",
                        "src/index.js", "src/index.tsx", "src/main.tsx"]:
            structure.entry_points.append(rel_path)

        # Track components (React/JSX/TSX)
        if language in ["jsx", "tsx"]:
            component_match = re.search(r'(?:export\s+default\s+)?(?:function|const)\s+(\w+)', content)
            if component_match:
                structure.components.append(component_match.group(1))

        # Track API routes
        if language == "python":
            if re.search(r"@app\.(?:get|post|put|delete|patch)\(['\"]", content):
                structure.apis.append(rel_path)
        elif language in ["javascript", "typescript"]:
            if re.search(r"router\.(?:get|post|put|delete|patch)\(['\"]", content):
                structure.apis.append(rel_path)

        file_count += 1

    # Build dependency graph
    structure.dependencies = build_dependency_graph(structure.files)

    return structure


def classify_ast_type(file_path: str, content: str, language: str) -> str:
    """Classify a file's role in the project."""
    path_lower = file_path.lower()

    if "component" in path_lower or (language in ["jsx", "tsx"] and
       re.search(r'(?:export\s+default\s+)?(?:function|const)\s+\w+\s*(?:<[^>]+>)?\s*(?:\(|\s*=>)', content)):
        return "component"

    if "page" in path_lower or "route" in path_lower:
        return "page"

    if "api" in path_lower or "handler" in path_lower:
        return "api"

    if "util" in path_lower or "helper" in path_lower:
        return "util"

    if re.search(r"(config|setting|environment)", path_lower):
        return "config"

    return "module"


def build_dependency_graph(files: List[ProjectFile]) -> Dict[str, List[str]]:
    """Build a dependency graph from parsed files."""
    graph = {}
    for f in files:
        graph[f.path] = []
        for imp in f.imports:
            # Map imports to project files
            for other in files:
                if other.path.startswith(imp.replace(".", "/")) or \
                   other.path.replace("/", ".").replace("." + other.language, "") == imp:
                    graph[f.path].append(other.path)
    return graph


# ── Cross-File Context Generation ────────────

def generate_cross_file_context(structure: ProjectStructure) -> str:
    """
    Generate a condensed cross-file context for model training.
    This helps the model understand repository-level structure.
    """
    lines = [
        f"Project Framework: {structure.framework}",
        f"Total Files: {len(structure.files)}",
        f"Entry Points: {', '.join(structure.entry_points) if structure.entry_points else 'None'}",
        f"Components: {', '.join(structure.components[:20])}",
        f"API Routes: {', '.join(structure.apis[:20])}",
        "",
        "Dependency Graph (top-level):",
    ]

    # Show top-level dependency relationships
    for file_path, deps in structure.dependencies.items():
        if deps:
            parent = Path(file_path).parts[0] if Path(file_path).parts else file_path
            child_dirs = set()
            for d in deps:
                child_parts = Path(d).parts
                if child_parts:
                    child_dirs.add(child_parts[0])
            lines.append(f"  {parent} → {', '.join(sorted(child_dirs))}")

    return "\n".join(lines)


# ── AST-Aware Chunking ───────────────────────

def chunk_by_ast(code: str, language: str, max_chunk_size: int = 4096) -> List[Dict]:
    """
    Split code into semantically meaningful chunks based on AST structure.

    Args:
        code: Source code
        language: Programming language
        max_chunk_size: Maximum characters per chunk

    Returns:
        List of dicts with 'content', 'type', and 'name' keys
    """
    chunks = []

    # Function/class detection patterns
    pattern_map = {
        "python": r"^(?:def |class |async def )(\w+)",
        "javascript": r"^(?:function |const |let |var |class |export )(\w+)",
        "typescript": r"^(?:function |const |let |var |class |export |interface |type )(\w+)",
        "jsx": r"^(?:function |const |let |class |export )(\w+)",
        "tsx": r"^(?:function |const |let |class |export |interface |type )(\w+)",
    }

    pattern = pattern_map.get(language, pattern_map["python"])
    lines = code.split("\n")

    current_chunk = []
    current_lines = 0
    chunk_name = "header"

    for line in lines:
        match = re.match(pattern, line.strip())
        if match:
            # Save previous chunk
            if current_chunk:
                chunks.append({
                    "content": "\n".join(current_chunk),
                    "type": "section",
                    "name": chunk_name,
                })
            current_chunk = []
            current_lines = 0
            chunk_name = match.group(1)

        current_chunk.append(line)
        current_lines += 1

        # Enforce max chunk size
        if current_lines >= max_chunk_size // 80:  # Rough line estimate
            chunks.append({
                "content": "\n".join(current_chunk),
                "type": "section",
                "name": chunk_name,
            })
            current_chunk = []
            current_lines = 0

    # Final chunk
    if current_chunk:
        chunks.append({
            "content": "\n".join(current_chunk),
            "type": "section",
            "name": chunk_name,
        })

    return chunks