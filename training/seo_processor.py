"""
APEXAI — SEO Intelligence Processor
SEO-specialized training capabilities for web development.

Teaches ApexAI to generate:
  - SEO-friendly React apps
  - Optimized metadata
  - Accessible HTML
  - Lighthouse-aware structures
  - Fast-loading frontend code

TRAINING RULES:
  - SEO datasets are streamed from cloud, never cached locally without permission
  - SEO token specializations are added to tokenizer as special tokens
  - Scoring is heuristic-based for training signal
"""

import re
import json
from typing import Dict, List, Optional, Tuple


# ── SEO Configuration ────────────────────────

SEO_SCORING_RULES = {
    "meta_tags": {
        "patterns": [
            r'<meta\s+name="description"',
            r'<meta\s+name="viewport"',
            r'<meta\s+name="robots"',
            r'<meta\s+property="og:',
            r'<meta\s+name="twitter:',
        ],
        "weight": 0.25,
        "description": "Essential meta tags for search visibility",
    },
    "structured_data": {
        "patterns": [
            r'application/ld\+json',
            r'schema\.org',
            r'"@context"',
            r'"@type"',
        ],
        "weight": 0.20,
        "description": "Schema.org structured data markup",
    },
    "semantic_html": {
        "patterns": [
            r'<header>', r'<nav>', r'<main>', r'<article>',
            r'<section>', r'<aside>', r'<footer>',
            r'<h1>', r'<h2>', r'<h3>',
        ],
        "weight": 0.20,
        "description": "Semantic HTML5 elements for accessibility and SEO",
    },
    "performance": {
        "patterns": [
            r'loading="lazy"',
            r'async',
            r'defer',
            r'rel="preload"',
            r'rel="preconnect"',
            r'media="(min-width|max-width)',
            r'srcset',
            r'sizes=',
        ],
        "weight": 0.15,
        "description": "Performance optimization for Core Web Vitals",
    },
    "accessibility": {
        "patterns": [
            r'aria-',
            r'role="',
            r'alt=',
            r'tabindex',
            r'<label',
            r'aria-label',
        ],
        "weight": 0.10,
        "description": "Accessibility (a11y) best practices",
    },
    "canonical": {
        "patterns": [
            r'rel="canonical"',
            r'rel="alternate"',
            r'hreflang',
        ],
        "weight": 0.10,
        "description": "Canonical URLs and language alternatives",
    },
}


# ── SEO Scoring ──────────────────────────────

def score_seo_quality(code: str, language: str = "html") -> Dict:
    """
    Score code quality from an SEO perspective.

    Args:
        code: Source code to evaluate
        language: Programming language ('html', 'jsx', 'tsx', 'javascript')

    Returns:
        Dict with scores per category and total
    """
    scores = {}
    total_weighted = 0.0
    total_weight = 0.0

    for category, config in SEO_SCORING_RULES.items():
        matches = 0
        for pattern in config["patterns"]:
            if re.search(pattern, code, re.IGNORECASE):
                matches += 1

        # Normalize score: cap at 1.0 based on expected matches
        expected_matches = len(config["patterns"])
        category_score = min(matches / max(expected_matches, 1), 1.0)

        scores[category] = {
            "score": round(category_score, 3),
            "matches": matches,
            "expected": expected_matches,
        }
        total_weighted += category_score * config["weight"]
        total_weight += config["weight"]

    total = round(total_weighted / max(total_weight, 0.01), 3)

    return {
        "total": total,
        "categories": scores,
        "grade": _seo_grade(total),
    }


def _seo_grade(score: float) -> str:
    """Convert numeric SEO score to letter grade."""
    if score >= 0.9:
        return "A"
    elif score >= 0.75:
        return "B"
    elif score >= 0.5:
        return "C"
    elif score >= 0.25:
        return "D"
    return "F"


# ── SEO Token Embeddings ─────────────────────

SEO_TOKEN_MAP = {
    "<|seo|>": "General SEO context marker",
    "<|meta|>": "Metadata optimization section",
    "<|schema|>": "Structured data / Schema.org markup",
    "<|accessibility|>": "Accessibility (a11y) best practices",
    "<|lighthouse|>": "Lighthouse performance optimization",
}


def get_seo_tokens() -> List[str]:
    """Get list of SEO special tokens."""
    return list(SEO_TOKEN_MAP.keys())


# ── SEO Training Data Generation ─────────────

def generate_seo_training_pairs() -> List[Tuple[str, str]]:
    """
    Generate SEO training pairs (prompt, optimal_code) for fine-tuning.

    These are lightweight templates — actual training data streamed from cloud.
    Returns list of (prompt, correct_answer) tuples.
    """
    pairs = [
        (
            "Generate SEO-optimized HTML head section with meta tags",
            """<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="description" content="[Page description for search results]" />
    <meta name="robots" content="index, follow" />
    <meta property="og:title" content="[Page Title]" />
    <meta property="og:description" content="[Description]" />
    <meta property="og:type" content="website" />
    <meta name="twitter:card" content="summary_large_image" />
    <link rel="canonical" href="https://example.com/" />
    <title>[Page Title]</title>
</head>"""
        ),
        (
            "Add structured data (JSON-LD) for a local business",
            """<script type="application/ld+json">
{
    "@context": "https://schema.org",
    "@type": "LocalBusiness",
    "name": "[Business Name]",
    "description": "[Description]",
    "url": "https://example.com",
    "telephone": "+1-555-555-5555",
    "address": {
        "@type": "PostalAddress",
        "streetAddress": "[Street]",
        "addressLocality": "[City]",
        "addressRegion": "[State]",
        "postalCode": "[Zip]"
    }
}
</script>"""
        ),
        (
            "Optimize images with lazy loading and responsive srcset",
            """<picture>
    <source media="(min-width: 1024px)" srcset="image-lg.webp" type="image/webp" />
    <source media="(min-width: 768px)" srcset="image-md.webp" type="image/webp" />
    <img
        src="image-fallback.jpg"
        srcset="image-sm.jpg 480w, image-md.jpg 768w, image-lg.jpg 1024w"
        sizes="(max-width: 480px) 100vw, (max-width: 768px) 50vw, 33vw"
        alt="Descriptive alt text for accessibility"
        loading="lazy"
        decoding="async"
        width="800"
        height="600"
    />
</picture>"""
        ),
        (
            "Create an accessible navigation with ARIA labels",
            """<nav aria-label="Main navigation">
    <ul role="list">
        <li><a href="/" aria-current="page">Home</a></li>
        <li><a href="/about">About</a></li>
        <li role="none">
            <button aria-haspopup="true" aria-expanded="false">
                Services
            </button>
            <ul role="menu" aria-label="Services submenu">
                <li role="none"><a href="/services/web" role="menuitem">Web</a></li>
                <li role="none"><a href="/services/mobile" role="menuitem">Mobile</a></li>
            </ul>
        </li>
    </ul>
</nav>"""
        ),
    ]
    return pairs


# ── React/TypeScript SEO Analysis ────────────

def analyze_react_seo(jsx_code: str) -> Dict:
    """
    Analyze React/JSX/TSX code for SEO best practices.

    Checks for:
      - Proper <Helmet> or <head> management
      - Semantic HTML usage in JSX
      - Accessibility attributes
      - Performance patterns
    """
    results = {
        "has_helmet": bool(re.search(r'<Helmet', jsx_code)),
        "has_meta_tags": bool(re.search(r'<meta\s', jsx_code)),
        "semantic_elements": [],
        "a11y_issues": [],
        "performance_score": 0.0,
    }

    # Check for semantic elements in JSX
    semantic = ["header", "nav", "main", "article", "section", "aside", "footer"]
    for elem in semantic:
        if re.search(rf'<{elem}[>\s]', jsx_code, re.IGNORECASE):
            results["semantic_elements"].append(elem)

    # Check for common a11y issues
    if not re.search(r'aria-', jsx_code):
        results["a11y_issues"].append("Missing ARIA attributes")
    if not re.search(r'alt=', jsx_code):
        results["a11y_issues"].append("Missing image alt text")
    if not re.search(r'<label', jsx_code):
        results["a11y_issues"].append("Missing form labels")

    # Performance score
    perf_score = 0.0
    if re.search(r'lazy|loading=', jsx_code):
        perf_score += 0.3
    if re.search(r'async|defer', jsx_code):
        perf_score += 0.3
    if re.search(r'rel="preload"|rel="preconnect"', jsx_code):
        perf_score += 0.2
    if re.search(r'srcset|sizes=', jsx_code):
        perf_score += 0.2
    results["performance_score"] = round(perf_score, 2)

    return results


# ── Lighthouse-Aware Generation Hints ────────

def get_lighthouse_hints(score_category: str = "all") -> List[str]:
    """
    Get generation hints for Lighthouse optimization.

    Args:
        score_category: 'seo', 'performance', 'accessibility', or 'all'

    Returns:
        List of hints to include in training signal
    """
    hints = {
        "seo": [
            "Include descriptive meta description (under 160 chars)",
            "Add Open Graph and Twitter Card meta tags",
            "Use semantic HTML5 elements",
            "Include structured data (JSON-LD)",
            "Set canonical URL",
            "Create descriptive, unique title tags",
            "Use descriptive heading hierarchy (h1 → h6)",
        ],
        "performance": [
            "Enable lazy loading for images with loading='lazy'",
            "Use async/defer for script loading",
            "Preload critical resources with rel='preload'",
            "Preconnect to third-party origins",
            "Use responsive images with srcset and sizes",
            "Optimize image formats (WebP, AVIF)",
            "Minimize render-blocking resources",
        ],
        "accessibility": [
            "Use ARIA landmarks (role, aria-label, aria-labelledby)",
            "Provide alt text for all images",
            "Ensure proper heading hierarchy",
            "Use descriptive link text",
            "Maintain color contrast ratios",
            "Support keyboard navigation (tabindex)",
            "Use form labels and error announcements",
        ],
    }

    if score_category == "all":
        return hints["seo"] + hints["performance"] + hints["accessibility"]
    return hints.get(score_category, [])