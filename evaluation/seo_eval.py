"""
APEXAI — SEO Evaluation Suite
Evaluates model's ability to generate SEO-optimized code.
"""

import re
from evaluation.benchmarks import EvalCase, BenchmarkResult
from training.seo_processor import score_seo_quality, analyze_react_seo
from typing import Dict, List, Callable


# ── SEO Test Cases ───────────────────────────

SEO_TEST_CASES = [
    EvalCase(
        prompt="Generate an SEO-optimized HTML head with meta tags, Open Graph, Twitter Cards, and canonical URL.",
        reference_code="""<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="description" content="Page description" />
    <meta name="robots" content="index, follow" />
    <meta property="og:title" content="Title" />
    <meta property="og:description" content="Desc" />
    <meta name="twitter:card" content="summary" />
    <link rel="canonical" href="https://example.com/" />
    <title>Page Title</title>
</head>""",
        test_code="",
        language="html",
    ),
    EvalCase(
        prompt="Create JSON-LD structured data for a website with organization schema.",
        reference_code="""<script type="application/ld+json">
{
    "@context": "https://schema.org",
    "@type": "Organization",
    "name": "Company Name",
    "url": "https://example.com"
}
</script>""",
        test_code="",
        language="html",
    ),
    EvalCase(
        prompt="Create an accessible form with proper ARIA labels and semantic HTML.",
        reference_code="""<form aria-label="Contact form">
    <label for="name">Name</label>
    <input id="name" type="text" aria-required="true" />
    <label for="email">Email</label>
    <input id="email" type="email" aria-required="true" />
    <button type="submit" aria-label="Submit form">Send</button>
</form>""",
        test_code="",
        language="html",
    ),
    EvalCase(
        prompt="Write a React component with SEO-friendly semantic structure.",
        reference_code="""function HomePage() {
    return (
        <>
            <Helmet>
                <title>Home | Site Name</title>
                <meta name="description" content="Description" />
                <meta property="og:title" content="Home" />
            </Helmet>
            <header>
                <nav aria-label="Main">
                    <a href="/">Home</a>
                </nav>
            </header>
            <main>
                <h1>Welcome</h1>
                <article>
                    <h2>Section Title</h2>
                    <p>Content</p>
                </article>
            </main>
            <footer>
                <p>&copy; 2024</p>
            </footer>
        </>
    );
}""",
        test_code="",
        language="tsx",
    ),
    EvalCase(
        prompt="Optimize images with lazy loading, WebP format, and responsive srcset.",
        reference_code="""<picture>
    <source media="(min-width: 1024px)" srcset="large.webp" type="image/webp" />
    <source media="(min-width: 768px)" srcset="medium.webp" type="image/webp" />
    <img src="fallback.jpg" srcset="small.jpg 480w, medium.jpg 768w" sizes="100vw"
         alt="Description" loading="lazy" decoding="async" width="800" height="600" />
</picture>""",
        test_code="",
        language="html",
    ),
]


def run_seo_benchmark(generate_fn: Callable, name: str = "seo") -> BenchmarkResult:
    """
    Evaluate model's SEO generation capabilities.

    Args:
        generate_fn: Function that takes (prompt) and returns generated code
        name: Benchmark name

    Returns:
        BenchmarkResult with SEO quality scores
    """
    passed = 0
    total = len(SEO_TEST_CASES)
    errors = []
    details = {}

    for i, case in enumerate(SEO_TEST_CASES):
        try:
            generated = generate_fn(case.prompt)

            # Score SEO quality
            seo_score = score_seo_quality(generated, case.language)
            html_score = seo_score["total"]

            # For React/TSX, also analyze React-specific SEO
            react_score = None
            if case.language in ("jsx", "tsx"):
                react_analysis = analyze_react_seo(generated)
                react_score = react_analysis
                if react_analysis["has_helmet"] and react_analysis["has_meta_tags"]:
                    html_score = min(html_score + 0.2, 1.0)

            # Pass threshold: SEO score >= 0.5
            case_passed = html_score >= 0.5
            if case_passed:
                passed += 1

            details[f"Case {i}"] = {
                "passed": case_passed,
                "seo_score": html_score,
                "grade": seo_score["grade"],
                "categories": seo_score["categories"],
                "react_analysis": react_score,
            }

            if not case_passed:
                errors.append(f"Case {i}: SEO score {html_score:.2f} (below 0.5 threshold)")

        except Exception as e:
            errors.append(f"Case {i}: {str(e)}")
            details[f"Case {i}"] = {"passed": False, "error": str(e)}

    accuracy = passed / max(total, 1)

    return BenchmarkResult(
        name=name,
        passed=passed,
        total=total,
        accuracy=accuracy,
        avg_time_ms=0,
        errors=errors[:10],
        details=details,
    )