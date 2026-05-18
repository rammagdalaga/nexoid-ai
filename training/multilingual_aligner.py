"""
APEXAI — Multilingual Language Alignment
Expands ApexAI beyond code-only intelligence into multilingual understanding.

Languages supported:
  - English (en)
  - Filipino/Tagalog (fil)
  - Japanese (ja)
  - Korean (ko)
  - Chinese (zh)
  - Spanish (es)

Features:
  - Language prefix tokens for training
  - Multilingual code comment preservation
  - Translation-aware training pairs
  - Language detection from code/text
"""

import re
from typing import Dict, List, Optional, Tuple


# ── Language Configuration ───────────────────

LANGUAGE_CONFIG = {
    "en": {
        "name": "English",
        "prefix": "<|en|>",
        "script": "latin",
        "code_comments": r"#(.+?)$|//(.+?)$|/\*[\s\S]*?\*/|'''[\s\S]*?'''",
        "stop_words": ["the", "is", "at", "which", "on", "a", "an", "for", "of", "by"],
    },
    "fil": {
        "name": "Filipino/Tagalog",
        "prefix": "<|fil|>",
        "script": "latin",
        "code_comments": r"#(.+?)$|//(.+?)$|/\*[\s\S]*?\*/",
        "stop_words": ["ang", "ay", "ng", "sa", "at", "ito", "ko", "na", "siya", "may"],
    },
    "ja": {
        "name": "Japanese",
        "prefix": "<|ja|>",
        "script": "cjk",
        "code_comments": r"#(.+?)$|//(.+?)$|/\*[\s\S]*?\*/|<!--[\s\S]*?-->",
        "stop_words": ["の", "を", "は", "に", "が", "で", "と", "も", "た", "する"],
    },
    "ko": {
        "name": "Korean",
        "prefix": "<|ko|>",
        "script": "hangul",
        "code_comments": r"#(.+?)$|//(.+?)$|/\*[\s\S]*?\*/",
        "stop_words": ["이", "그", "저", "것", "수", "있", "하", "것", "들", "의"],
    },
    "zh": {
        "name": "Chinese",
        "prefix": "<|zh|>",
        "script": "cjk",
        "code_comments": r"#(.+?)$|//(.+?)$|/\*[\s\S]*?\*/|<!--[\s\S]*?-->",
        "stop_words": ["的", "了", "在", "是", "我", "有", "和", "就", "不", "人"],
    },
    "es": {
        "name": "Spanish",
        "prefix": "<|es|>",
        "script": "latin",
        "code_comments": r"#(.+?)$|//(.+?)$|/\*[\s\S]*?\*/",
        "stop_words": ["el", "la", "los", "las", "de", "que", "y", "en", "un", "una"],
    },
}


def get_language_prefix(lang: str) -> str:
    """Get the language prefix token for a language code."""
    return LANGUAGE_CONFIG.get(lang, {}).get("prefix", "<|en|>")


# ── Language Detection ───────────────────────

def detect_language(text: str) -> str:
    """
    Detect human language from text using character script analysis.

    Returns BCP-47 language code.
    """
    if not text.strip():
        return "en"

    # Count characters by Unicode script
    cjk_count = len(re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))
    hangul_count = len(re.findall(r'[\uac00-\ud7af]', text))
    hiragana = len(re.findall(r'[\u3040-\u309f]', text))
    katakana = len(re.findall(r'[\u30a0-\u30ff]', text))
    latin_count = len(re.findall(r'[a-zA-Z]', text))

    total = max(len(text.strip()), 1)

    # CJK characters
    cjk_ratio = cjk_count / total
    hangul_ratio = hangul_count / total
    jp_ratio = (hiragana + katakana) / total

    if jp_ratio > 0.05 or (cjk_ratio > 0.1 and jp_ratio > 0.02):
        return "ja"
    if hangul_ratio > 0.05:
        return "ko"
    if cjk_ratio > 0.1:
        return "zh"

    # Latin script — check for Spanish/Filipino markers
    if latin_count > 0:
        words = text.lower().split()
        es_markers = ["el", "la", "los", "las", "que", "de", "y", "en", "por", "para"]
        fil_markers = ["ang", "ay", "ng", "sa", "at", "ito", "ko", "na", "siya"]
        es_score = sum(1 for w in words if w in es_markers)
        fil_score = sum(1 for w in words if w in fil_markers)
        if es_score > fil_score and es_score > 2:
            return "es"
        if fil_score > es_score and fil_score > 2:
            return "fil"

    return "en"


# ── Multilingual Training Data ───────────────

def create_multilingual_prompt(code: str, language: str, instruction: str) -> str:
    """
    Create a multilingual training prompt with language prefix.
    Used for instruction tuning across languages.

    Format:
        <|lang|>instruction
        {code}
    """
    prefix = get_language_prefix(language)
    return f"{prefix}\n{instruction}\n```\n{code}\n```"


MULTILINGUAL_TEMPLATES = {
    "explain_code": {
        "en": "Explain the following code in detail:",
        "fil": "Ipaliwanag ang sumusunod na code nang detalyado:",
        "ja": "次のコードを詳しく説明してください：",
        "ko": "다음 코드를 자세히 설명해주세요:",
        "zh": "请详细解释以下代码：",
        "es": "Explica el siguiente código en detalle:",
    },
    "add_comments": {
        "en": "Add detailed comments to this code in English:",
        "fil": "Magdagdag ng mga komentaryo sa code na ito sa Filipino:",
        "ja": "このコードに日本語でコメントを追加してください：",
        "ko": "이 코드에 한국어로 주석을 추가해주세요:",
        "zh": "请用中文为这段代码添加注释：",
        "es": "Añade comentarios detallados a este código en español:",
    },
    "translate_comments": {
        "en": "Translate the code comments to English:",
        "fil": "Isalin ang mga komentaryo ng code sa Filipino:",
        "ja": "コードのコメントを日本語に翻訳してください：",
        "ko": "코드 주석을 한국어로 번역해주세요:",
        "zh": "将代码注释翻译成中文：",
        "es": "Traduce los comentarios del código al español:",
    },
    "generate_code": {
        "en": "Write code that accomplishes the following in English:",
        "fil": "Sumulat ng code na gagawa ng sumusunod sa Filipino:",
        "ja": "次の日本語の指示を満たすコードを書いてください：",
        "ko": "다음 한국어 지시를 수행하는 코드를 작성해주세요:",
        "zh": "请编写满足以下中文要求的代码：",
        "es": "Escribe código que cumpla lo siguiente en español:",
    },
}


def get_multilingual_template(task: str, language: str) -> str:
    """
    Get a multilingual training template for a specific task and language.

    Args:
        task: One of 'explain_code', 'add_comments', 'translate_comments', 'generate_code'
        language: BCP-47 language code

    Returns:
        Template string in the target language
    """
    templates = MULTILINGUAL_TEMPLATES.get(task, MULTILINGUAL_TEMPLATES["explain_code"])
    return templates.get(language, templates["en"])


# ── Code Comment Analysis ────────────────────

def extract_comments(code: str, language: str = "python") -> List[Dict]:
    """
    Extract code comments with their context.

    Args:
        code: Source code
        language: Programming language for comment syntax

    Returns:
        List of dicts with 'comment', 'line', and 'context' keys
    """
    comments = []
    comment_patterns = {
        "python": r'(#.*?$|""".*?"""|\'\'\'.*?\'\'\')',
        "javascript": r'(//.*?$|/\*[\s\S]*?\*/)',
        "typescript": r'(//.*?$|/\*[\s\S]*?\*/)',
        "html": r'(<!--[\s\S]*?-->)',
        "css": r'(/\*[\s\S]*?\*/)',
    }

    pattern = comment_patterns.get(language, comment_patterns["python"])
    lines = code.split("\n")

    for i, line in enumerate(lines):
        for match in re.finditer(pattern, line, re.MULTILINE | re.DOTALL):
            comment_text = match.group(0).strip()
            if comment_text:
                comments.append({
                    "comment": comment_text,
                    "line": i + 1,
                    "context": lines[max(0, i - 2):i + 3],
                })

    return comments


def translate_code_comments(comments: List[Dict], source_lang: str, target_lang: str) -> List[Dict]:
    """
    Prepare code comments for multilingual translation fine-tuning.

    Args:
        comments: List of comment dicts from extract_comments()
        source_lang: Source language code
        target_lang: Target language code

    Returns:
        List of (source_comment, target_comment, context) tuples for training
    """
    pairs = []
    for c in comments:
        pairs.append({
            "source_comment": c["comment"],
            "source_lang": source_lang,
            "target_lang": target_lang,
            "context": "\n".join(c["context"]),
        })
    return pairs


# ── Multilingual Documentation Generation ────

def generate_documentation_prompt(code: str, language: str = "en") -> str:
    """
    Generate a prompt for creating multilingual documentation from code.
    """
    prefix = get_language_prefix(language)
    lang_name = LANGUAGE_CONFIG.get(language, {}).get("name", "English")
    return (
        f"{prefix}\n"
        f"Generate comprehensive documentation in {lang_name} for the following code.\n"
        f"Include:\n"
        f"- Purpose and functionality\n"
        f"- Parameters and return values\n"
        f"- Usage examples\n"
        f"- Edge cases and error handling\n\n"
        f"```\n{code}\n```"
    )