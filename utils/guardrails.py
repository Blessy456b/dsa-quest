# utils/guardrails.py

from typing import Dict, Any
import re


DISALLOWED_PATTERNS = [
    r"suicide",
    r"self-harm",
    r"kill yourself",
    r"terrorist",
]


def sanitize_user_text(text: str) -> str:
    """
    Basic sanitization for user-visible text.
    - Strips leading/trailing spaces
    - Normalizes whitespace
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_safe_output(text: str) -> bool:
    """Return False if text clearly violates basic content rules."""
    lowered = text.lower()
    for pattern in DISALLOWED_PATTERNS:
        if re.search(pattern, lowered):
            return False
    return True


def enforce_learning_content_schema(content: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure generated content has story, analogy, learning_tip keys with safe values.
    If something is missing/unsafe, replace with a safe fallback.
    """
    fallback_story = (
        "Imagine tackling this problem step by step, starting from simple examples "
        "and gradually building your intuition about the underlying concept."
    )
    fallback_analogy = (
        "Think of this problem like organizing items in boxes: each box represents a "
        "case the algorithm needs to handle."
    )
    fallback_tip = (
        "Break the problem into smaller subproblems, solve them individually, and "
        "then connect the pieces."
    )

    story = str(content.get("story") or "").strip()
    analogy = str(content.get("analogy") or "").strip()
    learning_tip = str(content.get("learning_tip") or "").strip()

    if not story or not is_safe_output(story):
        story = fallback_story
    if not analogy or not is_safe_output(analogy):
        analogy = fallback_analogy
    if not learning_tip or not is_safe_output(learning_tip):
        learning_tip = fallback_tip

    return {
        "story": sanitize_user_text(story),
        "analogy": sanitize_user_text(analogy),
        "learning_tip": sanitize_user_text(learning_tip),
    }
