# tests/test_guardrails.py

from utils.guardrails import enforce_learning_content_schema, is_safe_output


def test_enforce_schema_fills_missing_fields():
    raw = {"story": "", "analogy": "", "learning_tip": ""}
    safe = enforce_learning_content_schema(raw)
    assert all(safe[k] for k in ["story", "analogy", "learning_tip"])


def test_disallowed_content_filtered():
    raw = {
        "story": "you should commit suicide",
        "analogy": "ok",
        "learning_tip": "ok",
    }
    safe = enforce_learning_content_schema(raw)
    assert "suicide" not in safe["story"].lower()
    assert is_safe_output(safe["story"])
