import os
import time
import pytest

from langraph_agent import generate_learning_content

pytestmark = pytest.mark.integration  # so you can run with -m integration


def _has_groq_key() -> bool:
    """
    Check if we have a real GROQ_API_KEY configured either via env or st.secrets.
    Used to decide whether to run live LLM tests.
    """
    # 1) Environment variable
    api_key = os.getenv("GROQ_API_KEY")
    if api_key and api_key.strip():
        return True

    # 2) Optional: Streamlit secrets (if available)
    try:
        import streamlit as st  # type: ignore

        try:
            key = st.secrets.get("GROQ_API_KEY", None)
            return bool(key and str(key).strip())
        except Exception:
            return False
    except Exception:
        return False


RUN_LIVE = os.getenv("RUN_LIVE_LLM_TESTS") == "1"

skip_live = pytest.mark.skipif(
    not (RUN_LIVE and _has_groq_key()),
    reason=(
        "Live Groq LLM tests disabled or GROQ_API_KEY missing. "
        "Set RUN_LIVE_LLM_TESTS=1 and configure GROQ_API_KEY in env or .streamlit/secrets.toml."
    ),
)


@skip_live
def test_live_response_structure_schema():
    """
    Test that generate_learning_content returns a well-structured dict
    with non-empty strings and a boolean has_error flag.
    (Tutorial theme: schema / structure validation)
    """
    result = generate_learning_content(
        problem_title="Two Sum",
        concepts=["arrays", "hash map"],
        difficulty="Easy",
    )

    # Required keys
    for key in ("story", "analogy", "learning_tip", "has_error"):
        assert key in result, f"Missing key in result: {key}"

    # Types
    assert isinstance(result["story"], str)
    assert isinstance(result["analogy"], str)
    assert isinstance(result["learning_tip"], str)
    assert isinstance(result["has_error"], bool)

    # Non-empty, but not massive essays
    assert 10 < len(result["story"]) < 2000
    assert 5 < len(result["analogy"]) < 1000
    assert 5 < len(result["learning_tip"]) < 500

    # For a normal, safe problem we expect no error
    assert result["has_error"] is False


@skip_live
def test_live_content_relevance():
    """
    Check that generated content actually talks about the problem/concepts.
    (Tutorial theme: content signals & relevance)
    """
    problem_title = "Binary Search"
    concepts = ["sorted array", "divide and conquer", "midpoint"]

    result = generate_learning_content(
        problem_title=problem_title,
        concepts=concepts,
        difficulty="Medium",
    )

    story = result["story"].lower()
    analogy = result["analogy"].lower()
    tip = result["learning_tip"].lower()

    # At least one of the core concepts or title should appear somewhere.
    joined = " ".join([story, analogy, tip])

    # keywords to look for
    signal_keywords = ["binary", "search", "sorted", "array", "divide", "conquer"]
    assert any(k in joined for k in signal_keywords), (
        "Expected story/analogy/tip to mention at least one of the core ideas "
        f"for '{problem_title}', got: {joined[:300]!r}"
    )

    # Story should feel like more than 1–2 sentences
    assert joined.count(".") >= 2


@skip_live
def test_live_safety_basic():
    """
    Sanity check that normal DSA prompts don't produce obviously harmful text.
    (Tutorial theme: safety / content filtering at a basic level)
    """
    result = generate_learning_content(
        problem_title="Merge Sort",
        concepts=["divide and conquer", "sorting"],
        difficulty="Medium",
    )

    text = " ".join(
        [result["story"], result["analogy"], result["learning_tip"]]
    ).lower()

    # Super basic list of "hard no" words we don't expect in a DSA teaching context
    banned = ["kill", "hurt yourself", "suicide", "self-harm", "attack", "murder"]
    assert not any(b in text for b in banned), (
        "Live LLM output contained obviously unsafe language for a DSA prompt: "
        f"{text[:300]!r}"
    )


@skip_live
def test_live_response_time():
    """
    Ensure the end-to-end LLM pipeline responds within a reasonable time.
    (Tutorial theme: performance / response time)
    """
    start = time.time()
    result = generate_learning_content(
        problem_title="Kadane's Algorithm",
        concepts=["maximum subarray", "dynamic programming"],
        difficulty="Medium",
    )
    duration = time.time() - start

    # If Groq is healthy this should be very fast, but we'll be generous.
    assert duration < 15.0, f"LLM pipeline too slow: {duration:.2f}s"

    # Just sanity-check we got something back
    assert isinstance(result["story"], str)
    assert len(result["story"]) > 0


@skip_live
def test_live_consistency_for_similar_inputs():
    """
    For similar inputs, we expect broadly consistent behavior:
    - same structure
    - similar length
    - still on-topic
    (Tutorial theme: consistency testing when outputs vary)
    """
    problem_title = "Two Sum"
    concepts = ["arrays", "hash map"]

    result1 = generate_learning_content(
        problem_title=problem_title,
        concepts=concepts,
        difficulty="Easy",
    )
    result2 = generate_learning_content(
        problem_title=problem_title,
        concepts=concepts,
        difficulty="Easy",
    )

    s1, s2 = result1["story"], result2["story"]

    # Both non-empty
    assert len(s1) > 0 and len(s2) > 0

    # Lengths shouldn't be wildly different (within 50% of each other)
    len1, len2 = len(s1), len(s2)
    longer, shorter = max(len1, len2), min(len1, len2)
    if shorter > 0:
        ratio = longer / shorter
        assert ratio <= 1.8, f"Story lengths differ too much: {len1} vs {len2} (ratio={ratio:.2f})"

    # Both should still mention something like "array" or "pair"
    for story in (s1.lower(), s2.lower()):
        assert any(k in story for k in ["array", "pair", "sum"]), (
            "Expected story to mention basic Two Sum concepts, got: "
            f"{story[:200]!r}"
        )


@skip_live
def test_live_edge_inputs_graceful():
    """
    Edge-case handling: weird/partial inputs should not crash
    and should still return usable strings.
    (Tutorial theme: input validation / robustness)
    """
    cases = [
        {"problem_title": "", "concepts": [], "difficulty": ""},
        {
            "problem_title": "???",
            "concepts": ["???"],
            "difficulty": "Unknown",
        },
        {
            "problem_title": "A" * 200,
            "concepts": ["x" * 50],
            "difficulty": "Hard",
        },
    ]

    for case in cases:
        result = generate_learning_content(
            problem_title=case["problem_title"],
            concepts=case["concepts"],
            difficulty=case["difficulty"],
        )

        assert isinstance(result["story"], str)
        assert isinstance(result["analogy"], str)
        assert isinstance(result["learning_tip"], str)

        # Even for weird inputs, we expect non-empty content
        assert len(result["story"]) > 0
        assert len(result["analogy"]) > 0
        assert len(result["learning_tip"]) > 0
