# tests/test_langraph_agent_semantic.py
"""
Semantic / behavioral tests for langraph_agent.py

These tests follow the patterns from:
"Testing Agentic AI Applications: How to Use Pytest for LLM-Based Workflows"

They focus on:
- Response structure and schema
- Content relevance and behavioral signals
- Safety / guardrails for API key handling
- Edge-case handling
- Basic performance expectations

All tests avoid real LLM / network calls by using fallbacks or monkeypatching.
"""

import time
from typing import List

import pytest

import langraph_agent as la


# ---------- Helpers ----------

def _make_state(
    title: str = "Binary Search",
    concepts: List[str] | None = None,
    difficulty: str = "Medium",
):
    if concepts is None:
        concepts = ["Binary Search", "Divide and Conquer"]
    return {
        "problem_title": title,
        "concepts": concepts,
        "difficulty": difficulty,
        "story": "",
        "analogy": "",
        "learning_tip": "",
        "error": None,
        "has_api_key": False,
    }


# ---------- 1. Structure & Schema tests ----------

def test_generate_learning_content_structure_with_mock_llm(monkeypatch):
    """
    Test that generate_learning_content returns properly structured dict
    with required keys and reasonable types.
    """

    # Force: no LangGraph graph -> sequential runner path
    monkeypatch.setattr(la, "create_story_agent", lambda: None)

    # Make API key check always succeed (so we don't get API_KEY_MISSING copy)
    def fake_check_api_key(state):
        state["has_api_key"] = True
        state["error"] = None
        return state

    monkeypatch.setattr(la, "check_api_key", fake_check_api_key)

    # Return a fake LLM object (non-None so generate_* doesn't early fail)
    class FakeLLM:
        pass

    monkeypatch.setattr(la, "initialize_groq_llm", lambda: FakeLLM())

    # Avoid real LLM calls – make _invoke_llm_for return deterministic text
    def fake_invoke(prompt: str, llm: FakeLLM) -> str:
        return "This is a fake LLM response for testing."

    monkeypatch.setattr(la, "_invoke_llm_for", fake_invoke)

    result = la.generate_learning_content(
        problem_title="Binary Search",
        concepts=["Binary Search", "Divide and Conquer"],
        difficulty="Easy",
    )

    # Schema & types
    assert isinstance(result, dict)
    for key in ["story", "analogy", "learning_tip", "has_error"]:
        assert key in result

    assert isinstance(result["story"], str)
    assert isinstance(result["analogy"], str)
    assert isinstance(result["learning_tip"], str)
    assert isinstance(result["has_error"], bool)

    # Reasonable non-empty content
    assert len(result["story"]) > 0
    assert len(result["analogy"]) > 0
    assert len(result["learning_tip"]) > 0


# ---------- 2. Content relevance / behavior tests (use deterministic fallback) ----------

def test_local_template_generator_relevance_and_tone():
    """
    Test _local_template_generator for:
    - relevance (mentions title / concepts)
    - friendly, narrative tone (e.g. 'imagine', 'explorer')
    - analogy and tip contain expected patterns.
    """
    title = "Kadane's Algorithm"
    concepts = ["Dynamic Programming", "Arrays"]
    difficulty = "Medium"

    out = la._local_template_generator(title, concepts, difficulty)

    # Basic structure
    assert isinstance(out, dict)
    for key in ["story", "analogy", "learning_tip"]:
        assert key in out
        assert isinstance(out[key], str)
        assert len(out[key]) > 0

    story = out["story"].lower()
    analogy = out["analogy"].lower()
    tip = out["learning_tip"].lower()

    # Story relevance: should reference problem title and at least one concept
    assert "kadane" in story
    assert ("dynamic programming".lower() in story) or ("arrays".lower() in story)

    # Tone: narrative / encouraging phrases from your template
    assert "imagine" in story or "explorer" in story

    # Analogy behavior: should contain the "recipe / ingredients" mental model
    assert "recipe" in analogy or "ingredients" in analogy

    # Tip should be concrete and mention practice or corner cases
    assert "test" in tip or "practice" in tip or "corner" in tip


def test_local_template_generator_empty_concepts_still_useful():
    """
    Edge case: when concepts list is empty, generator should still
    produce a meaningful story using a generic phrase.
    """
    out = la._local_template_generator(
        problem_title="Two Sum",
        concepts=[],
        difficulty="Easy",
    )

    story = out["story"].lower()

    # Still non-empty and references the title
    assert "two sum" in story
    assert "programming concepts" in story  # from your fallback phrasing


# ---------- 3. Safety & API key guardrail tests ----------

def test_check_api_key_missing_sets_error_and_friendly_copy(monkeypatch):
    """
    When no API key is present, check_api_key should:
    - set has_api_key = False
    - set error = 'API_KEY_MISSING'
    - populate story/analogy/learning_tip with helpful guidance.
    """

    # Ensure environment doesn't accidentally provide a key
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    # Also make sure Streamlit secrets (if present in other envs) don't interfere
    if hasattr(la, "st") and la.st is not None:
        # Fake secrets that don't contain GROQ_API_KEY
        class FakeSecrets(dict):
            def get(self, key, default=None):
                return None

        la.st.secrets = FakeSecrets()

    state = _make_state()
    out = la.check_api_key(state)

    assert out["has_api_key"] is False
    assert out["error"] == "API_KEY_MISSING"

    # Story and analogy should mention that an API key is required
    assert "api key" in out["story"].lower()
    assert "groq" in out["story"].lower()
    assert "treasure" in out["analogy"].lower() or "unlock" in out["analogy"].lower()

    # Tip should encourage setting up the key
    assert "tip" in out["learning_tip"].lower() or "unlock" in out["learning_tip"].lower()


# ---------- 4. Edge-case handling for top-level API ----------

def test_generate_learning_content_handles_minimal_input(monkeypatch):
    """
    Test that generate_learning_content handles minimal / odd input gracefully
    (empty title, empty concepts, missing difficulty).
    """

    # Force sequential runner path (no LangGraph)
    monkeypatch.setattr(la, "create_story_agent", lambda: None)

    # Make check_api_key pass
    def fake_check_api_key(state):
        state["has_api_key"] = True
        state["error"] = None
        return state

    monkeypatch.setattr(la, "check_api_key", fake_check_api_key)

    # Force LLM init failure so we exercise the "LLM_INIT_FAILED" path + fallbacks
    monkeypatch.setattr(la, "initialize_groq_llm", lambda: None)

    # Use real generate_* functions, but they will hit the no-LLM behavior
    result = la.generate_learning_content(
        problem_title="",
        concepts=[],
        difficulty="",
    )

    # Even with weird input, we should get reasonable strings back
    assert isinstance(result["story"], str)
    assert isinstance(result["analogy"], str)
    assert isinstance(result["learning_tip"], str)

    assert len(result["story"]) > 0
    assert len(result["analogy"]) > 0
    assert len(result["learning_tip"]) > 0

    # has_error should be True because LLM init failed
    assert result["has_error"] is True


# ---------- 5. Performance-style test (no real LLM) ----------

def test_generate_learning_content_response_time_with_mocks(monkeypatch):
    """
    Basic performance-style test:
    With mocked LLM and no network, generate_learning_content should return quickly.
    (This is to mirror the tutorial's 'response time' pattern, not a real SLA.)
    """

    # No LangGraph
    monkeypatch.setattr(la, "create_story_agent", lambda: None)

    # Always pass API key check
    def fake_check_api_key(state):
        state["has_api_key"] = True
        state["error"] = None
        return state

    monkeypatch.setattr(la, "check_api_key", fake_check_api_key)

    # Fake LLM object
    class FakeLLM:
        pass

    monkeypatch.setattr(la, "initialize_groq_llm", lambda: FakeLLM())

    # Fast fake LLM invocation
    def fake_invoke(prompt: str, llm: FakeLLM) -> str:
        # Could use prompt to customize, but constant is enough here
        return "Short, fast test response."

    monkeypatch.setattr(la, "_invoke_llm_for", fake_invoke)

    start = time.time()
    result = la.generate_learning_content(
        problem_title="Stacks and Queues",
        concepts=["Stack", "Queue", "FIFO", "LIFO"],
        difficulty="Easy",
    )
    end = time.time()

    elapsed = end - start

    # Sanity check: we got something back
    assert isinstance(result["story"], str)
    assert len(result["story"]) > 0

    # Performance bound – this should basically always be true in tests
    assert elapsed < 1.0, f"generate_learning_content took too long: {elapsed:.3f}s"


# ---------- 6. Consistency-style test (using deterministic fallback) ----------

def test_local_template_generator_consistency_for_similar_inputs():
    """
    Similar inputs should produce structurally similar outputs and
    preserve key concepts in the story.

    This mirrors the tutorial's 'consistency' idea, but using the
    deterministic fallback generator instead of a real LLM.
    """

    titles = [
        "Largest Element in Array",
        "Find Largest Element in Array",
        "Array Maximum Element",
    ]
    concepts = ["Arrays", "Linear Scan"]

    outputs = [
        la._local_template_generator(title, concepts, "Easy")
        for title in titles
    ]

    # All stories should contain the word 'array'
    for out in outputs:
        assert "array" in out["story"].lower()

    # Stories should be non-empty and reasonably sized
    lengths = [len(out["story"]) for out in outputs]
    assert min(lengths) > 20
    assert max(lengths) < 1000
