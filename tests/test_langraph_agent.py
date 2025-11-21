import types
import langraph_agent


class FakeLLM:
    """Minimal fake LLM that returns an object with .content."""

    class Resp:
        def __init__(self, text: str):
            self.content = text

    def __init__(self, prefix: str):
        self.prefix = prefix
        self.calls = []

    def invoke(self, messages):
        # messages is a list of SystemMessage + HumanMessage
        self.calls.append(messages)
        return FakeLLM.Resp(f"{self.prefix}: {messages[-1].content[:20]}")


def test_generate_learning_content_with_mock_llm(monkeypatch):
    # Pretend we have an API key so check_api_key passes
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")

    # Force using sequential runner (no LangGraph)
    monkeypatch.setattr(langraph_agent, "LANGGRAPH_AVAILABLE", False)

    fake_llm = FakeLLM("story/analogy/tip")

    def fake_init():
        return fake_llm

    # Make all LLM initializations return our fake
    monkeypatch.setattr(langraph_agent, "initialize_groq_llm", fake_init)

    result = langraph_agent.generate_learning_content(
        problem_title="Two Sum",
        concepts=["Arrays", "HashMap"],
        difficulty="Easy",
    )

    assert "story" in result
    assert "analogy" in result
    assert "learning_tip" in result
    assert isinstance(result["story"], str)
    assert isinstance(result["analogy"], str)
    assert isinstance(result["learning_tip"], str)
    assert result["has_error"] in (True, False)


def test_generate_learning_content_falls_back_without_api_key(monkeypatch):
    # Ensure no API key is present
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    # Force sequential runner
    monkeypatch.setattr(langraph_agent, "LANGGRAPH_AVAILABLE", False)

    # Also, make LLM init return None to simulate failure
    monkeypatch.setattr(langraph_agent, "initialize_groq_llm", lambda: None)

    result = langraph_agent.generate_learning_content(
        problem_title="Binary Search",
        concepts=["Binary Search"],
        difficulty="Medium",
    )

    # We still expect safe, non-empty text from the local template + guardrails
    assert "story" in result
    assert "analogy" in result
    assert "learning_tip" in result
    assert isinstance(result["story"], str)
    assert isinstance(result["analogy"], str)
    assert isinstance(result["learning_tip"], str)
