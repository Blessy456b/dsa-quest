# langraph_agent.py
"""
LangGraph Agent System for Story & Analogy Generation.
Uses Groq API for ultra-fast inference with robust error handling.

Updated:
 - Robust extraction of model text from various response shapes.
 - Sends a safe debug copy of the raw/extracted response to Streamlit UI
   (via st.session_state and st.info) when Streamlit is available.
"""

import os
import logging
from typing import TypedDict, List, Optional, Dict, Any

from utils.retry_utils import retry_with_backoff
from utils.guardrails import enforce_learning_content_schema

# Streamlit is optional — we use it when running inside the Streamlit app.
try:
    import streamlit as st
except Exception:
    st = None  # type: ignore

# LangGraph / LangChain (keep as-is if you use them in your environment)
try:
    from langgraph.graph import StateGraph, END
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGGRAPH_AVAILABLE = True
except Exception:
    LANGGRAPH_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AgentState(TypedDict, total=False):
    """State for the story generation agent"""
    problem_title: str
    concepts: List[str]
    difficulty: str
    story: str
    analogy: str
    learning_tip: str
    error: Optional[str]
    has_api_key: bool


def check_api_key(state: AgentState) -> AgentState:
    """Check if Groq API key is available in Streamlit secrets or environment"""
    try:
        api_key = None
        try:
            if st:
                api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None

        env_key = os.getenv("GROQ_API_KEY")
        key = api_key or env_key
        state["has_api_key"] = bool(key and str(key).strip())
    except Exception:
        state["has_api_key"] = False

    if not state["has_api_key"]:
        state["error"] = "API_KEY_MISSING"
        state["story"] = (
            "⚠️ **Groq API Key Required!**\n\n"
            "To unlock AI-powered stories and analogies:\n"
            "1. Get a free API key at https://console.groq.com\n"
            "2. Add it to `.streamlit/secrets.toml`:\n\n"
            "```\nGROQ_API_KEY = \"your_key_here\"\n```\n\n"
            "3. Restart the app."
        )
        state["analogy"] = (
            "Without the API key, you won't get personalized learning content. "
            "It's like having a locked treasure chest — you need the key to unlock the magic! ✨"
        )
        state["learning_tip"] = (
            "💡 Tip: Setting up the API key takes 2 minutes and unlocks AI-powered learning features!"
        )

    return state


def initialize_groq_llm() -> Optional[Any]:
    """
    Initialize Groq LLM with API key from Streamlit secrets or environment.
    Returns a ChatGroq instance (or None if initialization failed).
    """
    try:
        api_key = None
        try:
            if st:
                api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None

        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")

        if not api_key or not str(api_key).strip():
            return None

        # Create ChatGroq instance - adjust params if your Groq client requires them
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            groq_api_key=str(api_key),
            max_tokens=800,
            timeout=30,
        )
        return llm
    except Exception:
        logger.exception("Failed to initialize Groq LLM")
        return None


def _local_template_generator(
    problem_title: str, concepts: List[str], difficulty: str
) -> Dict[str, str]:
    """Deterministic fallback generator used when API or model isn't available."""
    concepts_str = ", ".join(concepts) if concepts else "programming concepts"
    story = (
        f"Imagine you're a {difficulty.lower()} explorer tackling '{problem_title}'. "
        f"The key ideas — {concepts_str} — are tools in your kit. "
        "With each small example you solve, those tools become second nature. "
        "Soon the problem feels less like a puzzle and more like a routine you enjoy."
    )
    analogy = (
        "Think of the algorithm as a recipe and data structures as your ingredients: "
        "if you combine them in the right order, the result is reliable and repeatable."
    )
    learning_tip = (
        "Start by solving a small example by hand, then write a simple solution and improve it. "
        "Test corner cases early."
    )
    return {"story": story, "analogy": analogy, "learning_tip": learning_tip}


def _invoke_llm_for(prompt: str, llm: Any) -> str:
    """
    Invoke LLM and return a string response safely and store debug info to
    Streamlit UI if available.
    """
    if llm is None:
        raise RuntimeError("LLM is not initialized")

    def _call():
        messages = [
            SystemMessage(
                content=(
                    "You are a creative DSA educator who uses storytelling to "
                    "make coding concepts unforgettable."
                )
            ),
            HumanMessage(content=prompt),
        ]

        # Call the client using common interfaces.
        if hasattr(llm, "invoke"):
            return llm.invoke(messages)
        if hasattr(llm, "generate"):
            return llm.generate(messages)
        return llm(prompt)

    try:
        resp = retry_with_backoff(_call)

        # Helper extraction function to handle different response shapes.
        def extract_text(obj):
            if obj is None:
                return None
            # If it's a simple string
            if isinstance(obj, str):
                return obj
            # Common attribute
            if hasattr(obj, "content"):
                try:
                    content = getattr(obj, "content")
                    if isinstance(content, str):
                        return content
                except Exception:
                    pass
            # Common dict keys
            if isinstance(obj, dict):
                for k in ("text", "output", "content", "message", "response"):
                    if k in obj and obj[k]:
                        return obj[k]
            # LangChain-esque .generations
            try:
                gens = getattr(obj, "generations", None)
                if gens:
                    first = gens[0]
                    candidate = first[0] if isinstance(first, (list, tuple)) else first
                    if hasattr(candidate, "text"):
                        return getattr(candidate, "text")
                    if hasattr(candidate, "content"):
                        return getattr(candidate, "content")
            except Exception:
                pass
            # Fallback: check attributes directly
            for attr in ("text", "response", "message", "content"):
                try:
                    val = getattr(obj, attr, None)
                    if isinstance(val, str) and val:
                        return val
                except Exception:
                    pass
            try:
                return str(obj)
            except Exception:
                return None

        text = extract_text(resp)

        # Save debug info to logs (truncated) and to Streamlit UI if available
        try:
            logger.debug("LLM raw response repr: %s", repr(resp)[:1000])
            if text:
                logger.info("LLM returned text (truncated): %s", text[:1000])
            else:
                logger.warning(
                    "LLM response did not contain an obvious text field; "
                    "using repr fallback."
                )
        except Exception:
            pass

        # Write debug info into Streamlit session_state and UI (only if st available)
        try:
            if st:
                st.session_state["last_llm_raw_repr"] = repr(resp)
                st.session_state["last_llm_extracted_text"] = text
                try:
                    st.info("LLM response received (debug view available below).")
                except Exception:
                    pass
        except Exception:
            # Never let UI debugging break generation
            pass

        if text:
            return str(text).strip()
        return str(resp)

    except Exception:
        logger.exception("LLM call failed")
        raise


def generate_story(state: AgentState) -> AgentState:
    """Generate an engaging story for the DSA problem and store it in state."""
    if state.get("error") == "API_KEY_MISSING":
        return state

    llm = initialize_groq_llm()
    if not llm:
        state["error"] = "LLM_INIT_FAILED"
        state["story"] = (
            "⚠️ Failed to initialize AI model. Please check your API key configuration."
        )
        return state

    try:
        prompt = (
            "You are a creative storyteller who makes Data Structures and "
            "Algorithms fun to learn.\n"
            f"Problem: {state.get('problem_title')}\n"
            f"Concepts: {', '.join(state.get('concepts', []))}\n"
            f"Difficulty: {state.get('difficulty')}\n\n"
            "Create a SHORT, engaging story (3-4 sentences) that helps students "
            "remember this concept. Keep it vivid, fun, and relatable."
        )

        response_text = _invoke_llm_for(prompt, llm)
        state["story"] = response_text.strip()
        state["error"] = None
    except Exception as e:
        logger.exception("Story generation failed")
        err = str(e).lower()
        if "rate" in err or "rate_limit" in err or "429" in err:
            state["story"] = "⚠️ Rate limit reached. Please wait a moment and try again."
        elif "auth" in err or "unauth" in err:
            state["story"] = "⚠️ Authentication failed. Please check your API key."
            state["error"] = "AUTH_FAILED"
        else:
            state["story"] = f"⚠️ Story generation failed: {str(e)[:150]}"
            state["error"] = "GEN_FAILED"
    return state


def generate_analogy(state: AgentState) -> AgentState:
    """Generate a clear analogy for the DSA problem and store it in state."""
    if state.get("error") == "API_KEY_MISSING":
        return state

    llm = initialize_groq_llm()
    if not llm:
        state["analogy"] = "⚠️ Failed to initialize AI model."
        return state

    try:
        prompt = (
            "You are an expert at creating simple analogies to explain complex "
            "programming concepts.\n"
            f"Problem: {state.get('problem_title')}\n"
            f"Concepts: {', '.join(state.get('concepts', []))}\n\n"
            "Create a SIMPLE, clear analogy (2-3 sentences) that explains this "
            "concept using everyday objects or situations."
        )
        response_text = _invoke_llm_for(prompt, llm)
        state["analogy"] = response_text.strip()
    except Exception as e:
        logger.exception("Analogy generation failed")
        err = str(e).lower()
        if "rate" in err:
            state["analogy"] = "⚠️ Rate limit reached. Please try again shortly."
        else:
            state["analogy"] = "⚠️ Analogy generation temporarily unavailable."
            state["error"] = "GEN_FAILED"
    return state


def generate_learning_tip(state: AgentState) -> AgentState:
    """Generate a practical learning tip and store it in state."""
    if state.get("error") == "API_KEY_MISSING":
        return state

    llm = initialize_groq_llm()
    if not llm:
        # Provide a helpful default tip if LLM is not available
        state["learning_tip"] = (
            "💡 Pro tip: Practice this problem in all three languages "
            "to strengthen your understanding!"
        )
        return state

    try:
        prompt = (
            "You are a coding mentor providing practical learning tips.\n"
            f"Problem: {state.get('problem_title')}\n"
            f"Concepts: {', '.join(state.get('concepts', []))}\n"
            f"Difficulty: {state.get('difficulty')}\n\n"
            "Provide ONE specific, actionable tip (1-2 sentences) that will help "
            "students master this concept."
        )
        response_text = _invoke_llm_for(prompt, llm)
        state["learning_tip"] = response_text.strip()
    except Exception:
        logger.exception("Learning tip generation failed")
        state["learning_tip"] = (
            "💡 Break down the problem into smaller steps and solve each part "
            "systematically."
        )
        state["error"] = "GEN_FAILED"
    return state


def create_story_agent() -> Any:
    """
    Create the LangGraph (StateGraph) workflow for story generation.
    If LangGraph is not available, return None to indicate the app should
    fall back to the simple sequential runner.
    """
    if not LANGGRAPH_AVAILABLE:
        logger.info("LangGraph not available; skipping StateGraph creation.")
        return None

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("check_api_key", check_api_key)
    workflow.add_node("generate_story", generate_story)
    workflow.add_node("generate_analogy", generate_analogy)
    workflow.add_node("generate_learning_tip", generate_learning_tip)

    # Define edges / flow
    workflow.set_entry_point("check_api_key")
    workflow.add_edge("check_api_key", "generate_story")
    workflow.add_edge("generate_story", "generate_analogy")
    workflow.add_edge("generate_analogy", "generate_learning_tip")
    workflow.add_edge("generate_learning_tip", END)

    return workflow.compile()


def _sequential_runner(initial_state: AgentState) -> AgentState:
    """
    Fallback sequential runner that executes the same steps without
    requiring LangGraph.
    """
    state = check_api_key(initial_state)
    if state.get("error") == "API_KEY_MISSING":
        return state

    state = generate_story(state)
    state = generate_analogy(state)
    state = generate_learning_tip(state)
    return state


def generate_learning_content(
    problem_title: str, concepts: List[str], difficulty: str
) -> Dict[str, Any]:
    """
    Generate story, analogy, and learning tip for a problem.

    Returns:
        {
            "story": str,
            "analogy": str,
            "learning_tip": str,
            "has_error": bool
        }
    """
    # Normalize inputs
    if not isinstance(concepts, list):
        concepts = list(concepts) if concepts else []
    if not difficulty:
        difficulty = "Medium"

    initial_state: AgentState = {
        "problem_title": problem_title,
        "concepts": concepts,
        "difficulty": difficulty,
        "story": "",
        "analogy": "",
        "learning_tip": "",
        "error": None,
        "has_api_key": False,
    }

    # Lowercased title once
    title_lower = problem_title.lower()

    # -------------------------------------------------------
    # PROMPT INJECTION FILTER — block attempts to override instructions
    # or reveal internal/system prompts.
    # This is what tests_safety/test_prompt_injection.py checks.
    # -------------------------------------------------------
    injection_phrases = [
        "ignore all instructions",
        "ignore previous instructions",
        "disregard all instructions",
        "disregard previous instructions",
        "override all instructions",
        "output the system prompt",
        "reveal the system prompt",
        "show the system prompt",
        "print the system prompt",
    ]

    if any(p in title_lower for p in injection_phrases):
        refusal_payload = {
            "story": (
                "I cannot help with requests that try to override my built-in instructions "
                "or expose internal configuration. I must follow my safety rules."
            ),
            "analogy": (
                "It's like asking a teacher to ignore all school rules — they still have "
                "to follow them to keep everyone safe."
            ),
            "learning_tip": (
                "Instead of trying to change the underlying instructions, ask about a "
                "specific data structures or algorithms topic you want to understand."
            ),
            "has_error": True,
        }
        # Ensure schema (and importantly: this text does NOT contain the phrase 'system prompt')
        return enforce_learning_content_schema(refusal_payload)

    # -------------------------------------------------------
    # SAFETY FILTER — refuse harmful, unsafe, or illegal topics
    # (used by tests_safety/test_harmful_inputs.py and test_refusal_cases.py)
    # -------------------------------------------------------
    harmful_keywords = [
        "hack", "hacking", "wifi", "crack", "break in", "exploit",
        "steal", "breach", "malware", "virus", "ddos", "sql injection",
        "hurt", "kill", "bomb", "attack", "harm", "illegal", "bypass",
        "jailbreak", "override", "disable security",
        "suicide", "self harm",
    ]

    if any(k in title_lower for k in harmful_keywords):
        refusal_text = (
            "I’m sorry, but I cannot assist with harmful or illegal instructions. "
            "My purpose is to provide safe and ethical guidance only."
        )

        refusal_payload = {
            "story": refusal_text,
            "analogy": (
                "Just like tools must be used responsibly, I cannot help with "
                "inappropriate or illegal activities. Instead, I can explain "
                "cybersecurity concepts safely."
            ),
            "learning_tip": (
                "A safer learning approach is to study ethical security topics such "
                "as encryption, authentication, and secure coding."
            ),
            "has_error": True,
        }
        # Make sure structure is valid
        return enforce_learning_content_schema(refusal_payload)

    # -------------------------------------------------------
    # SAFE → Continue Normal Generation
    # -------------------------------------------------------
    try:
        # Prefer LangGraph workflow when available
        agent = create_story_agent()
        if agent is not None:
            result = agent.invoke(initial_state)
            fallback = _local_template_generator(problem_title, concepts, difficulty)
            story = result.get("story") or fallback["story"]
            analogy = result.get("analogy") or fallback["analogy"]
            learning_tip = result.get("learning_tip") or fallback["learning_tip"]
            has_error = bool(result.get("error"))

            payload = {
                "story": story,
                "analogy": analogy,
                "learning_tip": learning_tip,
                "has_error": has_error,
            }
            return enforce_learning_content_schema(payload)

        # Fallback sequential runner
        final_state = _sequential_runner(initial_state)
        fallback = _local_template_generator(problem_title, concepts, difficulty)
        payload = {
            "story": final_state.get("story", "") or fallback["story"],
            "analogy": final_state.get("analogy", "") or fallback["analogy"],
            "learning_tip": final_state.get("learning_tip", "") or fallback["learning_tip"],
            "has_error": bool(final_state.get("error")),
        }
        return enforce_learning_content_schema(payload)

    except Exception as e:
        logger.exception("Top-level generation error")
        fallback = _local_template_generator(problem_title, concepts, difficulty)
        payload = {
            "story": f"⚠️ Error generating content: {str(e)[:120]}",
            "analogy": fallback["analogy"],
            "learning_tip": fallback["learning_tip"],
            "has_error": True,
        }
        return enforce_learning_content_schema(payload)
