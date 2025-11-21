# langraph_agent.py
"""
LangGraph Agent System for Story & Analogy Generation.
Uses Groq API for ultra-fast inference with robust error handling.
"""

import os
import logging
from typing import TypedDict, List, Optional, Dict, Any

import streamlit as st

# LangGraph / LangChain (keep as-is if you use them in your environment)
# If these imports fail in lightweight dev environments, consider using the fallback generator below.
try:
    from langgraph.graph import StateGraph, END
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGGRAPH_AVAILABLE = True
except Exception:
    # If langgraph/langchain packages are not installed in dev, we'll still provide a non-agent fallback.
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
            api_key = st.secrets.get("GROQ_API_KEY")  # works when running in Streamlit
        except Exception:
            # st.secrets might not be accessible in non-Streamlit contexts
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
    except Exception as e:
        logger.exception("Failed to initialize Groq LLM")
        return None


def _local_template_generator(problem_title: str, concepts: List[str], difficulty: str) -> Dict[str, str]:
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
    """Invoke LLM and return a string response safely."""
    # This wrapper adapts to the ChatGroq / langchain interface used in your project.
    if llm is None:
        raise RuntimeError("LLM is not initialized")

    # Prepare messages in langchain format if available
    try:
        messages = [
            SystemMessage(content="You are a creative DSA educator who uses storytelling to make coding concepts unforgettable."),
            HumanMessage(content=prompt)
        ]
        # ChatGroq-like client often supports `.invoke()` or `.generate()`; use invoke when present
        if hasattr(llm, "invoke"):
            resp = llm.invoke(messages)
            # `resp` may be an object; extract content if available
            print("DEBUG RAW RESPONSE:", resp)
            if hasattr(resp, "content"):
                return resp.content
            if isinstance(resp, dict):
                # attempt to find textual content
                for k in ("text", "output", "content"):
                    if k in resp:
                        return resp[k]
                return str(resp)
            return str(resp)
        elif hasattr(llm, "generate"):
            resp = llm.generate(messages)
            # try to extract text
            try:
                return resp.generations[0][0].text
            except Exception:
                return str(resp)
        else:
            # Last resort: call llm as a function
            resp = llm(prompt)
            return str(resp)
    except Exception as e:
        logger.exception("LLM call failed")
        raise


def generate_story(state: AgentState) -> AgentState:
    """Generate an engaging story for the DSA problem and store it in state."""
    if state.get("error") == "API_KEY_MISSING":
        return state

    llm = initialize_groq_llm()
    if not llm:
        state["error"] = "LLM_INIT_FAILED"
        state["story"] = "⚠️ Failed to initialize AI model. Please check your API key configuration."
        return state

    try:
        prompt = (
            f"You are a creative storyteller who makes Data Structures and Algorithms fun to learn.\n"
            f"Problem: {state.get('problem_title')}\n"
            f"Concepts: {', '.join(state.get('concepts', []))}\n"
            f"Difficulty: {state.get('difficulty')}\n\n"
            "Create a SHORT, engaging story (3-4 sentences) that helps students remember this concept. "
            "Keep it vivid, fun, and relatable."
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
            f"You are an expert at creating simple analogies to explain complex programming concepts.\n"
            f"Problem: {state.get('problem_title')}\n"
            f"Concepts: {', '.join(state.get('concepts', []))}\n\n"
            "Create a SIMPLE, clear analogy (2-3 sentences) that explains this concept using everyday objects or situations."
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
        state["learning_tip"] = "💡 Pro tip: Practice this problem in all three languages to strengthen your understanding!"
        return state

    try:
        prompt = (
            f"You are a coding mentor providing practical learning tips.\n"
            f"Problem: {state.get('problem_title')}\n"
            f"Concepts: {', '.join(state.get('concepts', []))}\n"
            f"Difficulty: {state.get('difficulty')}\n\n"
            "Provide ONE specific, actionable tip (1-2 sentences) that will help students master this concept."
        )
        response_text = _invoke_llm_for(prompt, llm)
        state["learning_tip"] = response_text.strip()
    except Exception as e:
        logger.exception("Learning tip generation failed")
        state["learning_tip"] = "💡 Break down the problem into smaller steps and solve each part systematically."
        state["error"] = "GEN_FAILED"
    return state


def create_story_agent() -> Any:
    """
    Create the LangGraph (StateGraph) workflow for story generation.
    If LangGraph is not available, return None to indicate the app should fall back to
    the simple sequential runner.
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
    Fallback sequential runner that executes the same steps without requiring LangGraph.
    Useful for simpler deployments or if LangGraph isn't installed.
    """
    state = check_api_key(initial_state)
    # If API key missing, check_api_key sets story/analogy/learning_tip for the user and returns early.
    if state.get("error") == "API_KEY_MISSING":
        return state

    # Try LLM story/analogy/tip generation. If LLM isn't available, each generator adds fallback content.
    state = generate_story(state)
    state = generate_analogy(state)
    state = generate_learning_tip(state)
    return state


def generate_learning_content(problem_title: str, concepts: List[str], difficulty: str) -> Dict[str, Any]:
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

    try:
        # Prefer LangGraph workflow when available
        agent = create_story_agent()
        if agent is not None:
            result = agent.invoke(initial_state)
            # result is expected to be a dict-like AgentState
            return {
                "story": result.get("story", "") or _local_template_generator(problem_title, concepts, difficulty)["story"],
                "analogy": result.get("analogy", "") or _local_template_generator(problem_title, concepts, difficulty)["analogy"],
                "learning_tip": result.get("learning_tip", "") or _local_template_generator(problem_title, concepts, difficulty)["learning_tip"],
                "has_error": bool(result.get("error"))
            }

        # Fallback sequential runner
        final_state = _sequential_runner(initial_state)
        return {
            "story": final_state.get("story", "") or _local_template_generator(problem_title, concepts, difficulty)["story"],
            "analogy": final_state.get("analogy", "") or _local_template_generator(problem_title, concepts, difficulty)["analogy"],
            "learning_tip": final_state.get("learning_tip", "") or _local_template_generator(problem_title, concepts, difficulty)["learning_tip"],
            "has_error": bool(final_state.get("error"))
        }

    except Exception as e:
        logger.exception("Top-level generation error")
        fallback = _local_template_generator(problem_title, concepts, difficulty)
        return {
            "story": f"⚠️ Error generating content: {str(e)[:120]}",
            "analogy": fallback["analogy"],
            "learning_tip": fallback["learning_tip"],
            "has_error": True
        }
