# langraph_agent.py
"""
Lightweight OpenAI-backed agent for generating story, analogy, and a learning tip.
Replaces Groq/LangGraph specific code so you can use your OpenAI (ChatGPT 5) API key.
"""

import os
import streamlit as st
from typing import List, Dict, Any

# Use the official OpenAI package if available; otherwise fall back to httpx (quiet fallback).
try:
    import openai
    OPENAI_PKG = "openai"
except Exception:
    openai = None
    OPENAI_PKG = None

DEFAULT_MODEL = "gpt-4o"  # override via secrets OPENAI_MODEL if you prefer

def get_openai_api_key() -> str | None:
    """Get OpenAI API key from Streamlit secrets or environment."""
    try:
        key = None
        if hasattr(st, "secrets") and st.secrets.get("OPENAI_API_KEY"):
            key = st.secrets["OPENAI_API_KEY"]
        if not key:
            key = os.getenv("OPENAI_API_KEY")
        if key and isinstance(key, str) and key.strip():
            return key.strip()
    except Exception:
        pass
    return None

def openai_chat_completion(messages: List[Dict[str, str]], model: str, max_tokens: int = 400) -> str:
    """Call OpenAI ChatCompletions (sync). Returns assistant text or raises Exception."""
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY_MISSING")
    
    # prefer openai package if installed
    model_to_use = model or DEFAULT_MODEL

    if OPENAI_PKG:
        openai.api_key = api_key
        try:
            resp = openai.ChatCompletion.create(
                model=model_to_use,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            # OpenAI response structure: choices[0].message.content
            return resp.choices[0].message.content.strip()
        except Exception as e:
            # re-raise so caller can handle/log
            raise
    else:
        # If openai package isn't installed, use simple HTTP request to OpenAI REST API via requests
        import requests
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model_to_use, "messages": messages, "max_tokens": max_tokens, "temperature": 0.7}
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"OpenAI API error: {r.status_code} {r.text[:300]}")
        j = r.json()
        return j["choices"][0]["message"]["content"].strip()

def generate_story(problem_title: str, concepts: List[str], difficulty: str) -> str:
    """Generate a short, memorable story (3-4 sentences)."""
    concepts_str = ", ".join(concepts) if concepts else "core idea"
    system = {
        "role": "system",
        "content": "You are a creative teacher who converts DSA concepts into short, vivid stories that help learners remember algorithms and patterns."
    }
    human = {
        "role": "user",
        "content": (
            f"Create a SHORT (3-4 sentence), engaging story to help a student remember the concept for the problem:\n\n"
            f"Problem: {problem_title}\nConcepts: {concepts_str}\nDifficulty: {difficulty}\n\n"
            "Make it relatable, use imagery and a small character, and ensure it's easy to recall."
        )
    }
    try:
        model = (st.secrets.get("OPENAI_MODEL") if hasattr(st, "secrets") else None) or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
        text = openai_chat_completion([system, human], model=model, max_tokens=220)
        return text
    except Exception as e:
        err = str(e).lower()
        if "rate" in err or "rate_limit" in err:
            return "⚠️ Rate limit reached. Please try again shortly."
        if "OPENAI_API_KEY_MISSING" in err:
            return (
                "⚠️ OpenAI API Key required. Add OPENAI_API_KEY to `.streamlit/secrets.toml` or set the environment variable."
            )
        return f"⚠️ Story generation failed: {str(e)[:180]}"

def generate_analogy(problem_title: str, concepts: List[str]) -> str:
    """Generate a short (2-3 sentence) clear analogy."""
    concepts_str = ", ".join(concepts) if concepts else "the concept"
    system = {
        "role": "system",
        "content": "You are an expert who crafts concise analogies using everyday objects to explain programming concepts clearly."
    }
    human = {
        "role": "user",
        "content": (
            f"Create a SIMPLE, clear analogy (2-3 sentences) that explains {problem_title} using everyday objects/situations.\n"
            f"Concepts: {concepts_str}\n"
            "Keep it crystal clear and easy to visualize."
        )
    }
    try:
        model = (st.secrets.get("OPENAI_MODEL") if hasattr(st, "secrets") else None) or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
        text = openai_chat_completion([system, human], model=model, max_tokens=140)
        return text
    except Exception as e:
        err = str(e).lower()
        if "rate" in err:
            return "⚠️ Rate limit reached. Please try again shortly."
        if "OPENAI_API_KEY_MISSING" in err:
            return "⚠️ OpenAI API Key required. Add OPENAI_API_KEY to `.streamlit/secrets.toml` or set the environment variable."
        return "⚠️ Analogy generation temporarily unavailable."

def generate_learning_tip(problem_title: str, concepts: List[str], difficulty: str) -> str:
    """Generate a practical 1-2 sentence tip focused on common mistakes or key insights."""
    concepts_str = ", ".join(concepts) if concepts else ""
    system = {
        "role": "system",
        "content": "You are a concise coding mentor who provides practical, actionable tips in 1-2 sentences."
    }
    human = {
        "role": "user",
        "content": (
            f"Provide ONE specific, actionable learning tip (1-2 sentences) for Problem: {problem_title}\n"
            f"Concepts: {concepts_str}\nDifficulty: {difficulty}\n"
            "Focus on common pitfalls or a key insight that helps mastery."
        )
    }
    try:
        model = (st.secrets.get("OPENAI_MODEL") if hasattr(st, "secrets") else None) or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
        text = openai_chat_completion([system, human], model=model, max_tokens=120)
        return text
    except Exception:
        return "💡 Pro tip: Dry-run the algorithm with a small example, and pay attention to edge cases and off-by-one errors."

def generate_learning_content(problem_title: str, concepts: List[str], difficulty: str) -> Dict[str, Any]:
    """
    Public function used by app.py
    Returns dict: {story, analogy, learning_tip, has_error}
    """
    # Quick check for API key
    api_key = get_openai_api_key()
    if not api_key:
        # graceful fallback (non-AI)
        story = (
            f"⚠️ OpenAI API Key Required!\n\n"
            "To unlock AI-powered stories and analogies:\n"
            "1. Get an OpenAI API key (or set OPENAI_API_KEY in your environment).\n"
            "2. Add it to `.streamlit/secrets.toml` as: OPENAI_API_KEY = \"your_key_here\"\n"
            "3. Restart the app."
        )
        analogy = "Without the API key, the app shows static guidance. Add your key to get instant AI content."
        tip = "💡 Tip: Practice this problem manually and write out the state transitions."
        return {"story": story, "analogy": analogy, "learning_tip": tip, "has_error": True}

    # Generate content using OpenAI
    story = generate_story(problem_title, concepts, difficulty)
    analogy = generate_analogy(problem_title, concepts)
    learning_tip = generate_learning_tip(problem_title, concepts, difficulty)

    # If any of the outputs indicate API-key missing or error, mark has_error True
    has_error = any(
        isinstance(x, str) and x.startswith("⚠️") for x in (story, analogy)
    )

    return {
        "story": story,
        "analogy": analogy,
        "learning_tip": learning_tip,
        "has_error": has_error
    }
