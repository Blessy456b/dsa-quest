import os
import json
from groq import Groq


def evaluate_solution(problem_title, problem_description, user_code, language):
    """
    Standalone LLM-based evaluator for DSA solutions.
    Compatible with Groq SDK 2024–2025 response format.
    """

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"error": "Groq API key not found in environment variables."}

    client = Groq(api_key=api_key)

    prompt = f"""
You are an expert competitive programming judge.

Evaluate the user's solution for the following DSA problem:

### Problem Title:
{problem_title}

### Description / Concepts:
{problem_description}

### User's Code ({language}):
{user_code}

### Your evaluation must be valid JSON ONLY:
{{
  "correctness": "...",
  "logic_review": "...",
  "edge_cases": "...",
  "time_complexity": "...",
  "score": "X/10"
}}

RULES:
- DO NOT execute the code.
- Review based on algorithmic logic only.
- Score must be an integer 1–10.
- Output must be PURE JSON. No markdown, no backticks.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=700,
        )

        # NEW: Correct Groq response format (2024–25)
        raw = response.choices[0].message.content.strip()

        # Parse JSON safely
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {
                "error": "Invalid JSON from LLM",
                "raw_response": raw
            }

    except Exception as e:
        return {"error": str(e)}
