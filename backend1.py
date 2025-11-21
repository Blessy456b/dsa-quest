# backend.py
from supabase import create_client, Client
from typing import Optional, Dict, Any
import streamlit as st
from datetime import date, datetime

SUPABASE_URL = st.secrets.get("SUPABASE_URL") or st.secrets.get("SUPABASE_URL", None) or None
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY", None) or None

if not SUPABASE_URL or not SUPABASE_KEY:
    # allow fallback to environment variables (if running outside Streamlit secret system)
    import os
    SUPABASE_URL = SUPABASE_URL or os.getenv("SUPABASE_URL")
    SUPABASE_KEY = SUPABASE_KEY or os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase URL/Key not configured. Put them in .streamlit/secrets.toml or env vars.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------- Auth helpers ----------
def sign_up(email: str, password: str) -> Dict[str, Any]:
    """Sign up a new user. Returns supabase response dict."""
    res = supabase.auth.sign_up({"email": email, "password": password})
    return res

def sign_in(email: str, password: str) -> Dict[str, Any]:
    """Sign in and store user in session_state (id, access_token)."""
    res = supabase.auth.sign_in_with_password({"email": email, "password": password})
    # res contains 'data' with 'user' and 'session'
    return res

def sign_out():
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    st.session_state.pop("user", None)


def get_current_user() -> Optional[Dict[str, Any]]:
    """Return user dict stored in st.session_state if signed in"""
    return st.session_state.get("user")


# ---------- Backend data helpers ----------
class UserBackend:
    """Light wrapper for per-user DB operations."""

    def __init__(self, user_id: str):
        self.user_id = user_id

    # ---------- progress ----------
    def load_progress(self) -> Dict[str, Any]:
        resp = supabase.table("users_progress").select("*").eq("user_id", self.user_id).maybe_single().execute()
        data = resp.data
        if not data:
            # create initial row
            payload = {"user_id": self.user_id}
            supabase.table("users_progress").insert(payload).execute()
            return {
                "user_id": self.user_id,
                "completed_problems": [],
                "total_xp": 0,
                "current_streak": 0,
                "badges": [],
                "last_activity": None
            }
        # normalize fields
        return {
            "user_id": data.get("user_id"),
            "completed_problems": data.get("completed_problems") or [],
            "total_xp": int(data.get("total_xp") or 0),
            "current_streak": int(data.get("current_streak") or 0),
            "badges": data.get("badges") or [],
            "last_activity": data.get("last_activity")
        }

    def save_progress(self, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload["updated_at"] = datetime.utcnow().isoformat()
        supabase.table("users_progress").update(payload).eq("user_id", self.user_id).execute()

    # Convenience methods
    def add_completed_problem(self, problem_id: int, xp: int) -> None:
        row = self.load_progress()
        completed = set(row["completed_problems"])
        if problem_id not in completed:
            completed.add(problem_id)
            row["completed_problems"] = list(completed)
            row["total_xp"] = row.get("total_xp", 0) + int(xp)
            row["last_activity"] = date.today().isoformat()
            self.save_progress(row)

    # ---------- content cache ----------
    def upsert_content_cache(self, problem_id: int, story: str = None, analogy: str = None, learning_tip: str = None, raw_llm_response: str = None) -> None:
        now = datetime.utcnow().isoformat()
        payload = {
            "user_id": self.user_id,
            "problem_id": int(problem_id),
            "story": story,
            "analogy": analogy,
            "learning_tip": learning_tip,
            "raw_llm_response": raw_llm_response,
            "updated_at": now
        }
        # try update first
        supabase.table("content_cache").upsert(payload, on_conflict=("user_id", "problem_id")).execute()

    def get_content_cache(self, problem_id: int):
        resp = supabase.table("content_cache").select("*").eq("user_id", self.user_id).eq("problem_id", problem_id).maybe_single().execute()
        return resp.data

    # ---------- user settings ----------
    def set_groq_api_key(self, groq_key: str):
        payload = {"user_id": self.user_id, "groq_api_key": groq_key}
        supabase.table("user_settings").upsert(payload, on_conflict="user_id").execute()

    def get_groq_api_key(self) -> Optional[str]:
        resp = supabase.table("user_settings").select("groq_api_key").eq("user_id", self.user_id).maybe_single().execute()
        if resp.data:
            return resp.data.get("groq_api_key")
        return None

