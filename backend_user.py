# backend_user.py
"""
Central Supabase backend + auth + per-user data access.

Compatible with Supabase Python SDK v2.24.0
- Correct handling of AuthResponse.user object (user.id, user.email)
- Unified UserBackend for app.py + progress_tracker.py
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import streamlit as st
from supabase import create_client, Client

logger = logging.getLogger("backend_user")
logging.basicConfig(level=logging.INFO)


# ============================================================
# Supabase client (cached)
# ============================================================

@st.cache_resource
def get_supabase_client() -> Client:
    url = st.secrets.get("SUPABASE_URL")
    key = (
        st.secrets.get("SUPABASE_ANON_KEY")
        or st.secrets.get("SUPABASE_KEY")
    )

    if not url:
        url = os.getenv("SUPABASE_URL")
    if not key:
        key = os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError(
            "Supabase URL/Key missing. "
            "Set SUPABASE_URL and SUPABASE_ANON_KEY in .streamlit/secrets.toml"
        )

    return create_client(url, key)


supabase: Client = get_supabase_client()


# ============================================================
# Auth helpers (Supabase v2.x)
# ============================================================

def sign_in(email: str, password: str):
    """Login user using password auth and store session in Streamlit."""
    try:
        res = supabase.auth.sign_in_with_password(
            {"email": email, "password": password}
        )
        data = getattr(res, "data", None)
        user = getattr(data, "user", None)

        if user:  # Supabase v2: user is object, not dict
            st.session_state["user"] = {
                "id": user.id,
                "email": user.email,
                "raw": user,
            }

        return res

    except Exception as e:
        logger.exception("sign_in failed")
        return type("E", (), {"error": str(e)})


def sign_up(email: str, password: str, username: Optional[str] = None):
    """Create account + insert default rows in tables."""
    try:
       # res = supabase.auth.sign_up({"email": email, "password": password}) changed on nov21 16:27
        res = supabase.auth.sign_up(
    {
        "email": email,
        "password": password,
        "options": {
            "email_redirect_to": f"{st.secrets['SITE_URL']}"
        }
    }
)
        data = getattr(res, "data", None)
        user = getattr(data, "user", None)

        if user:
            uid = user.id  # FIX: use attribute, NOT dict get()

            # Create default progress row
            try:
                supabase.table("users_progress").insert(
                    {"user_id": uid}
                ).execute()
            except Exception:
                logger.exception("Failed creating users_progress row")

            # Create default settings row
            try:
                supabase.table("user_settings").insert(
                    {"user_id": uid}
                ).execute()
            except Exception:
                logger.exception("Failed creating user_settings row")

            # Optional profile username
            if username:
                try:
                    supabase.table("users").insert(
                        {"id": uid, "username": username}
                    ).execute()
                except Exception:
                    logger.warning("Users table does not exist; skipping username.")

        return res

    except Exception as e:
        logger.exception("sign_up failed")
        return type("E", (), {"error": str(e)})


def sign_out():
    """Logout + clear Streamlit user state."""
    try:
        supabase.auth.sign_out()
    except Exception:
        pass

    for k in list(st.session_state.keys()):
        if k != "_rerun_count":
            del st.session_state[k]


def get_current_user() -> Optional[Dict[str, Any]]:
    """Return authenticated user from session_state or Supabase session."""
    
    # 1. Already logged in during this Streamlit session
    user = st.session_state.get("user")
    if user:
        return user

    # 2. Check Supabase session
    try:
        session = supabase.auth.get_session()
        if session and session.user:
            user_obj = session.user
            st.session_state["user"] = {
                "id": user_obj.id,
                "email": user_obj.email,
                "raw": user_obj,
            }
            return st.session_state["user"]
    except Exception:
        pass

    return None


# ============================================================
# UserBackend (Progress, Content Cache, Settings)
# ============================================================

class UserBackend:
    """Per-user Supabase database operations."""

    def __init__(self, user_id: str):
        if not user_id:
            raise RuntimeError("User ID required")
        self.user_id = user_id
        self.supabase = supabase

    # ---------------- Progress ----------------

    def load_progress(self) -> Dict[str, Any]:
        try:
            res = (
                self.supabase.table("users_progress")
                .select("*")
                .eq("user_id", self.user_id)
                .maybe_single()
                .execute()
            )
            data = res.data

            if not data:
                defaults = {
                    "user_id": self.user_id,
                    "completed_problems": [],
                    "total_xp": 0,
                    "current_streak": 0,
                    "badges": [],
                    "last_activity": None,
                }
                self.supabase.table("users_progress").insert(defaults).execute()
                return defaults

            return {
                "user_id": data.get("user_id"),
                "completed_problems": data.get("completed_problems") or [],
                "total_xp": int(data.get("total_xp") or 0),
                "current_streak": int(data.get("current_streak") or 0),
                "badges": data.get("badges") or [],
                "last_activity": data.get("last_activity"),
            }

        except Exception:
            logger.exception("load_progress failed")
            return {
                "user_id": self.user_id,
                "completed_problems": [],
                "total_xp": 0,
                "current_streak": 0,
                "badges": [],
                "last_activity": None,
            }

    def upsert_progress(
        self,
        completed_problems=None,
        total_xp=None,
        current_streak=None,
        badges=None,
        last_activity=None,
    ):
        payload = {
            "user_id": self.user_id,
            "updated_at": datetime.utcnow().isoformat(),
        }
        if completed_problems is not None:
            payload["completed_problems"] = completed_problems
        if total_xp is not None:
            payload["total_xp"] = int(total_xp)
        if current_streak is not None:
            payload["current_streak"] = int(current_streak)
        if badges is not None:
            payload["badges"] = badges
        if last_activity is not None:
            payload["last_activity"] = last_activity

        try:
            self.supabase.table("users_progress").upsert(
                payload, on_conflict="user_id"
            ).execute()
        except Exception:
            logger.exception("upsert_progress failed")
            
    def save_progress(self, payload: Dict[str, Any]):   # ✅ FIXED — OUTSIDE upsert_progress
        """Compatibility wrapper for ProgressTracker.save()."""
        try:
            self.upsert_progress(
                completed_problems=payload.get("completed_problems"),
                total_xp=payload.get("total_xp"),
                current_streak=payload.get("current_streak"),
                badges=payload.get("badges"),
                last_activity=payload.get("last_activity"),
            )
        except Exception:
            logger.exception("save_progress failed")
    # ---------------- Content Cache ----------------

    def get_content_cache(self, problem_id: int):
        try:
            res = (
                self.supabase.table("content_cache")
                .select("*")
                .eq("user_id", self.user_id)
                .eq("problem_id", problem_id)
                .maybe_single()
                .execute()
            )
            return res.data
        except Exception:
            logger.exception("get_content_cache failed")
            return None

    def upsert_content_cache(
        self, problem_id, story=None, analogy=None, learning_tip=None, raw_llm_response=None
    ):
        payload = {
            "user_id": self.user_id,
            "problem_id": int(problem_id),
            "story": story,
            "analogy": analogy,
            "learning_tip": learning_tip,
            "raw_llm_response": raw_llm_response,
            "updated_at": datetime.utcnow().isoformat(),
        }
        try:
            self.supabase.table("content_cache").upsert(
                payload, on_conflict="user_id,problem_id"
            ).execute()
        except Exception:
            logger.exception("upsert_content_cache failed")

    # ---------------- Settings ----------------

    def get_groq_api_key(self) -> Optional[str]:
        try:
            res = (
                self.supabase.table("user_settings")
                .select("groq_api_key")
                .eq("user_id", self.user_id)
                .maybe_single()
                .execute()
            )
            data = res.data
            return data.get("groq_api_key") if data else None
        except Exception:
            logger.exception("get_groq_api_key failed")
            return None

    def set_groq_api_key(self, key: str):
        payload = {
            "user_id": self.user_id,
            "groq_api_key": key,
            "updated_at": datetime.utcnow().isoformat(),
        }
        try:
            self.supabase.table("user_settings").upsert(
                payload, on_conflict="user_id"
            ).execute()
        except Exception:
            logger.exception("set_groq_api_key failed")
