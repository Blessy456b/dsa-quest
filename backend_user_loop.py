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
    try:
        res = supabase.auth.sign_in_with_password(
            {"email": email, "password": password}
        )

        data = getattr(res, "data", None)
        user = getattr(data, "user", None)

        if user:
            # Save user
            st.session_state["user"] = {
                "id": user.id,
                "email": user.email,
                "raw": user,
            }

            # Save session tokens (CRITICAL)
            if data.session:
                st.session_state["auth_session"] = {
                    "access_token": data.session.access_token,
                    "refresh_token": data.session.refresh_token,
                }

        return res

    except Exception as e:
        logger.exception("sign_in failed")
        return type("E", (), {"error": str(e)})



def get_current_user():
    # 1: If already in memory
    if "user" in st.session_state and st.session_state["user"]:
        return st.session_state["user"]

    # 2: Restore session if available (CRITICAL)
    if "auth_session" in st.session_state:
        session = st.session_state["auth_session"]
        supabase.auth.set_session(
            session["access_token"],
            session["refresh_token"]
        )

    # 3: Ask Supabase for user
    try:
        info = supabase.auth.get_user()
        data = getattr(info, "data", None)
        user = getattr(data, "user", None)

        if user:
            st.session_state["user"] = {
                "id": user.id,
                "email": user.email,
                "raw": user
            }
            return st.session_state["user"]
    except:
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
