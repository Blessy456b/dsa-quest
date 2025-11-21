# backend_user.py
"""
Central Supabase backend + auth + per-user data access.

- Caches Supabase client with st.cache_resource
- Auth helpers: sign_in, sign_up, sign_out, get_current_user
- UserBackend: wraps users_progress, content_cache, user_settings
"""

import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List

import streamlit as st
from supabase import create_client, Client

logger = logging.getLogger("backend_user")
logging.basicConfig(level=logging.INFO)


# ------------------------- #
# Supabase client (cached)  #
# ------------------------- #

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
            "Set SUPABASE_URL and SUPABASE_ANON_KEY in .streamlit/secrets.toml "
            "or environment variables."
        )

    return create_client(url, key)


supabase: Client = get_supabase_client()


# ------------------------- #
# Auth helpers              #
# ------------------------- #

def sign_in(email: str, password: str):
    """Sign in and store user in session_state."""
    try:
        res = supabase.auth.sign_in_with_password(
            {"email": email, "password": password}
        )
        data = getattr(res, "data", {}) or {}
        user = data.get("user") or getattr(res, "user", None)
        if user:
            st.session_state["user"] = {
                "id": user.get("id"),
                "email": user.get("email"),
                "raw": user,
            }
        return res
    except Exception as e:
        logger.exception("sign_in failed")
        return type("E", (), {"error": str(e)})


def sign_up(email: str, password: str, username: Optional[str] = None):
    """Sign up new user + create default rows."""
    try:
        res = supabase.auth.sign_up({"email": email, "password": password})
        data = getattr(res, "data", {}) or {}
        user = data.get("user") or getattr(res, "user", None)
        if user:
            uid = user.get("id")

            # default users_progress
            try:
                supabase.table("users_progress").insert(
                    {"user_id": uid}
                ).execute()
            except Exception:
                logger.exception("Failed creating users_progress row")

            # default user_settings
            try:
                supabase.table("user_settings").insert(
                    {"user_id": uid}
                ).execute()
            except Exception:
                logger.exception("Failed creating user_settings row")

            # optional profile username
            if username:
                try:
                    supabase.table("users").insert(
                        {"id": uid, "username": username}
                    ).execute()
                except Exception:
                    logger.exception("Failed inserting into users profile table")

        return res
    except Exception as e:
        logger.exception("sign_up failed")
        return type("E", (), {"error": str(e)})


def sign_out():
    """Sign out + wipe Streamlit session user info."""
    try:
        supabase.auth.sign_out()
    except Exception:
        pass

    for k in list(st.session_state.keys()):
        if k != "_rerun_count":
            del st.session_state[k]


def get_current_user() -> Optional[Dict[str, Any]]:
    """Returns currently authenticated user from session_state."""
    u = st.session_state.get("user")
    if u:
        return u

    try:
        info = supabase.auth.get_user()
        data = getattr(info, "data", {}) or {}
        user = data.get("user") or getattr(info, "user", None)
        if user:
            st.session_state["user"] = {
                "id": user.get("id"),
                "email": user.get("email"),
                "raw": user,
            }
            return st.session_state["user"]
    except Exception:
        pass
    return None


# ------------------------- #
# UserBackend wrapper       #
# ------------------------- #

class UserBackend:
    """
    Handles database reads/writes for:
    - users_progress
    - content_cache
    - user_settings
    """

    def __init__(self, user_id: str):
        if not user_id:
            raise RuntimeError("user_id required")
        self.user_id = user_id
        self.supabase = supabase

    # -------- progress table --------
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
                # create default row
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
        completed_problems: Optional[List[int]] = None,
        total_xp: Optional[int] = None,
        current_streak: Optional[int] = None,
        badges: Optional[List[str]] = None,
        last_activity: Optional[str] = None,
    ):
        payload: Dict[str, Any] = {
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

    def save_progress(self, payload: Dict[str, Any]):
        body = dict(payload)
        body["updated_at"] = datetime.utcnow().isoformat()
        try:
            self.supabase.table("users_progress").update(body).eq(
                "user_id", self.user_id
            ).execute()
        except Exception:
            logger.exception("save_progress failed")

    # -------- content cache --------
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
        self,
        problem_id: int,
        story: str = None,
        analogy: str = None,
        learning_tip: str = None,
        raw_llm_response: str = None,
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
                payload,
                on_conflict="user_id,problem_id"
            ).execute()
        except Exception:
            logger.exception("upsert_content_cache failed")

    # -------- user settings --------
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
            if data:
                return data.get("groq_api_key")
            return None
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
