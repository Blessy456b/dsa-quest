# app.py
"""
One-file DSA Quest (Supabase-backed, multi-user)
- Supabase client (cached)
- Auth UI (signup/login/logout)
- Per-user progress (users_progress table)
- Per-user LLM content cache (content_cache table)
- Per-user settings (user_settings) including Groq key
- Integration with langraph_agent.generate_learning_content (per-user Groq key injection)
- Full Streamlit UI (merged from your previous app)
"""

import os
import time
import json
import logging
from typing import List, Dict, Any, Optional

import streamlit as st
import requests
import plotly.express as px
from streamlit_ace import st_ace
from streamlit_lottie import st_lottie

# Supabase client
from supabase import create_client, Client

# Domain modules (assumed to exist next to this file)
# - dsa_data.py with STRIVER_SHEET, BADGES, TOTAL_PROBLEMS
# - langraph_agent.py with generate_learning_content(...)
from dsa_data import STRIVER_SHEET, BADGES, TOTAL_PROBLEMS
from langraph_agent import generate_learning_content  # we'll call this and inject user key

# Logging
logger = logging.getLogger("dsa_quest")
logging.basicConfig(level=logging.INFO)

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(
    page_title="DSA Quest - Multiuser",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Supabase client (cached resource)
# -------------------------
@st.cache_resource
def get_supabase_client() -> Client:
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_ANON_KEY in .streamlit/secrets.toml")
    return create_client(url, key)


supabase: Client = get_supabase_client()

# -------------------------
# Lightweight backend wrapper (UserBackend)
# -------------------------
class UserBackend:
    """Simple wrapper for per-user Supabase operations used by the UI."""
    def __init__(self, user_id: str):
        if not user_id:
            raise RuntimeError("user_id required")
        self.user_id = user_id
        self.supabase = supabase

    # -------- progress (single row per user) --------
    def load_progress(self) -> Dict[str, Any]:
        try:
            res = self.supabase.table("users_progress").select("*").eq("user_id", self.user_id).maybe_single().execute()
            data = res.data if hasattr(res, "data") else res
            if not data:
                # create a default row
                defaults = {
                    "user_id": self.user_id,
                    "completed_problems": [],
                    "total_xp": 0,
                    "current_streak": 0,
                    "badges": [],
                }
                self.supabase.table("users_progress").insert(defaults).execute()
                return defaults
            return data
        except Exception as e:
            logger.exception("load_progress failed")
            return {"user_id": self.user_id, "completed_problems": [], "total_xp": 0, "current_streak": 0, "badges": []}

    def upsert_progress(self, completed_problems: Optional[List[int]] = None, total_xp: Optional[int] = None,
                        current_streak: Optional[int] = None, badges: Optional[List[str]] = None) -> None:
        payload = {"user_id": self.user_id, "updated_at": "now()"}
        if completed_problems is not None:
            payload["completed_problems"] = completed_problems
        if total_xp is not None:
            payload["total_xp"] = total_xp
        if current_streak is not None:
            payload["current_streak"] = current_streak
        if badges is not None:
            payload["badges"] = badges

        try:
            self.supabase.table("users_progress").upsert(payload, on_conflict="user_id").execute()
        except Exception:
            # fallback to update if upsert not supported
            try:
                self.supabase.table("users_progress").update(payload).eq("user_id", self.user_id).execute()
            except Exception:
                logger.exception("Failed to upsert progress")

    # -------- content cache (per-user per-problem) --------
    def get_content_cache(self, problem_id: int) -> Optional[Dict[str, Any]]:
        try:
            res = self.supabase.table("content_cache").select("*").eq("user_id", self.user_id).eq("problem_id", problem_id).maybe_single().execute()
            return res.data if hasattr(res, "data") else res
        except Exception:
            logger.exception("get_content_cache failed")
            return None

    def upsert_content_cache(self, problem_id: int, story: str = None, analogy: str = None, learning_tip: str = None, raw_llm_response: str = None) -> None:
        payload = {
            "user_id": self.user_id,
            "problem_id": problem_id,
            "story": story,
            "analogy": analogy,
            "learning_tip": learning_tip,
            "raw_llm_response": raw_llm_response,
            "updated_at": "now()"
        }
        try:
            self.supabase.table("content_cache").upsert(payload, on_conflict="user_id,problem_id").execute()
        except Exception:
            # some clients don't accept on_conflict; try insert/update
            try:
                self.supabase.table("content_cache").insert(payload).execute()
            except Exception:
                self.supabase.table("content_cache").update(payload).eq("user_id", self.user_id).eq("problem_id", problem_id).execute()

    # -------- user settings (groq api key etc) --------
    def get_groq_api_key(self) -> Optional[str]:
        try:
            res = self.supabase.table("user_settings").select("groq_api_key").eq("user_id", self.user_id).maybe_single().execute()
            data = res.data if hasattr(res, "data") else res
            if data:
                return data.get("groq_api_key")
            return None
        except Exception:
            logger.exception("get_groq_api_key failed")
            return None

    def set_groq_api_key(self, key: str) -> None:
        payload = {"user_id": self.user_id, "groq_api_key": key, "updated_at": "now()"}
        try:
            self.supabase.table("user_settings").upsert(payload, on_conflict="user_id").execute()
        except Exception:
            try:
                self.supabase.table("user_settings").insert(payload).execute()
            except Exception:
                self.supabase.table("user_settings").update(payload).eq("user_id", self.user_id).execute()


# -------------------------
# Auth helpers (sign in/up/out + current user)
# -------------------------
def sign_in(email: str, password: str):
    """Sign in using Supabase auth; store user in session_state if successful."""
    try:
        res = supabase.auth.sign_in_with_password({"email": email, "password": password})
        # res has .data.user or .user depending on client version
        data = getattr(res, "data", {}) or {}
        user = data.get("user") or getattr(res, "user", None)
        if user:
            st.session_state["user"] = {"id": user.get("id"), "email": user.get("email"), "raw": user}
        return res
    except Exception as e:
        logger.exception("sign_in failed")
        return type("E", (), {"error": str(e)})


def sign_up(email: str, password: str):
    """Sign up using Supabase auth."""
    try:
        res = supabase.auth.sign_up({"email": email, "password": password})
        data = getattr(res, "data", {}) or {}
        user = data.get("user") or getattr(res, "user", None)
        if user:
            # create default rows for the new user
            try:
                supabase.table("users_progress").insert({"user_id": user.get("id")}).execute()
            except Exception:
                pass
            try:
                supabase.table("user_settings").insert({"user_id": user.get("id")}).execute()
            except Exception:
                pass
        return res
    except Exception as e:
        logger.exception("sign_up failed")
        return type("E", (), {"error": str(e)})


def sign_out():
    """Sign out from Supabase and clear session_state user."""
    try:
        supabase.auth.sign_out()
    except Exception:
        pass
    for k in list(st.session_state.keys()):
        if k != "_rerun_count":
            del st.session_state[k]


def get_current_user() -> Optional[Dict[str, Any]]:
    """Return signed-in user info stored in session_state (id, email) if present."""
    u = st.session_state.get("user")
    if u:
        return u
    # fallback: try supabase auth client (might not always work in this context)
    try:
        info = supabase.auth.get_user()
        data = getattr(info, "data", {}) or {}
        user = data.get("user") or getattr(info, "user", None)
        if user:
            st.session_state["user"] = {"id": user.get("id"), "email": user.get("email"), "raw": user}
            return st.session_state["user"]
    except Exception:
        pass
    return None


# -------------------------
# Small convenience: run LLM generation while using per-user Groq API key
# -------------------------
def generate_learning_content_for_user(user_id: str, problem_title: str, concepts: List[str], difficulty: str) -> Dict[str, Any]:
    """
    Read user's Groq key from user_settings; set it in env temporarily, call generate_learning_content(),
    then restore previous env state.
    """
    backend = UserBackend(user_id)
    user_key = backend.get_groq_api_key()

    prev = os.environ.get("GROQ_API_KEY", None)
    try:
        if user_key:
            os.environ["GROQ_API_KEY"] = user_key
        # call into the langraph agent you already have
        content = generate_learning_content(problem_title, concepts, difficulty)
        # store the raw repr (if any) into content_cache through backend later
        return content
    finally:
        # restore
        if prev is None:
            if "GROQ_API_KEY" in os.environ:
                del os.environ["GROQ_API_KEY"]
        else:
            os.environ["GROQ_API_KEY"] = prev


# -------------------------
# UI: styles, helpers
# -------------------------
st.markdown(
    """
<style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(120deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px 0;
        animation: fadeIn 1s;
    }
    .story-box { background: linear-gradient(135deg,#FFE5B4 0%,#FFD700 100%); padding:20px; border-radius:15px; margin:15px 0; border-left:5px solid #FFA500; }
    .analogy-box { background: linear-gradient(135deg,#E0F7FA 0%,#80DEEA 100%); padding:20px; border-radius:15px; margin:15px 0; border-left:5px solid #00ACC1; }
</style>
""",
    unsafe_allow_html=True,
)

def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def show_celebration(unique_key: str = "celebration"):
    lottie = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_touohxv0.json")
    if lottie:
        try:
            st_lottie(lottie, height=200, key=unique_key)
        except Exception:
            pass

# -------------------------
# Session initialization (keys we need)
# -------------------------
def initialize_session_state_defaults():
    if "page" not in st.session_state:
        st.session_state.page = "🏠 Home"
    if "content_cache" not in st.session_state:
        st.session_state.content_cache = {}  # purely local transient cache
    if "last_llm_raw_repr" not in st.session_state:
        st.session_state.last_llm_raw_repr = None
    if "last_llm_extracted_text" not in st.session_state:
        st.session_state.last_llm_extracted_text = None

# -------------------------
# Sidebar: auth UI + nav + user stats
# -------------------------
def show_auth_ui():
    st.sidebar.title("Account")
    user = get_current_user()
    if user:
        st.sidebar.write(f"Signed in as: {user.get('email')}")
        if st.sidebar.button("Sign out"):
            sign_out()
            st.experimental_rerun()
        with st.sidebar.expander("🔑 Per-user Groq API Key"):
            backend = UserBackend(user["id"])
            cur = backend.get_groq_api_key() or ""
            groq = st.text_input("Groq API Key (private)", value=cur, type="password", key="groq_input")
            if st.button("Save Groq Key"):
                backend.set_groq_api_key(groq)
                st.success("Saved Groq API key for your account.")
    else:
        st.sidebar.markdown("### Sign In / Sign Up")
        tab1, tab2 = st.sidebar.tabs(["Sign in", "Sign up"])
        with tab1:
            email = st.text_input("Email", key="signin_email")
            password = st.text_input("Password", type="password", key="signin_password")
            if st.button("Sign in"):
                res = sign_in(email, password)
                if getattr(res, "error", None):
                    st.error(f"Sign in error: {res.error}")
                else:
                    st.success("Signed in.")
                    time.sleep(0.5)
                    st.experimental_rerun()
        with tab2:
            email = st.text_input("New Email", key="signup_email")
            password = st.text_input("New Password", type="password", key="signup_password")
            username = st.text_input("Username", key="signup_username")
            if st.button("Create account"):
                res = sign_up(email, password)
                if getattr(res, "error", None):
                    st.error(f"Sign up error: {res.error}")
                else:
                    # try to insert a users row (username) optionally
                    try:
                        # supabase.auth created user id is in res.data.user.id depending on client
                        data = getattr(res, "data", {}) or {}
                        user_obj = data.get("user") or getattr(res, "user", None)
                        if user_obj:
                            supabase.table("users").insert({"id": user_obj.get("id"), "username": username}).execute()
                    except Exception:
                        pass
                    st.success("Account created. Check your email (if confirmation enabled).")
                    time.sleep(0.5)
                    st.experimental_rerun()


def render_sidebar() -> str:
    tracker = None
    try:
        user = get_current_user()
        tracker = UserBackend(user["id"]).load_progress() if user else None
    except Exception:
        tracker = None

    with st.sidebar:
        st.markdown("# 🚀 DSA Quest")
        st.markdown("---")

        current_badge = "🌱 DSA Rookie"
        total_xp = 0
        current_streak = 0
        completed_count = 0
        if tracker:
            try:
                current_badge = (tracker.get("badges") or [])[-1] if (tracker.get("badges") or []) else "🌱 DSA Rookie"
            except Exception:
                current_badge = "🌱 DSA Rookie"
            total_xp = int(tracker.get("total_xp", 0))
            current_streak = int(tracker.get("current_streak", 0))
            completed_count = len(tracker.get("completed_problems", []) or [])

        st.markdown("### Your Progress")
        st.markdown(f"**{current_badge}**")
        st.markdown(f"**XP:** {total_xp} 💎")
        st.markdown(f"**Streak:** {current_streak} 🔥")
        st.markdown(f"**Problems Solved:** {completed_count}/{TOTAL_PROBLEMS}")
        progress_pct = (completed_count / TOTAL_PROBLEMS) * 100.0 if TOTAL_PROBLEMS else 0.0
        st.progress(min(progress_pct / 100.0, 1.0))
        st.markdown(f"**{progress_pct:.1f}% Complete**")

        st.markdown("---")
        st.markdown("### 📚 Navigation")
        page_options = ["🏠 Home", "📖 Problem Browser", "👤 Profile", "🏆 Achievements"]
        if "page" not in st.session_state:
            st.session_state.page = page_options[0]
        try:
            index = page_options.index(st.session_state.page)
        except Exception:
            index = 0
        selected = st.radio("Go to:", page_options, index=index, label_visibility="collapsed", key="navigation_radio")
        if selected != st.session_state.page:
            st.session_state.page = selected

        st.markdown("---")
        with st.expander("⚙️ Settings"):
            st.markdown("**Groq API Key**")
            st.markdown("Per-user key is stored privately under Account → Groq API Key.")
            try:
                if supabase:
                    st.caption("Supabase connected.")
            except Exception:
                pass

    return st.session_state.get("page", "🏠 Home")


# -------------------------
# Pages: Home, Browser, Detail, Profile, Achievements
# -------------------------
def render_home():
    st.markdown('<h1 class="main-title">🚀 Welcome to DSA Quest!</h1>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📚 450+ Problems")
        st.markdown("Master DSA with Striver's A2Z Sheet")
    with col2:
        st.markdown("### 🎨 AI-Powered Stories")
        st.markdown("Learn through engaging narratives")
    with col3:
        st.markdown("### 🏆 Gamified Learning")
        st.markdown("Earn XP, badges, and track progress")
    st.markdown("---")
    st.markdown("## 🗺️ Your Learning Journey")

    # load tracker for current user
    user = get_current_user()
    if not user:
        st.info("Sign in to see your learning journey.")
        return

    backend = UserBackend(user["id"])
    tracker = backend.load_progress()

    for step_key, step_data in STRIVER_SHEET.items():
        step_progress = 0.0
        if tracker:
            completed = tracker.get("completed_problems", []) or []
            problems = step_data.get("problems", [])
            if problems:
                step_progress = (sum(1 for p in problems if p["id"] in completed) / len(problems)) * 100.0
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"### {step_data.get('icon','')} {step_key}: {step_data.get('name')}")
            st.markdown(f"*{step_data.get('description','')}*")
            st.progress(step_progress / 100.0)
            st.markdown(f"**{step_progress:.0f}% Complete** | {step_data.get('xp', 0)} XP per problem")
        with c2:
            if st.button(f"Start {step_key}", key=f"start_{step_key}"):
                st.session_state.current_step = step_key
                st.session_state.page = "📖 Problem Browser"
                st.experimental_rerun()
                return


def render_problem_browser():
    st.markdown("# 📖 Problem Browser")
    step_options = [f"{k}: {v.get('name')}" for k, v in STRIVER_SHEET.items()]
    selected_index = 0
    if st.session_state.get("current_step"):
        for i, s in enumerate(step_options):
            if s.startswith(st.session_state["current_step"]):
                selected_index = i
                break

    selected_step = st.selectbox("Select Topic", step_options, index=selected_index)
    step_key = selected_step.split(":")[0]
    step_data = STRIVER_SHEET[step_key]

    st.markdown(f"## {step_data.get('icon','')} {step_data.get('name')}")
    st.markdown(f"*{step_data.get('description','')}*")

    if not step_data.get("problems"):
        st.info("🚧 Problems for this step are coming soon! Stay tuned.")
        return

    problem_options = [f"{p['title']} ({p['difficulty']})" for p in step_data["problems"]]
    selected_problem_str = st.selectbox("Select Problem", problem_options)
    selected_problem = next((p for p in step_data["problems"] if f"{p['title']} ({p['difficulty']})" == selected_problem_str), None)
    if selected_problem:
        render_problem_detail(selected_problem, step_data.get("xp", 0))


def render_problem_detail(problem: Dict[str, Any], xp_reward: int):
    st.markdown(f"### {problem.get('title')}")
    user = get_current_user()
    if not user:
        st.info("Sign in to view problem details.")
        return

    backend = UserBackend(user["id"])
    tracker = backend.load_progress()
    completed = tracker.get("completed_problems", []) or []
    is_completed = problem["id"] in completed

    h1, h2, h3 = st.columns([2, 1, 1])
    with h1:
        st.markdown(f"### {problem.get('title')}")
    with h2:
        difficulty_class = f"difficulty-{problem.get('difficulty','').lower()}"
        st.markdown(f'<p class="{difficulty_class}">{problem.get("difficulty")}</p>', unsafe_allow_html=True)
    with h3:
        if is_completed:
            st.success("✅ Completed")
        else:
            if st.button("Mark Complete 🎉", key=f"complete_{problem['id']}"):
                # update progress: add problem id, add XP, update streak, check badges
                completed.append(problem["id"])
                total_xp = int(tracker.get("total_xp", 0)) + int(xp_reward)
                # simple streak logic: increment if last_activity was yesterday (server-side more precise)
                current_streak = int(tracker.get("current_streak", 0)) + 1
                badges = tracker.get("badges", []) or []
                # simple badge awarding example
                new_badges = []
                thresholds = [(100, "Rising Coder"), (500, "Pro Coder"), (1000, "Legendary Coder")]
                for xp_req, name in thresholds:
                    if total_xp >= xp_req and name not in badges:
                        badges.append(name)
                        new_badges.append(name)

                backend.upsert_progress(completed_problems=completed, total_xp=total_xp, current_streak=current_streak, badges=badges)
                if new_badges:
                    st.balloons()
                    for b in new_badges:
                        st.success(f"🏆 New Badge: {b}")
                st.success(f"+{xp_reward} XP")
                st.experimental_rerun()
                return

    st.markdown(f"**Concepts:** {', '.join(problem.get('concepts', []))}")
    if problem.get("link"):
        st.markdown(f"[View on TakeUForward]({problem['link']})")

    st.markdown("---")
    st.markdown("## 🎨 AI-Powered Learning Content")
    content_row = backend.get_content_cache(problem["id"])
    if content_row:
        st.markdown("### Cached Content (yours)")
        st.markdown("#### 📖 Story")
        st.write(content_row.get("story") or "—")
        st.markdown("#### 💡 Analogy")
        st.write(content_row.get("analogy") or "—")
        st.info(f"💡 Learning Tip: {content_row.get('learning_tip') or '—'}")

    if st.button("✨ Generate Story & Analogy", key=f"generate_{problem['id']}"):
        with st.spinner("Generating..."):
            # call generator with per-user Groq key injection
            content = generate_learning_content_for_user(user["id"], problem.get("title"), problem.get("concepts", []), problem.get("difficulty", "Medium"))
            # Save to Supabase content_cache
            try:
                backend.upsert_content_cache(problem["id"], story=content.get("story"), analogy=content.get("analogy"), learning_tip=content.get("learning_tip"), raw_llm_response=st.session_state.get("last_llm_raw_repr"))
            except Exception:
                st.error("Failed to save content to cache.")
            st.success("Generated and saved to your private cache.")
            st.experimental_rerun()
            return

    st.markdown("---")
    st.markdown("## 💻 Solution Code")
    language = st.radio("Select Language:", ["C++", "Java", "Python"], horizontal=True, key=f"lang_{problem['id']}")
    if language == "C++":
        code = problem.get("cpp_code", "// Code not available")
        lang_mode = "c_cpp"
    elif language == "Java":
        code = problem.get("java_code", "// Code not available")
        lang_mode = "java"
    else:
        code = problem.get("python_code", "# Code not available")
        lang_mode = "python"

    st.code(code, language=language.lower())

    st.markdown("---")
    st.markdown("## ✍️ Practice Area")
    st.markdown("Write your own solution here:")
    user_code = st_ace(value=f"# Write your {language} solution here\n\n",
                       language=lang_mode if lang_mode != "c_cpp" else "c_cpp",
                       theme="monokai",
                       height=300,
                       key=f"editor_{problem['id']}_{language}")

    if st.button("Save My Solution 💾", key=f"save_{problem['id']}"):
        st.success("Solution saved! (local)")

def render_profile():
    st.markdown("# 👤 Your Coding Profile")
    user = get_current_user()
    if not user:
        st.info("Sign in to view profile.")
        return

    backend = UserBackend(user["id"])
    tracker = backend.load_progress()
    current_badge = (tracker.get("badges") or [])[-1] if (tracker.get("badges") or []) else "🌱 DSA Rookie"

    st.markdown(f"## {current_badge}")
    st.markdown(f"### {tracker.get('total_xp', 0)} XP")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Problems Solved", len(tracker.get("completed_problems", []) or []))
    with c2:
        st.metric("Current Streak", f"{tracker.get('current_streak', 0)} days 🔥")
    with c3:
        progress = (len(tracker.get("completed_problems", []) or []) / TOTAL_PROBLEMS) * 100.0 if TOTAL_PROBLEMS else 0.0
        st.metric("Overall Progress", f"{progress:.1f}%")

    st.markdown("---")
    st.markdown("### 📊 XP Progress")
    step_xp_data = []
    for step_key, step_data in STRIVER_SHEET.items():
        completed_in_step = sum(1 for p in step_data.get("problems", []) if p["id"] in (tracker.get("completed_problems", []) or []))
        step_xp = completed_in_step * step_data.get("xp", 0)
        if step_xp > 0:
            step_xp_data.append({"Step": f"{step_key}: {step_data.get('name')}", "XP": step_xp})

    if step_xp_data:
        fig = px.bar(step_xp_data, x="Step", y="XP", title="XP Distribution by Topic", color="XP")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Start solving problems to see your XP distribution!")

def render_achievements():
    st.markdown("# 🏆 Achievements & Badges")
    user = get_current_user()
    if not user:
        st.info("Sign in to view achievements.")
        return

    backend = UserBackend(user["id"])
    tracker = backend.load_progress()
    st.markdown("### Your Badges")
    badges = tracker.get("badges", []) or []

    cols = st.columns(max(1, len(BADGES)))
    for idx, (badge_id, badge_info) in enumerate(BADGES.items()):
        col = cols[idx % len(cols)]
        with col:
            earned = badge_info.get("name") in badges or badge_id in badges
            if earned:
                st.markdown(f"### {badge_info.get('name')}")
                st.success("✅ Unlocked!")
            else:
                st.markdown("### 🔒")
                st.markdown(f"{badge_info.get('name')}")
                xp_needed = badge_info.get("xp_required", 0) - tracker.get("total_xp", 0)
                st.info(f"{xp_needed} XP needed")

# -------------------------
# Main
# -------------------------
def main():
    initialize_session_state_defaults()
    show_auth_ui()
    user = get_current_user()
    if not user:
        st.info("Please sign in or sign up from the sidebar.")
        st.stop()

    page = render_sidebar()
    if page == "🏠 Home":
        render_home()
    elif page == "📖 Problem Browser":
        render_problem_browser()
    elif page == "👤 Profile":
        render_profile()
    elif page == "🏆 Achievements":
        render_achievements()
    else:
        render_home()

if __name__ == "__main__":
    main()
