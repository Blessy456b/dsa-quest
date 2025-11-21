"""
DSA Learning Platform - Striver's A2Z Sheet
Gamified learning with AI-powered stories and analogies
"""

import os
from typing import List

import requests
import streamlit as st
import plotly.express as px
from streamlit_ace import st_ace
from streamlit_lottie import st_lottie

from dsa_data import STRIVER_SHEET, BADGES, TOTAL_PROBLEMS
from langraph_agent import generate_learning_content
from progress_tracker import get_progress_tracker
from backend_user import sign_in, sign_up, sign_out, get_current_user

st.write("DEBUG → Secrets Loaded:", list(st.secrets.keys()))
st.write("DEBUG → GROQ:", st.secrets.get("GROQ_API_KEY", "(missing)"))
# Page configuration (must be before other Streamlit calls)
st.set_page_config(
    page_title="DSA Quest - Master Coding with Fun!",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styles ----------
st.markdown(
    """
<style>
    /* Main title styling */
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
    .step-card { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); padding: 20px; border-radius:15px; margin:10px 0; box-shadow:0 4px 6px rgba(0,0,0,0.1); transition: transform 0.3s; }
    .step-card:hover { transform: translateY(-5px); box-shadow:0 6px 12px rgba(0,0,0,0.15); }
    .problem-card { background: white; padding: 15px; border-radius:10px; border-left:5px solid #4ECDC4; margin:10px 0; box-shadow:0 2px 4px rgba(0,0,0,0.1); }
    .badge { display:inline-block; padding:8px 16px; border-radius:20px; font-weight:600; margin:5px; animation:pulse 2s infinite; }
    .xp-bar { background: linear-gradient(90deg,#FFD700,#FFA500); height:30px; border-radius:15px; text-align:center; color:white; font-weight:bold; line-height:30px; }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes pulse { 0%,100% { transform: scale(1); } 50% { transform: scale(1.05); } }
    .code-container { background: #1e1e1e; border-radius:10px; padding:10px; margin:10px 0; }
    .difficulty-easy { color: #4CAF50; font-weight:bold; }
    .difficulty-medium { color: #FF9800; font-weight:bold; }
    .difficulty-hard { color: #F44336; font-weight:bold; }
    .story-box { background: linear-gradient(135deg,#FFE5B4 0%,#FFD700 100%); padding:20px; border-radius:15px; margin:15px 0; border-left:5px solid #FFA500; }
    .analogy-box { background: linear-gradient(135deg,#E0F7FA 0%,#80DEEA 100%); padding:20px; border-radius:15px; margin:15px 0; border-left:5px solid #00ACC1; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Helpers ----------
def load_lottie_url(url: str):
    """Load Lottie animation JSON from URL (returns dict or None)."""
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def show_celebration(unique_key: str = "celebration"):
    """Play a celebration Lottie if available."""
    lottie = load_lottie_url(
        "https://assets5.lottiefiles.com/packages/lf20_touohxv0.json"
    )
    if lottie:
        st_lottie(lottie, height=200, key=unique_key)


def get_current_badge_from_tracker(tracker):
    """Helper to compute current badge from ProgressTracker (multiuser version)."""
    if not tracker:
        return {"name": "🌱 DSA Rookie"}

    if getattr(tracker, "badges_earned", None):
        last_id = tracker.badges_earned[-1]
        info = BADGES.get(last_id, {})
        return {"name": info.get("name", "🌱 DSA Rookie")}

    return {"name": "🌱 DSA Rookie"}


# ---------- Auth UI (MAIN PAGE) ----------
def show_auth_ui():
    """Main page: hero + login / signup tabs."""
    st.markdown(
        '<h1 class="main-title">🚀 Welcome to DSA Quest!</h1>',
        unsafe_allow_html=True,
    )

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
    st.markdown("## 🔐 Sign In or Create your account")

    tab1, tab2 = st.tabs(["🔓 Sign In", "🆕 Sign Up"])

    # ---- Sign in ----
    with tab1:
        email = st.text_input("Email", key="signin_email")
        password = st.text_input(
            "Password", type="password", key="signin_password"
        )
        if st.button("Sign in"):
            res = sign_in(email, password)
            if getattr(res, "error", None):
                st.error(f"Sign in error: {res.error}")
            else:
                st.success("Signed in.")
                st.rerun()

    # ---- Sign up ----
    with tab2:
        email_s = st.text_input("New Email", key="signup_email")
        password_s = st.text_input(
            "New Password", type="password", key="signup_password"
        )
        username_s = st.text_input("Username", key="signup_username")
        if st.button("Create account"):
            res = sign_up(email_s, password_s, username=username_s)
            if getattr(res, "error", None):
                st.error(f"Sign up error: {res.error}")
            else:
                st.success("Account created. You can now sign in.")
                st.rerun()


# ---------- Sidebar ----------
def render_sidebar():
    """Render sidebar with navigation and user stats (no auth UI here)."""
    user = get_current_user()
    tracker = get_progress_tracker() if user else None

    with st.sidebar:
        st.markdown("# 🚀 DSA Quest")
        if user:
            st.caption(f"Signed in as **{user.get('email')}**")
        else:
            st.caption("Not signed in")

        st.markdown("---")

        # Progress section
        try:
            current_badge = (
                get_current_badge_from_tracker(tracker)
                if tracker
                else {"name": "🌱 DSA Rookie"}
            )
        except Exception:
            current_badge = {"name": "🌱 DSA Rookie"}

        total_xp = getattr(tracker, "total_xp", 0) if tracker else 0
        current_streak = getattr(tracker, "current_streak", 0) if tracker else 0
        completed_problems = (
            len(getattr(tracker, "completed_problems", [])) if tracker else 0
        )

        st.markdown("### Your Progress")
        st.markdown(f"**{current_badge.get('name', '🌱 DSA Rookie')}**")
        st.markdown(f"**XP:** {total_xp} 💎")
        st.markdown(f"**Streak:** {current_streak} 🔥")
        st.markdown(
            f"**Problems Solved:** {completed_problems}/{TOTAL_PROBLEMS}"
        )

        progress = (
            tracker.get_progress_percentage()
            if tracker and hasattr(tracker, "get_progress_percentage")
            else 0.0
        )
        st.progress(progress / 100 if TOTAL_PROBLEMS else 0)
        st.markdown(f"**{progress:.1f}% Complete**")

        st.markdown("---")
        st.markdown("### 📚 Navigation")
        page_options = ["🏠 Home", "📖 Problem Browser", "👤 Profile", "🏆 Achievements"]

        if "page" not in st.session_state:
            st.session_state.page = page_options[0]

        try:
            index = page_options.index(st.session_state.page)
        except Exception:
            index = 0

        selected = st.radio(
            "Go to:",
            page_options,
            index=index,
            label_visibility="collapsed",
            key="navigation_radio",
        )
        if selected != st.session_state.page:
            st.session_state.page = selected

        st.markdown("---")
        with st.expander("⚙️ Settings"):
            st.markdown("**Groq API Key**")
            st.markdown(
                "Add your API key to unlock AI-powered stories and analogies!"
            )
            try:
                if "GROQ_API_KEY" in st.secrets and st.secrets["GROQ_API_KEY"]:
                    st.success("✅ API Key configured!")
                else:
                    st.warning("⚠️ No API key found")
                    st.markdown(
                        "[Get your free Groq API key](https://console.groq.com)"
                    )
            except Exception:
                st.warning("⚠️ No API key found")
                st.markdown(
                    "[Get your free Groq API key](https://console.groq.com)"
                )

        # Logout button
        if user:
            st.markdown("---")
            if st.button("🚪 Sign out"):
                sign_out()
                st.rerun()

    return st.session_state.get("page", "🏠 Home")


# ---------- Pages ----------
def render_home():
    """Render home page with overview (for logged-in users)."""
    st.markdown(
        '<h1 class="main-title">🚀 Welcome to DSA Quest!</h1>',
        unsafe_allow_html=True,
    )
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

    user = get_current_user()
    if not user:
        st.info("Sign in to see your learning journey.")
        return

    tracker = get_progress_tracker()
    if not tracker:
        st.error("Could not load your progress. Please try again.")
        return

    for step_key, step_data in STRIVER_SHEET.items():
        step_progress = tracker.get_step_progress(step_key)
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(
                f"### {step_data.get('icon','')} {step_key}: {step_data.get('name')}"
            )
            st.markdown(f"*{step_data.get('description','')}*")
            st.progress(step_progress / 100)
            st.markdown(
                f"**{step_progress:.0f}% Complete** | {step_data.get('xp', 0)} XP per problem"
            )
       # with c2:
       #     if st.button(f"Start {step_key}", key=f"start_{step_key}"):
       #         st.session_state.current_step = step_key
        #        st.session_state.page = "📖 Problem Browser"
         #       st.session_state["navigation_radio"] = "📖 Problem Browser"
          #      st.rerun()
           #     return


def render_problem_browser():
    """Render problem browser with AI-powered content."""
    user = get_current_user()
    if not user:
        st.info("Please sign in to browse problems.")
        return

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

    problem_options = [
        f"{p['title']} ({p['difficulty']})" for p in step_data["problems"]
    ]
    selected_problem_str = st.selectbox("Select Problem", problem_options)

    selected_problem = next(
        (
            p
            for p in step_data["problems"]
            if f"{p['title']} ({p['difficulty']})" == selected_problem_str
        ),
        None,
    )
    if selected_problem:
        render_problem_detail(selected_problem, step_data.get("xp", 0))


def render_problem_detail(problem: dict, xp_reward: int):
    """Render detailed problem view with AI content."""
    user = get_current_user()
    if not user:
        st.info("Please sign in to view problem details.")
        return

    tracker = get_progress_tracker()
    if not tracker:
        st.error("Could not load your progress. Please try again.")
        return

    is_completed = problem["id"] in tracker.completed_problems

    h1, h2, h3 = st.columns([2, 1, 1])
    with h1:
        st.markdown(f"### {problem.get('title')}")
    with h2:
        difficulty_class = f"difficulty-{problem.get('difficulty','').lower()}"
        st.markdown(
            f'<p class="{difficulty_class}">{problem.get("difficulty")}</p>',
            unsafe_allow_html=True,
        )
    with h3:
        if is_completed:
            st.success("✅ Completed")
        else:
            if st.button("Mark Complete 🎉", key=f"complete_{problem['id']}"):
                is_new, new_badges = tracker.mark_problem_complete(
                    problem["id"], xp_reward
                )
                if is_new:
                    st.success(f"🎉 +{xp_reward} XP earned!")
                    if new_badges:
                        st.balloons()
                        for badge in new_badges:
                            st.success(f"🏆 New Badge Unlocked: {badge}")
                    show_celebration(f"celebration_{problem['id']}")
                    st.rerun()
                    return

    st.markdown(f"**Concepts:** {', '.join(problem.get('concepts', []))}")
    if problem.get("link"):
        st.markdown(f"[View on TakeUForward]({problem['link']})")

    st.markdown("---")
    st.markdown("## 🎨 AI-Powered Learning Content")
    content_key = f"content_{problem['id']}"
    if "content_cache" not in st.session_state:
        st.session_state.content_cache = {}

    col_gen, col_regen = st.columns([1, 1])
    with col_gen:
        if st.button("✨ Generate Story & Analogy", key=f"generate_{problem['id']}"):
            with st.spinner("🪄 Creating magical learning content with AI..."):
                try:
                    content = generate_learning_content(
                        problem["title"],
                        problem.get("concepts", []),
                        problem.get("difficulty", "Medium"),
                    )
                except Exception as e:
                    content = {
                        "story": f"⚠️ Error: {e}",
                        "analogy": "Unavailable",
                        "learning_tip": "Unavailable",
                        "has_error": True,
                    }
                st.session_state["content_cache"][content_key] = content
    with col_regen:
        if st.button("🔄 Regenerate Content", key=f"regenerate_{problem['id']}"):
            with st.spinner("🪄 Creating new learning content..."):
                try:
                    content = generate_learning_content(
                        problem["title"],
                        problem.get("concepts", []),
                        problem.get("difficulty", "Medium"),
                    )
                except Exception as e:
                    content = {
                        "story": f"⚠️ Error: {e}",
                        "analogy": "Unavailable",
                        "learning_tip": "Unavailable",
                        "has_error": True,
                    }
                st.session_state["content_cache"][content_key] = content

    if content_key in st.session_state["content_cache"]:
        content = st.session_state["content_cache"][content_key]
        if content.get("has_error"):
            st.warning(
                "⚠️ There was an issue generating AI content. Showing fallback/partial content."
            )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="story-box">', unsafe_allow_html=True)
            st.markdown("### 📖 Story")
            st.markdown(content.get("story", "No story available."))
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="analogy-box">', unsafe_allow_html=True)
            st.markdown("### 💡 Analogy")
            st.markdown(content.get("analogy", "No analogy available."))
            st.markdown("</div>", unsafe_allow_html=True)

        st.info(
            f"**💡 Learning Tip:** {content.get('learning_tip', 'No tip available.')}"
        )

        st.markdown("---")
        st.markdown("### 🔎 Debug: Raw LLM response (dev-only)")
        extracted = st.session_state.get("last_llm_extracted_text")
        raw_repr = st.session_state.get("last_llm_raw_repr")

        if extracted:
            st.subheader("Extracted text (what we used)")
            st.write(extracted)
        else:
            st.info("No extracted LLM text found in session state.")

        if raw_repr:
            st.subheader("Raw repr (truncated)")
            st.code(raw_repr[:3000])
        else:
            st.info("No raw LLM repr found in session state.")

        st.markdown(
            "⚠️ *This debug output is intended for local/dev use only. Remove it in production to avoid exposing sensitive data.*"
        )

    st.markdown("---")
    st.markdown("## 💻 Solution Code")
    language = st.radio(
        "Select Language:",
        ["C++", "Java", "Python"],
        horizontal=True,
        key=f"lang_{problem['id']}",
    )
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
    user_code = st_ace(
        value=f"# Write your {language} solution here\n\n",
        language=lang_mode if lang_mode != "c_cpp" else "c_cpp",
        theme="monokai",
        height=300,
        key=f"editor_{problem['id']}_{language}",
    )

    if st.button("Save My Solution 💾", key=f"save_{problem['id']}"):
        st.success("Solution saved! Keep practicing! 🚀")


def render_profile():
    """Render user profile dashboard."""
    user = get_current_user()
    if not user:
        st.info("Please sign in to view your profile.")
        return

    tracker = get_progress_tracker()
    if not tracker:
        st.error("Could not load your progress. Please try again.")
        return

    st.markdown("# 👤 Your Coding Profile")
    current_badge = get_current_badge_from_tracker(tracker)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"## {current_badge.get('name')}")
        st.markdown(f"### {tracker.total_xp} XP")
    with c2:
        st.metric("Problems Solved", len(tracker.completed_problems))
        st.metric("Current Streak", f"{tracker.current_streak} days 🔥")
    with c3:
        progress = tracker.get_progress_percentage()
        st.metric("Overall Progress", f"{progress:.1f}%")

    st.markdown("---")
    st.markdown("### 📊 XP Progress")
    step_xp_data = []
    for step_key, step_data in STRIVER_SHEET.items():
        completed_in_step = sum(
            1
            for p in step_data.get("problems", [])
            if p["id"] in tracker.completed_problems
        )
        step_xp = completed_in_step * step_data.get("xp", 0)
        if step_xp > 0:
            step_xp_data.append(
                {"Step": f"{step_key}: {step_data.get('name')}", "XP": step_xp}
            )

    if step_xp_data:
        fig = px.bar(
            step_xp_data,
            x="Step",
            y="XP",
            title="XP Distribution by Topic",
            color="XP",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Start solving problems to see your XP distribution!")

    st.markdown("---")
    st.markdown("### 🗺️ Your Journey Map")
    for step_key, step_data in STRIVER_SHEET.items():
        progress = tracker.get_step_progress(step_key)
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"## {step_data.get('icon','')}")
        with col2:
            st.markdown(f"**{step_data.get('name')}**")
            st.progress(progress / 100)
            st.markdown(f"{progress:.0f}%")


def render_achievements():
    """Render achievements and badges."""
    user = get_current_user()
    if not user:
        st.info("Please sign in to view achievements.")
        return

    tracker = get_progress_tracker()
    if not tracker:
        st.error("Could not load your progress. Please try again.")
        return

    st.markdown("# 🏆 Achievements & Badges")
    st.markdown("### Your Badges")

    cols = st.columns(len(BADGES))
    for idx, (badge_id, badge_info) in enumerate(BADGES.items()):
        with cols[idx]:
            earned = badge_id in tracker.badges_earned
            if earned:
                st.markdown(f"### {badge_info.get('name')}")
                st.success("✅ Unlocked!")
            else:
                st.markdown("### 🔒")
                st.markdown(f"{badge_info.get('name')}")
                xp_needed = badge_info.get("xp_required", 0) - tracker.total_xp
                st.info(f"{xp_needed} XP needed")

    st.markdown("---")
    st.markdown("### 📈 Your Statistics")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total XP", tracker.total_xp)
    with c2:
        st.metric("Problems Solved", len(tracker.completed_problems))
    with c3:
        st.metric("Badges Earned", len(tracker.badges_earned))
    with c4:
        st.metric("Current Streak", f"{tracker.current_streak} 🔥")

    st.markdown("---")
    st.markdown("### 🎯 Next Milestone")
    next_badge = None
    for badge_id, badge_info in BADGES.items():
        if badge_info.get("xp_required", 0) > tracker.total_xp:
            next_badge = badge_info
            break

    if next_badge:
        xp_needed = next_badge["xp_required"] - tracker.total_xp
        progress_pct = (
            tracker.total_xp / next_badge["xp_required"] * 100
            if next_badge["xp_required"]
            else 100
        )
        st.markdown(f"**Next Badge:** {next_badge['name']}")
        st.progress(progress_pct / 100)
        st.markdown(f"**{xp_needed} XP** to go!")
    else:
        st.success("🎉 You've unlocked all badges! You're a Coding Legend! 👑")


# ---------- Session Initialization ----------
def initialize_session_state():
    """Initialize session state variables used by the app."""
    if "page" not in st.session_state:
        st.session_state.page = "🏠 Home"
    if "current_step" not in st.session_state:
        st.session_state.current_step = None
    if "content_cache" not in st.session_state:
        st.session_state.content_cache = {}
    if "show_celebration" not in st.session_state:
        st.session_state.show_celebration = False
    if "last_llm_extracted_text" not in st.session_state:
        st.session_state.last_llm_extracted_text = None
    if "last_llm_raw_repr" not in st.session_state:
        st.session_state.last_llm_raw_repr = None


# ---------- Main ----------
def main():
    initialize_session_state()

    user = get_current_user()

    # If not logged in, show landing + auth on MAIN page, minimal sidebar
    if not user:
        # Sidebar still shows app name + simple message
        with st.sidebar:
            st.markdown("# 🚀 DSA Quest")
            st.info("Please sign in to start your DSA Quest.")
        show_auth_ui()
        return

    # Logged in: normal sidebar + pages
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
