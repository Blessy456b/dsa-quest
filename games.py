# games.py
"""
Interactive coding mini-games for DSA Quest.

Games:
1. Rearrange-the-code
2. Multiple-choice quiz
3. Fill-in-the-blanks

Includes optional AI-powered story challenge using generate_learning_content.
"""

import random
from typing import Dict, Any, List

import streamlit as st

from backend_user import get_current_user
from dsa_data import STRIVER_SHEET
from langraph_agent import generate_learning_content
from progress_tracker import get_progress_tracker


GAME_XP_REWARD = 10  # XP per correct answer (if tracker supports it)


# ------------------------------
# Helper: safe XP award
# ------------------------------
def award_game_xp(tracker, reason: str = ""):
    """Try to award XP without crashing if tracker doesn't support it."""
    if not tracker:
        return
    try:
        # Prefer explicit methods if they exist
        if hasattr(tracker, "add_xp_from_game"):
            tracker.add_xp_from_game(GAME_XP_REWARD)
        elif hasattr(tracker, "add_xp"):
            tracker.add_xp(GAME_XP_REWARD)
        elif hasattr(tracker, "total_xp"):
            tracker.total_xp = getattr(tracker, "total_xp", 0) + GAME_XP_REWARD
            if hasattr(tracker, "save"):
                tracker.save()
    except Exception:
        # Silent fail – don't break UI just for XP
        pass


# ------------------------------
# Game Data (static, can later be moved to data.json)
# ------------------------------

REARRANGE_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "rearr_1",
        "title": "Print all elements of an array",
        "language": "Python",
        "lines_correct": [
            "arr = [1, 2, 3, 4]",
            "for x in arr:",
            "    print(x)",
        ],
        # We show them shuffled, user types order
    },
    {
        "id": "rearr_2",
        "title": "Compute sum of an array",
        "language": "C++",
        "lines_correct": [
            "int sum = 0;",
            "for (int x : arr) {",
            "    sum += x;",
            "}",
        ],
    },
]

QUIZ_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "quiz_1",
        "question": "Which data structure gives O(1) average time for search, insert, and delete?",
        "options": ["Array", "Linked List", "Hash Map", "Binary Search Tree"],
        "answer_index": 2,
        "explanation": "Hash maps (or unordered maps) provide average O(1) for search, insert and delete.",
    },
    {
        "id": "quiz_2",
        "question": "What is the time complexity of binary search?",
        "options": ["O(n)", "O(log n)", "O(n log n)", "O(1)"],
        "answer_index": 1,
        "explanation": "Binary search repeatedly halves the search space, so it is O(log n).",
    },
    {
        "id": "quiz_3",
        "question": "Which traversal visits nodes in sorted order in a BST?",
        "options": ["Preorder", "Inorder", "Postorder", "Level order"],
        "answer_index": 1,
        "explanation": "Inorder traversal of a Binary Search Tree yields nodes in sorted order.",
    },
]

FILL_BLANK_QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "fill_1",
        "language": "Python",
        "snippet": "for i in range(___):\n    print(i)",
        "answer": "n",
        "hint": "We usually loop from 0 to n-1.",
    },
    {
        "id": "fill_2",
        "language": "C++",
        "snippet": "for (int i = 0; i < ___; i++) {\n    cout << i << endl;\n}",
        "answer": "n",
        "hint": "Classic C++ for-loop from 0 to n-1.",
    },
    {
        "id": "fill_3",
        "language": "Python",
        "snippet": "if left <= right:\n    mid = (left + right) // ___",
        "answer": "2",
        "hint": "Binary search splits the subarray in half.",
    },
]


# ------------------------------
# Rearrange-the-code Game
# ------------------------------
def render_rearrange_game(tracker):
    st.subheader("🧩 Game 1: Rearrange the Code")

    if "rearrange_idx" not in st.session_state:
        st.session_state.rearrange_idx = 0

    q = REARRANGE_QUESTIONS[st.session_state.rearrange_idx]
    st.markdown(f"**Task:** Arrange the lines to form a valid `{q['language']}` snippet.")
    st.markdown(f"**Snippet:** {q['title']}")

    # Show shuffled lines
    shuffled = list(enumerate(q["lines_correct"], start=1))
    random.seed(q["id"])  # deterministic shuffle per question
    random.shuffle(shuffled)

    st.markdown("**Lines (shuffled):**")
    for idx, line in shuffled:
        st.code(f"[{idx}] {line}", language=q["language"].lower())

    st.markdown(
        "👉 Enter the correct order as comma-separated indices. "
        "For example: `2,1,3`"
    )
    user_order = st.text_input(
        "Your order", key=f"rearr_order_{q['id']}", placeholder="e.g., 2,1,3"
    )

    if st.button("Check answer", key=f"rearr_check_{q['id']}"):
        try:
            user_indices = [
                int(x.strip()) for x in user_order.split(",") if x.strip()
            ]
        except ValueError:
            st.error("Please enter valid integers separated by commas.")
            return

        correct_indices = [i for i, _ in shuffled]
        # map shuffled indices back to original positions
        # we want user choice to reconstruct q["lines_correct"] order
        # So compute the lines in the order user selected:
        user_lines = []
        for idx in user_indices:
            for j, line in shuffled:
                if j == idx:
                    user_lines.append(line)
                    break

        if user_lines == q["lines_correct"]:
            st.success(f"✅ Correct! +{GAME_XP_REWARD} XP")
            award_game_xp(tracker, reason="rearrange_game")
        else:
            st.error("❌ Not quite. Try again!")
            st.markdown("**Correct order should be:**")
            for line in q["lines_correct"]:
                st.code(line, language=q["language"].lower())

    if st.button("Next puzzle ➡️", key=f"rearr_next_{q['id']}"):
        st.session_state.rearrange_idx = (st.session_state.rearrange_idx + 1) % len(
            REARRANGE_QUESTIONS
        )
        st.rerun()


# ------------------------------
# Multiple-choice Quiz Game
# ------------------------------
def render_quiz_game(tracker):
    st.subheader("🃏 Game 2: DSA Quiz Cards")

    if "quiz_idx" not in st.session_state:
        st.session_state.quiz_idx = 0

    q = QUIZ_QUESTIONS[st.session_state.quiz_idx]

    st.markdown(f"**Question:** {q['question']}")
    choice = st.radio(
        "Choose your answer:",
        q["options"],
        index=None,
        key=f"quiz_choice_{q['id']}",
    )

    if st.button("Check answer", key=f"quiz_check_{q['id']}"):
        if choice is None:
            st.warning("Please select an option first.")
            return

        correct = q["options"][q["answer_index"]]
        if choice == correct:
            st.success(f"✅ Correct! +{GAME_XP_REWARD} XP")
            award_game_xp(tracker, reason="quiz_game")
        else:
            st.error(f"❌ Incorrect. Correct answer is **{correct}**.")
        st.info(q.get("explanation", ""))

    if st.button("Next question ➡️", key=f"quiz_next_{q['id']}"):
        st.session_state.quiz_idx = (st.session_state.quiz_idx + 1) % len(
            QUIZ_QUESTIONS
        )
        st.rerun()


# ------------------------------
# Fill-in-the-blanks Game
# ------------------------------
def render_fill_blank_game(tracker):
    st.subheader("✏️ Game 3: Fill in the Blanks")

    if "fill_idx" not in st.session_state:
        st.session_state.fill_idx = 0

    q = FILL_BLANK_QUESTIONS[st.session_state.fill_idx]

    st.markdown("**Complete the code snippet:**")
    st.code(q["snippet"], language=q["language"].lower())
    st.caption(f"Hint: {q.get('hint', '')}")

    ans = st.text_input(
        "Fill in the `___`:", key=f"fill_ans_{q['id']}", placeholder="Your answer"
    )

    if st.button("Check answer", key=f"fill_check_{q['id']}"):
        if not ans.strip():
            st.warning("Please enter an answer.")
            return
        correct = str(q["answer"]).strip()
        if ans.strip() == correct:
            st.success(f"✅ Correct! +{GAME_XP_REWARD} XP")
            award_game_xp(tracker, reason="fill_blank_game")
        else:
            st.error(f"❌ Incorrect. Correct answer is `{correct}`.")

    if st.button("Next snippet ➡️", key=f"fill_next_{q['id']}"):
        st.session_state.fill_idx = (st.session_state.fill_idx + 1) % len(
            FILL_BLANK_QUESTIONS
        )
        st.rerun()


# ------------------------------
# AI Story Challenge (optional)
# ------------------------------
def render_ai_story_challenge(tracker):
    st.markdown("---")
    st.subheader("🧠 Bonus: AI Story Challenge")

    st.markdown(
        "Let the AI generate a short story from one of the Striver problems, "
        "and then guess the **main concept**."
    )

    if st.button("✨ Generate AI story challenge"):
        # Pick a random problem from STRIVER_SHEET
        all_problems: List[Dict[str, Any]] = []
        for step in STRIVER_SHEET.values():
            all_problems.extend(step.get("problems", []))

        if not all_problems:
            st.warning("No problems found to generate AI story from.")
            return

        problem = random.choice(all_problems)
        concepts = problem.get("concepts", []) or ["arrays"]
        difficulty = problem.get("difficulty", "Medium")

        with st.spinner("Generating AI story..."):
            try:
                content = generate_learning_content(
                    problem.get("title", "DSA problem"),
                    concepts,
                    difficulty,
                )
            except Exception as e:
                st.error(f"Error generating AI content: {e}")
                return

        st.markdown("### 📖 Story")
        st.markdown(content.get("story", "No story available."))

        # Build options: correct concept + some distractors
        all_concepts = set()
        for step in STRIVER_SHEET.values():
            for p in step.get("problems", []):
                for c in p.get("concepts", []):
                    all_concepts.add(c)

        correct_concept = concepts[0]
        distractors = list(all_concepts - {correct_concept})
        random.shuffle(distractors)
        distractors = distractors[:3]
        options = [correct_concept] + distractors
        random.shuffle(options)

        choice = st.radio(
            "Which concept does this story mainly illustrate?",
            options,
            key=f"ai_story_choice_{problem.get('id','unknown')}",
        )

        if st.button("Check AI challenge answer"):
            if choice == correct_concept:
                st.success(f"🔥 Nailed it! The main concept is indeed **{correct_concept}**. +{GAME_XP_REWARD} XP")
                award_game_xp(tracker, reason="ai_story_game")
            else:
                st.error(f"Not quite. The main concept was **{correct_concept}**.")


# ------------------------------
# Public entry point
# ------------------------------
def render_games():
    """Main entry point for the 🎮 Coding Games page."""
    user = get_current_user()
    if not user:
        st.info("Please sign in to play coding games.")
        return

    tracker = get_progress_tracker()

    st.markdown("## 🎮 Coding Games")
    st.markdown(
        "Strengthen your DSA skills with fun mini-games. "
        "Each correct answer can reward you with XP!"
    )

    game_tabs = st.tabs(
        ["🧩 Rearrange Code", "🃏 Quiz", "✏️ Fill Blanks", "🧠 AI Story Challenge"]
    )

    with game_tabs[0]:
        render_rearrange_game(tracker)

    with game_tabs[1]:
        render_quiz_game(tracker)

    with game_tabs[2]:
        render_fill_blank_game(tracker)

    with game_tabs[3]:
        render_ai_story_challenge(tracker)
