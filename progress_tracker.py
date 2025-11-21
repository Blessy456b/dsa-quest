# progress_tracker.py
"""
FINAL VERSION — Combined for Supabase Multiuser Storage

Features:
- Per-user isolated Supabase backend
- XP system
- Streaks (Asia/Kolkata timezone)
- Step progress (using STRIVER_SHEET)
- Badge unlocking (using BADGES)
- Reload-safe + persistence
"""

from __future__ import annotations

import json
from datetime import datetime, date
from typing import List, Tuple, Optional

import streamlit as st

try:
    from zoneinfo import ZoneInfo
    KOLKATA_TZ = ZoneInfo("Asia/Kolkata")
except Exception:
    KOLKATA_TZ = None

from backend_user import UserBackend, get_current_user


# ------------------------------------------------------------------- #
# Helper: Get Supabase per-user backend                               #
# ------------------------------------------------------------------- #

def _get_user_backend():
    user = get_current_user()
    if not user:
        raise RuntimeError("No authenticated Supabase user in session_state")
    return UserBackend(user["id"])


# ------------------------------------------------------------------- #
# Final Combined ProgressTracker                                      #
# ------------------------------------------------------------------- #

class ProgressTracker:
    """Per-user progress tracker stored in Supabase."""

    def __init__(self):
        self.backend = _get_user_backend()
        self._reload()

    # --------------------------------------------------------------- #
    # Load from Supabase                                              #
    # --------------------------------------------------------------- #

    def _reload(self):
        data = self.backend.load_progress()

        self.total_xp: int = int(data.get("total_xp", 0))
        self.completed_problems: set = set(data.get("completed_problems", []))
        self.current_streak: int = int(data.get("current_streak", 0))
        self.badges_earned: List[str] = data.get("badges", [])
        self.last_activity: Optional[str] = data.get("last_activity", None)

    # --------------------------------------------------------------- #
    # Save to Supabase                                                #
    # --------------------------------------------------------------- #

    def save(self):
        payload = {
            "user_id": self.backend.user_id,
            "completed_problems": list(self.completed_problems),
            "total_xp": self.total_xp,
            "current_streak": self.current_streak,
            "badges": self.badges_earned,
            "last_activity": self.last_activity,
        }
        self.backend.save_progress(payload)

    # --------------------------------------------------------------- #
    # Utility: Today's date (timezone-aware)                          #
    # --------------------------------------------------------------- #

    def _today_iso(self) -> str:
        if KOLKATA_TZ:
            now = datetime.now(KOLKATA_TZ)
        else:
            now = datetime.utcnow()
        return now.date().isoformat()

    # --------------------------------------------------------------- #
    # Mark Problem Complete                                           #
    # --------------------------------------------------------------- #

    def mark_problem_complete(self, problem_id: int, xp_reward: int) -> Tuple[bool, List[str]]:
        """
        Returns: (is_new_completion, new_badges_list)
        """
        if problem_id in self.completed_problems:
            return False, []

        # add problem
        self.completed_problems.add(problem_id)
        self.total_xp += int(xp_reward)

        # streak logic
        today = self._today_iso()

        if self.last_activity == today:
            pass  # streak already counted today
        else:
            if self.last_activity:
                try:
                    last = datetime.strptime(self.last_activity, "%Y-%m-%d").date()
                    today_dt = datetime.strptime(today, "%Y-%m-%d").date()
                    delta = (today_dt - last).days
                    if delta == 1:
                        self.current_streak += 1
                    else:
                        self.current_streak = 1
                except Exception:
                    self.current_streak = 1
            else:
                self.current_streak = 1

            self.last_activity = today

        # check new badges
        new_badges = self._check_new_badges()
        self.save()
        return True, new_badges

    # --------------------------------------------------------------- #
    # Badge Logic (using your dsa_data.BADGES)                        #
    # --------------------------------------------------------------- #

    def _check_new_badges(self) -> List[str]:
        try:
            from dsa_data import BADGES
        except Exception:
            BADGES = {}

        new = []
        for badge_id, badge in BADGES.items():
            if badge_id not in self.badges_earned:
                if self.total_xp >= int(badge["xp_required"]):
                    self.badges_earned.append(badge_id)
                    new.append(badge["name"])
        return new

    # --------------------------------------------------------------- #
    # Overall Progress Percentage                                     #
    # --------------------------------------------------------------- #

    def get_progress_percentage(self) -> float:
        try:
            from dsa_data import TOTAL_PROBLEMS
        except Exception:
            TOTAL_PROBLEMS = 0

        if TOTAL_PROBLEMS == 0:
            return 0.0

        return (len(self.completed_problems) / TOTAL_PROBLEMS) * 100.0

    # --------------------------------------------------------------- #
    # Step-level Progress                                             #
    # --------------------------------------------------------------- #

    def get_step_progress(self, step_name: str) -> float:
        try:
            from dsa_data import STRIVER_SHEET
        except Exception:
            STRIVER_SHEET = {}

        step = STRIVER_SHEET.get(step_name, {})
        problems = step.get("problems", [])

        if not problems:
            return 0.0

        completed = sum(1 for p in problems if p["id"] in self.completed_problems)
        return (completed / len(problems)) * 100.0

    # --------------------------------------------------------------- #
    # Reset user progress                                             #
    # --------------------------------------------------------------- #

    def reset_progress(self):
        self.total_xp = 0
        self.completed_problems = set()
        self.current_streak = 0
        self.badges_earned = []
        self.last_activity = None
        self.save()

    # --------------------------------------------------------------- #
    # Export (for user downloads)                                     #
    # --------------------------------------------------------------- #

    def export_progress_json(self) -> str:
        payload = {
            "total_xp": self.total_xp,
            "completed_problems": sorted(list(self.completed_problems)),
            "current_streak": self.current_streak,
            "badges": self.badges_earned,
            "last_activity": self.last_activity,
        }
        return json.dumps(payload, indent=2)


# ------------------------------------------------------------------- #
# Session Helper                                                     #
# ------------------------------------------------------------------- #

def get_progress_tracker():
    """Each user gets an isolated tracker stored in session."""
    user = get_current_user()
    if not user:
        st.error("Please login first.")
        return None

    uid = user["id"]
    key = f"progress_tracker::{uid}"

    if key not in st.session_state:
        st.session_state[key] = ProgressTracker()

    return st.session_state[key]
