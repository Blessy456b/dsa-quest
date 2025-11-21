import types
from progress_tracker import ProgressTracker

class FakeBackend:
    def __init__(self, initial=None):
        initial = initial or {}
        self.user_id = initial.get("user_id", "test-user")
        self.data = initial

    def load_progress(self):
        return self.data or {
            "user_id": self.user_id,
            "completed_problems": [],
            "total_xp": 0,
            "current_streak": 0,
            "badges": [],
            "last_activity": None,
        }

    def save_progress(self, payload):
        self.data = payload


def test_mark_problem_complete_new_day(monkeypatch):
    # Patch _get_user_backend to return our FakeBackend
    from progress_tracker import _get_user_backend

    fake_backend = FakeBackend()
    monkeypatch.setattr("progress_tracker._get_user_backend", lambda: fake_backend)

    tracker = ProgressTracker()
    is_new, new_badges = tracker.mark_problem_complete(problem_id=1, xp_reward=50)

    assert is_new is True
    assert 1 in tracker.completed_problems
    assert tracker.total_xp == 50
    assert tracker.current_streak == 1
    assert isinstance(new_badges, list)


def test_mark_problem_complete_duplicate(monkeypatch):
    fake_backend = FakeBackend(
        initial={
            "user_id": "user-123",
            "completed_problems": [1],
            "total_xp": 50,
            "current_streak": 2,
            "badges": [],
            "last_activity": None,
        }
    )
    monkeypatch.setattr("progress_tracker._get_user_backend", lambda: fake_backend)

    tracker = ProgressTracker()
    is_new, new_badges = tracker.mark_problem_complete(problem_id=1, xp_reward=50)

    assert is_new is False
    assert new_badges == []
    assert tracker.total_xp == 50  # unchanged


def test_reset_progress(monkeypatch):
    fake_backend = FakeBackend(
        initial={
            "user_id": "user-123",
            "completed_problems": [1, 2],
            "total_xp": 200,
            "current_streak": 3,
            "badges": ["rookie"],
            "last_activity": "2025-01-01",
        }
    )
    monkeypatch.setattr("progress_tracker._get_user_backend", lambda: fake_backend)

    tracker = ProgressTracker()
    tracker.reset_progress()

    assert tracker.total_xp == 0
    assert tracker.completed_problems == set()
    assert tracker.current_streak == 0
    assert tracker.badges_earned == []
    assert tracker.last_activity is None


def test_export_progress_json(monkeypatch):
    fake_backend = FakeBackend()
    monkeypatch.setattr("progress_tracker._get_user_backend", lambda: fake_backend)

    tracker = ProgressTracker()
    tracker.mark_problem_complete(1, 50)
    exported = tracker.export_progress_json()

    assert '"total_xp"' in exported
    assert '"completed_problems"' in exported
