import types
import backend_user


class DummyAuth:
    def __init__(self):
        self.signed_out = False
        self.last_credentials = None

    def sign_in_with_password(self, creds):
        self.last_credentials = creds
        # Return a minimal object with .data.user.id & .email
        user = types.SimpleNamespace(id="user-123", email=creds["email"])
        data = types.SimpleNamespace(user=user)
        return types.SimpleNamespace(data=data)

    def sign_up(self, creds):
        user = types.SimpleNamespace(id="user-456", email=creds["email"])
        data = types.SimpleNamespace(user=user)
        return types.SimpleNamespace(data=data)

    def sign_out(self):
        self.signed_out = True

    def get_session(self):
        # Simulate "no session" here; get_current_user can still use st.session_state
        return None


class DummyTable:
    def __init__(self):
        self.operations = []

    def select(self, *_args, **_kwargs):
        self.operations.append(("select", _args, _kwargs))
        return self

    def eq(self, *_args, **_kwargs):
        self.operations.append(("eq", _args, _kwargs))
        return self

    def maybe_single(self):
        self.operations.append(("maybe_single", (), {}))
        return self

    def execute(self):
        # Minimal response with .data
        return types.SimpleNamespace(data=None)

    def insert(self, payload):
        self.operations.append(("insert", payload))
        return self

    def upsert(self, payload, **kwargs):
        self.operations.append(("upsert", payload, kwargs))
        return self


class DummySupabase:
    def __init__(self):
        self.auth = DummyAuth()
        self.tables = {}

    def table(self, name):
        if name not in self.tables:
            self.tables[name] = DummyTable()
        return self.tables[name]


def test_sign_in_sets_session_state(monkeypatch):
    dummy_sb = DummySupabase()
    monkeypatch.setattr(backend_user, "supabase", dummy_sb)

    # Ensure a clean session_state
    import streamlit as st
    st.session_state.clear()

    res = backend_user.sign_in("test@example.com", "password123")
    assert hasattr(res, "data")
    assert "user" in st.session_state
    assert st.session_state["user"]["id"] == "user-123"
    assert st.session_state["user"]["email"] == "test@example.com"


def test_sign_up_creates_default_rows(monkeypatch):
    dummy_sb = DummySupabase()
    monkeypatch.setattr(backend_user, "supabase", dummy_sb)

    res = backend_user.sign_up("new@example.com", "secret", username="tester")
    assert hasattr(res, "data")
    # Ensure progress & settings tables are used
    assert "users_progress" in dummy_sb.tables
    assert "user_settings" in dummy_sb.tables


def test_sign_out_clears_session_state(monkeypatch):
    dummy_sb = DummySupabase()
    monkeypatch.setattr(backend_user, "supabase", dummy_sb)

    import streamlit as st
    st.session_state.clear()
    st.session_state["user"] = {"id": "user-123"}
    st.session_state["_rerun_count"] = 1

    backend_user.sign_out()

    assert "user" not in st.session_state  # cleared
    assert "_rerun_count" in st.session_state  # preserved


def test_user_backend_load_progress_creates_default(monkeypatch):
    dummy_sb = DummySupabase()
    monkeypatch.setattr(backend_user, "supabase", dummy_sb)

    ub = backend_user.UserBackend(user_id="user-123")
    data = ub.load_progress()

    assert data["user_id"] == "user-123"
    assert isinstance(data["completed_problems"], list)
    assert data["total_xp"] == 0


def test_user_backend_upsert_progress(monkeypatch):
    dummy_sb = DummySupabase()
    monkeypatch.setattr(backend_user, "supabase", dummy_sb)

    ub = backend_user.UserBackend(user_id="user-123")
    ub.upsert_progress(completed_problems=[1, 2], total_xp=100)

    table = dummy_sb.tables["users_progress"]
    # Ensure at least one upsert operation was recorded
    assert any(op[0] == "upsert" for op in table.operations)
