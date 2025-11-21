import builtins
from utils.retry_utils import retry_with_backoff


def test_retry_with_backoff_success_first_try(monkeypatch):
    calls = {"count": 0}

    def fn():
        calls["count"] += 1
        return "ok"

    # Avoid real sleeping
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    result = retry_with_backoff(fn)
    assert result == "ok"
    assert calls["count"] == 1


def test_retry_with_backoff_eventual_success(monkeypatch):
    calls = {"count": 0}

    class CustomError(Exception):
        pass

    def fn():
        calls["count"] += 1
        if calls["count"] < 3:
            raise CustomError("temporary")
        return "success"

    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    result = retry_with_backoff(fn, max_retries=5, exceptions=(CustomError,))
    assert result == "success"
    assert calls["count"] == 3  # failed twice, then succeeded


def test_retry_with_backoff_exhausts_retries(monkeypatch):
    calls = {"count": 0}

    class CustomError(Exception):
        pass

    def fn():
        calls["count"] += 1
        raise CustomError("always fails")

    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)

    try:
        retry_with_backoff(fn, max_retries=3, exceptions=(CustomError,))
    except CustomError as e:
        assert "always fails" in str(e)
        assert calls["count"] == 4  # initial + 3 retries
    else:
        raise AssertionError("Expected CustomError to be raised")
