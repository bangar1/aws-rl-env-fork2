"""Regression tests for scripts/grpo_pool.py.

Focus: `GrpoPool.connect()` must be all-or-nothing. If any single WebSocket
handshake fails, every session that DID connect must be closed before the
error propagates, so the server never ends up with leaked pool slots.

No pytest-asyncio dependency — each test drives the loop via asyncio.run().
"""

from __future__ import annotations

import asyncio

import pytest

from scripts.grpo_pool import GrpoPool


class _FakeEnv:
    """Minimal stand-in for AwsRlEnv. Tracks connect/close lifecycle."""

    connect_calls = 0  # class-level so the factory can index envs in order

    def __init__(self, *, should_fail_on_index: int | None = None) -> None:
        self.connected = False
        self.close_called = False
        self._index = _FakeEnv.connect_calls
        _FakeEnv.connect_calls += 1
        self._should_fail = (
            should_fail_on_index is not None and self._index == should_fail_on_index
        )

    async def connect(self) -> None:
        if self._should_fail:
            raise ConnectionError(f"fake failure on env#{self._index}")
        await asyncio.sleep(0)  # yield so sibling connects can interleave
        self.connected = True

    async def close(self) -> None:
        self.close_called = True


def _install_fake_env(monkeypatch, fail_on_index: int | None) -> list[_FakeEnv]:
    """Monkeypatch AwsRlEnv inside scripts.grpo_pool so GrpoPool builds FakeEnvs.

    Returns a shared list the test can inspect after connect() runs.
    """
    _FakeEnv.connect_calls = 0
    created: list[_FakeEnv] = []

    def factory(*args, **kwargs) -> _FakeEnv:
        env = _FakeEnv(should_fail_on_index=fail_on_index)
        created.append(env)
        return env

    monkeypatch.setattr("scripts.grpo_pool.AwsRlEnv", factory)
    return created


# ---------------------------------------------------------------------------
# Happy path — sanity check the fake harness before running the failure cases
# ---------------------------------------------------------------------------


class TestConnectHappyPath:
    def test_all_sessions_connect_and_land_on_pool(self, monkeypatch) -> None:
        created = _install_fake_env(monkeypatch, fail_on_index=None)
        pool = GrpoPool(base_url="http://x", size=4)
        asyncio.run(pool.connect())
        assert len(pool.envs) == 4
        assert all(e.connected for e in created)
        assert not any(e.close_called for e in created)


# ---------------------------------------------------------------------------
# The review: partial failure must roll back
# ---------------------------------------------------------------------------


class TestConnectRollbackOnPartialFailure:
    def test_failure_closes_every_env_including_successful_ones(
        self, monkeypatch
    ) -> None:
        created = _install_fake_env(monkeypatch, fail_on_index=2)
        pool = GrpoPool(base_url="http://x", size=4)

        with pytest.raises(ConnectionError):
            asyncio.run(pool.connect())

        # Every FakeEnv must have had close() called — successful ones so
        # server slots are released; the failing one as a harmless no-op.
        assert all(e.close_called for e in created), (
            "Regression: successful sessions leaked after partial connect failure"
        )

    def test_pool_envs_stays_empty_on_failure(self, monkeypatch) -> None:
        _install_fake_env(monkeypatch, fail_on_index=1)
        pool = GrpoPool(base_url="http://x", size=3)

        with pytest.raises(ConnectionError):
            asyncio.run(pool.connect())

        # connect() must NOT leave a half-initialised pool visible to callers.
        assert pool.envs == []

    def test_failure_does_not_block_retry(self, monkeypatch) -> None:
        """After a failed connect(), the caller can fix the root cause and
        call connect() again. pool.envs should be fresh."""
        _install_fake_env(monkeypatch, fail_on_index=0)
        pool = GrpoPool(base_url="http://x", size=2)
        with pytest.raises(ConnectionError):
            asyncio.run(pool.connect())

        # Second attempt with no injected failure should succeed.
        _install_fake_env(monkeypatch, fail_on_index=None)
        asyncio.run(pool.connect())
        assert len(pool.envs) == 2
        assert all(e.connected for e in pool.envs)

    def test_async_context_manager_cleans_up_when_enter_fails(
        self, monkeypatch
    ) -> None:
        """If `async with GrpoPool(...)` raises during __aenter__,
        __aexit__ is NOT called — so rollback must live inside connect()
        itself. This test exercises exactly that scenario.
        """
        created = _install_fake_env(monkeypatch, fail_on_index=2)

        async def enter_and_fail() -> None:
            async with GrpoPool(base_url="http://x", size=4):
                pytest.fail("should never enter the body")

        with pytest.raises(ConnectionError):
            asyncio.run(enter_and_fail())

        assert all(e.close_called for e in created)
