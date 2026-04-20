"""Unit tests for the MiniStackPool and env factory (parallel-rollout support).

These are pure unit tests — no MiniStack, no Docker, no network.

Run:
    python -m pytest tests/test_pool.py -v
"""

from __future__ import annotations

import threading
from unittest.mock import patch

import pytest

from server.app import MiniStackPool, make_env_factory
from server.aws_rl_env_environment import AwsRlEnvironment


# ---------------------------------------------------------------------------
# MiniStackPool
# ---------------------------------------------------------------------------


class TestMiniStackPoolBasics:
    def test_init_records_all_ports_as_free(self) -> None:
        pool = MiniStackPool([4566, 4567, 4568])
        assert pool.free_count == 3

    def test_init_with_empty_iterable(self) -> None:
        pool = MiniStackPool([])
        assert pool.free_count == 0

    def test_acquire_decrements_free_count(self) -> None:
        pool = MiniStackPool([4566, 4567])
        pool.acquire()
        assert pool.free_count == 1

    def test_acquire_returns_port_from_pool(self) -> None:
        pool = MiniStackPool([4566, 4567])
        port = pool.acquire()
        assert port in {4566, 4567}

    def test_release_increments_free_count(self) -> None:
        pool = MiniStackPool([4566, 4567])
        port = pool.acquire()
        pool.release(port)
        assert pool.free_count == 2


class TestMiniStackPoolExhaustion:
    def test_acquire_beyond_capacity_raises(self) -> None:
        pool = MiniStackPool([4566])
        pool.acquire()
        with pytest.raises(RuntimeError, match="exhausted"):
            pool.acquire()

    def test_empty_pool_raises_on_acquire(self) -> None:
        pool = MiniStackPool([])
        with pytest.raises(RuntimeError, match="exhausted"):
            pool.acquire()

    def test_can_acquire_again_after_release(self) -> None:
        pool = MiniStackPool([4566])
        pool.acquire()
        with pytest.raises(RuntimeError):
            pool.acquire()
        pool.release(4566)
        assert pool.acquire() == 4566


class TestMiniStackPoolRecycling:
    def test_released_port_is_reused(self) -> None:
        pool = MiniStackPool([4566])
        first = pool.acquire()
        pool.release(first)
        second = pool.acquire()
        assert second == first

    def test_multiple_cycles_stay_bounded(self) -> None:
        """Open+close 100 sessions on a pool of 4 ports — must never exhaust."""
        pool = MiniStackPool(range(4566, 4570))
        for _ in range(100):
            port = pool.acquire()
            pool.release(port)
        assert pool.free_count == 4

    def test_full_drain_then_full_refill(self) -> None:
        pool = MiniStackPool(range(4566, 4574))
        acquired = [pool.acquire() for _ in range(8)]
        assert pool.free_count == 0
        for port in acquired:
            pool.release(port)
        assert pool.free_count == 8


class TestMiniStackPoolConcurrency:
    def test_concurrent_acquire_no_duplicate_ports(self) -> None:
        """100 threads compete for 50 ports. Winners must hold unique ports,
        losers must see RuntimeError — no double-assignment.
        """
        pool = MiniStackPool(range(10000, 10050))
        acquired: list[int] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def worker() -> None:
            try:
                port = pool.acquire()
                with lock:
                    acquired.append(port)
            except RuntimeError as e:
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(acquired) == 50
        assert len(set(acquired)) == 50  # no duplicates
        assert len(errors) == 50
        assert pool.free_count == 0

    def test_concurrent_release_preserves_all_ports(self) -> None:
        """All 50 ports released concurrently end up back in the pool."""
        pool = MiniStackPool(range(10000, 10050))
        ports = [pool.acquire() for _ in range(50)]

        threads = [threading.Thread(target=pool.release, args=(p,)) for p in ports]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert pool.free_count == 50

    def test_acquire_release_cycle_under_contention(self) -> None:
        """10 threads acquire-release 50 times each against a pool of 3. No port is lost."""
        pool = MiniStackPool([4566, 4567, 4568])

        def churn() -> None:
            for _ in range(50):
                try:
                    p = pool.acquire()
                    pool.release(p)
                except RuntimeError:
                    pass  # contention — expected

        threads = [threading.Thread(target=churn) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert pool.free_count == 3


# ---------------------------------------------------------------------------
# make_env_factory — single-mode vs multi-mode branch
# ---------------------------------------------------------------------------


class TestFactorySingleMode:
    def test_pool_size_1_returns_no_pool(self) -> None:
        pool, factory = make_env_factory(pool_size=1, base_port=4566)
        assert pool is None
        assert callable(factory)

    def test_pool_size_0_returns_no_pool(self) -> None:
        """Treat 0 or negative the same as 1 — no pool, legacy behavior."""
        pool, factory = make_env_factory(pool_size=0, base_port=4566)
        assert pool is None

    def test_factory_returns_env_without_pool_release(self) -> None:
        _, factory = make_env_factory(pool_size=1, base_port=4566)
        env = factory()
        assert isinstance(env, AwsRlEnvironment)
        assert env._pool_release is None


class TestFactoryMultiMode:
    def test_pool_size_8_creates_pool_of_8(self) -> None:
        pool, _ = make_env_factory(pool_size=8, base_port=4566)
        assert pool is not None
        assert pool.free_count == 8

    def test_factory_acquires_port_from_pool(self) -> None:
        pool, factory = make_env_factory(pool_size=4, base_port=4566)
        assert pool is not None
        assert pool.free_count == 4
        env = factory()
        assert pool.free_count == 3
        assert env._pool_release is not None

    def test_env_bound_to_port_in_configured_range(self) -> None:
        pool, factory = make_env_factory(pool_size=4, base_port=5000)
        env = factory()
        url = env._backend._aws_infra_url
        # Port should be one of 5000..5003
        port = int(url.rsplit(":", 1)[-1])
        assert 5000 <= port < 5004

    def test_multiple_factory_calls_drain_pool(self) -> None:
        pool, factory = make_env_factory(pool_size=3, base_port=4566)
        assert pool is not None
        envs = [factory() for _ in range(3)]
        assert pool.free_count == 0
        with pytest.raises(RuntimeError, match="exhausted"):
            factory()
        # Keep envs referenced to avoid GC warning
        assert len(envs) == 3

    def test_envs_get_distinct_ports(self) -> None:
        _, factory = make_env_factory(pool_size=4, base_port=4566)
        envs = [factory() for _ in range(4)]
        urls = {e._backend._aws_infra_url for e in envs}
        assert len(urls) == 4  # all distinct

    def test_custom_base_port_is_respected(self) -> None:
        pool, factory = make_env_factory(pool_size=3, base_port=9000)
        env = factory()
        port = int(env._backend._aws_infra_url.rsplit(":", 1)[-1])
        assert 9000 <= port < 9003


# ---------------------------------------------------------------------------
# AwsRlEnvironment.close() — pool interaction
# ---------------------------------------------------------------------------


class TestEnvCloseReleasesPort:
    def test_close_returns_port_to_pool(self) -> None:
        pool, factory = make_env_factory(pool_size=4, base_port=4566)
        assert pool is not None
        env = factory()
        assert pool.free_count == 3
        # Mock the MiniStack scrub so close() doesn't try to hit the network
        with patch.object(env._backend, "reset_environment"):
            env.close()
        assert pool.free_count == 4

    def test_close_clears_pool_release_to_prevent_double_release(self) -> None:
        pool, factory = make_env_factory(pool_size=4, base_port=4566)
        env = factory()
        with patch.object(env._backend, "reset_environment"):
            env.close()
            env.close()  # second close must be a no-op
        assert pool.free_count == 4  # not 5

    def test_close_releases_port_even_if_scrub_fails(self) -> None:
        """If MiniStack is unreachable, close() still returns the port — leaking ports
        on network hiccups would drain the pool.
        """
        pool, factory = make_env_factory(pool_size=4, base_port=4566)
        env = factory()
        with patch.object(
            env._backend,
            "reset_environment",
            side_effect=ConnectionError("boom"),
        ):
            env.close()
        assert pool.free_count == 4

    def test_close_on_non_pooled_env_is_noop(self) -> None:
        _, factory = make_env_factory(pool_size=1, base_port=4566)
        env = factory()
        # Not from pool — no release callback to fire
        env.close()
        assert env._pool_release is None  # still None

    def test_close_invokes_backend_scrub(self) -> None:
        _, factory = make_env_factory(pool_size=2, base_port=4566)
        env = factory()
        with patch.object(env._backend, "reset_environment") as mock_scrub:
            env.close()
        mock_scrub.assert_called_once()


class TestFactoryConcurrencyIntegration:
    def test_concurrent_factory_calls_get_distinct_ports(self) -> None:
        """The factory + pool combo must hand out unique ports under contention."""
        _, factory = make_env_factory(pool_size=50, base_port=10000)
        envs: list[AwsRlEnvironment] = []
        lock = threading.Lock()

        def worker() -> None:
            env = factory()
            with lock:
                envs.append(env)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        ports = {int(e._backend._aws_infra_url.rsplit(":", 1)[-1]) for e in envs}
        assert len(ports) == 50

    def test_concurrent_close_returns_all_ports(self) -> None:
        pool, factory = make_env_factory(pool_size=20, base_port=10000)
        assert pool is not None
        envs = [factory() for _ in range(20)]
        assert pool.free_count == 0

        for env in envs:
            env._backend.reset_environment = lambda: None  # type: ignore[assignment]

        threads = [threading.Thread(target=e.close) for e in envs]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert pool.free_count == 20
