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


class TestServerAppImportIsSafeForLegacyPoolSizes:
    """Regression: `AWS_RL_ENV_POOL_SIZE=0` used to crash at module import
    because OpenEnv's create_app rejects `max_concurrent_envs=0`. The server
    now clamps the raw env var to >= 1 so legacy-style zero / negative values
    silently fall back to single-MiniStack mode.
    """

    def _import_server_app(self, pool_size_env: str) -> int:
        """Import server.app in a fresh subprocess with a controlled env var.

        Returns the POOL_SIZE the module settled on after clamping.
        """
        import os
        import subprocess
        import sys

        code = "import server.app as m;import sys;sys.stdout.write(str(m.POOL_SIZE))"
        env = {**os.environ, "AWS_RL_ENV_POOL_SIZE": pool_size_env}
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"server.app import crashed with POOL_SIZE={pool_size_env!r}: "
            f"stderr={result.stderr}"
        )
        return int(result.stdout.strip().splitlines()[-1])

    def test_pool_size_zero_clamps_to_one(self) -> None:
        assert self._import_server_app("0") == 1

    def test_pool_size_negative_clamps_to_one(self) -> None:
        assert self._import_server_app("-5") == 1

    def test_pool_size_one_is_unchanged(self) -> None:
        assert self._import_server_app("1") == 1

    def test_pool_size_eight_is_unchanged(self) -> None:
        assert self._import_server_app("8") == 8


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


# ---------------------------------------------------------------------------
# Web playground coexistence with the MiniStack pool
# ---------------------------------------------------------------------------


def _run_in_subprocess(env_overrides: dict[str, str], code: str) -> tuple[int, str, str]:
    """Run `code` in a fresh subprocess with the given env overrides.

    Mirrors the pattern used by TestServerAppImportIsSafeForLegacyPoolSizes
    to avoid module-cache pollution across env-var changes.
    """
    import os
    import subprocess
    import sys

    env = {**os.environ, **env_overrides}
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout, result.stderr


class TestWebRoutesMountUnconditionally:
    """The web playground used to be gated on POOL_SIZE <= 1. It now mounts
    regardless of pool size, with a dedicated lazy MiniStack on
    AWS_RL_ENV_WEB_MINISTACK_PORT.
    """

    def test_web_routes_present_when_pool_size_8(self) -> None:
        code = (
            "import server.app as m;"
            "paths = {getattr(r, 'path', None) for r in m.app.routes};"
            "import sys;"
            "missing = {'/web', '/web/reset', '/web/state', '/web/step', '/web/solution'} - paths;"
            "sys.stdout.write('MISSING=' + repr(missing))"
        )
        rc, out, err = _run_in_subprocess({"AWS_RL_ENV_POOL_SIZE": "8"}, code)
        assert rc == 0, f"import failed: {err}"
        assert "MISSING=set()" in out, out

    def test_web_routes_present_when_pool_size_1(self) -> None:
        code = (
            "import server.app as m;"
            "paths = {getattr(r, 'path', None) for r in m.app.routes};"
            "import sys;"
            "missing = {'/web', '/web/reset', '/web/state', '/web/step', '/web/solution'} - paths;"
            "sys.stdout.write('MISSING=' + repr(missing))"
        )
        rc, out, err = _run_in_subprocess({"AWS_RL_ENV_POOL_SIZE": "1"}, code)
        assert rc == 0, f"import failed: {err}"
        assert "MISSING=set()" in out, out


class TestWebMiniStackPortConflictDetection:
    """The startup-time guard refuses to boot if the configured web port falls
    inside the pool's port range. Without it, a WebSocket session could acquire
    the same port the web _env writes to and corrupt state in both directions.
    """

    def test_collision_inside_pool_range_raises(self) -> None:
        code = "import server.app"
        rc, _, err = _run_in_subprocess(
            {
                "AWS_RL_ENV_POOL_SIZE": "8",
                "AWS_RL_ENV_MINISTACK_BASE_PORT": "4566",
                "AWS_RL_ENV_WEB_MINISTACK_PORT": "4570",  # inside [4566..4573]
            },
            code,
        )
        assert rc != 0
        assert "collides with pool range" in err

    def test_web_port_just_below_pool_range_is_allowed(self) -> None:
        code = "import server.app"
        rc, _, err = _run_in_subprocess(
            {
                "AWS_RL_ENV_POOL_SIZE": "8",
                "AWS_RL_ENV_MINISTACK_BASE_PORT": "4566",
                "AWS_RL_ENV_WEB_MINISTACK_PORT": "4565",  # default
            },
            code,
        )
        assert rc == 0, err

    def test_web_port_just_above_pool_range_is_allowed(self) -> None:
        code = "import server.app"
        rc, _, err = _run_in_subprocess(
            {
                "AWS_RL_ENV_POOL_SIZE": "8",
                "AWS_RL_ENV_MINISTACK_BASE_PORT": "4566",
                "AWS_RL_ENV_WEB_MINISTACK_PORT": "4574",  # one past 4573
            },
            code,
        )
        assert rc == 0, err

    def test_collision_check_skipped_when_pool_size_1(self) -> None:
        """POOL_SIZE=1 means no pool object exists, so the constant web port
        is allowed to coincide with BASE_PORT (it just means the web env
        shares the lone MiniStack). Backward-compat for legacy single-mode.
        """
        code = "import server.app"
        rc, _, err = _run_in_subprocess(
            {
                "AWS_RL_ENV_POOL_SIZE": "1",
                "AWS_RL_ENV_MINISTACK_BASE_PORT": "4566",
                "AWS_RL_ENV_WEB_MINISTACK_PORT": "4566",
            },
            code,
        )
        assert rc == 0, err

    def test_collision_check_skipped_when_backend_aws(self) -> None:
        """BACKEND_TYPE=aws skips the pool entirely (all sessions share
        AwsStrategy), so a "collision" with the pool's range is hypothetical
        — the pool object is never constructed. Refusing to boot here would
        be a false positive.
        """
        code = "import server.app"
        rc, _, err = _run_in_subprocess(
            {
                "AWS_RL_ENV_POOL_SIZE": "8",
                "AWS_RL_ENV_MINISTACK_BASE_PORT": "4566",
                "AWS_RL_ENV_WEB_MINISTACK_PORT": "4570",  # would collide if simulator
                "BACKEND_TYPE": "aws",
            },
            code,
        )
        assert rc == 0, err


class TestWebEnvLazyConstruction:
    def test_web_env_is_none_immediately_after_import(self) -> None:
        """Lazy: the dedicated MiniStack should NOT spawn until a /web/*
        request arrives. Importing the module must not subprocess anything.
        """
        code = (
            "import server.app as m;"
            "import sys;"
            "sys.stdout.write('\\nRESULT=' + ('NONE' if m._web_env is None else 'NOT_NONE'))"
        )
        rc, out, err = _run_in_subprocess({"AWS_RL_ENV_POOL_SIZE": "8"}, code)
        assert rc == 0, err
        assert out.strip().splitlines()[-1] == "RESULT=NONE"

    def test_get_web_env_legacy_uses_default_port_for_pool_size_1(self) -> None:
        """POOL_SIZE=1: web env shares the single MiniStack on :4566 — the
        original behavior, locked down so it doesn't drift.
        """
        code = (
            "import server.app as m;"
            "env = m._get_web_env();"
            "import sys;"
            "sys.stdout.write('\\nRESULT=' + env._backend._aws_infra_url)"
        )
        rc, out, err = _run_in_subprocess({"AWS_RL_ENV_POOL_SIZE": "1"}, code)
        assert rc == 0, err
        assert out.strip().splitlines()[-1] == "RESULT=http://localhost:4566"

    def test_get_web_env_uses_aws_strategy_when_backend_aws(self) -> None:
        """BACKEND_TYPE=aws: web env wires AwsStrategy too. No MiniStack spawn.
        Fixes the latent inconsistency where the web playground always used
        the simulator regardless of training backend.
        """
        code = (
            "import server.app as m;"
            "from server.services.aws_strategy import AwsStrategy;"
            "env = m._get_web_env();"
            "import sys;"
            "sys.stdout.write('\\nRESULT=' + ('AWS' if isinstance(env._backend, AwsStrategy) else 'NOT_AWS'))"
        )
        rc, out, err = _run_in_subprocess(
            {"AWS_RL_ENV_POOL_SIZE": "8", "BACKEND_TYPE": "aws"},
            code,
        )
        assert rc == 0, err
        assert out.strip().splitlines()[-1] == "RESULT=AWS"


class TestSpawnWebMiniStackShortCircuit:
    """`_spawn_web_ministack` must not subprocess if the port is already
    listening — otherwise a server restart would race against the existing
    detached MiniStack and stall on the bind check.
    """

    def test_does_not_spawn_when_port_already_listening(self) -> None:
        import socket

        from server.app import _spawn_web_ministack

        # Bind an ephemeral port to simulate a MiniStack already running.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sentinel:
            sentinel.bind(("127.0.0.1", 0))
            sentinel.listen(1)
            port = sentinel.getsockname()[1]

            with patch("server.app.subprocess.Popen") as popen:
                _spawn_web_ministack(port, timeout_s=0.5)

            popen.assert_not_called()

    def test_raises_on_bind_timeout(self) -> None:
        """If the spawned MiniStack never binds, raise instead of hanging."""
        from server.app import _spawn_web_ministack

        # Pick a port that is almost certainly free; mock Popen so nothing
        # actually starts. _spawn_web_ministack should poll and time out.
        with patch("server.app.subprocess.Popen"):
            with pytest.raises(RuntimeError, match="failed to bind"):
                _spawn_web_ministack(port=1, timeout_s=0.3)


class TestGetWebEnvAdversarial:
    """Stress-test _get_web_env against the failure modes a real deployment
    will eventually hit: concurrent first-request races, ministack-not-installed,
    and spawn timeouts.

    Each test patches at the module level inside an isolated subprocess so
    real ministacks are never spawned.
    """

    def test_concurrent_first_requests_spawn_at_most_once(self) -> None:
        """N threads racing on the cold start must result in exactly one
        Popen call. The double-checked lock + cached _web_env enforce this.
        Otherwise a busy /web/* moment at boot would spawn N ministacks all
        fighting for the same port.
        """
        code = """
import sys, threading
from unittest.mock import patch
import server.app as m
with patch('server.app._spawn_web_ministack') as spawn:
    spawn.return_value = None
    def call():
        m._get_web_env()
    threads = [threading.Thread(target=call) for _ in range(20)]
    for t in threads: t.start()
    for t in threads: t.join()
    sys.stdout.write('\\nRESULT=' + str(spawn.call_count))
"""
        rc, out, err = _run_in_subprocess({"AWS_RL_ENV_POOL_SIZE": "8"}, code)
        assert rc == 0, err
        assert out.strip().splitlines()[-1] == "RESULT=1"

    def test_get_web_env_does_not_spawn_when_backend_aws(self) -> None:
        """BACKEND_TYPE=aws path takes the AwsStrategy branch and never
        subprocesses ministack — even with POOL_SIZE=8.
        """
        code = """
import sys
from unittest.mock import patch
import server.app as m
with patch('server.app.subprocess.Popen') as popen:
    m._get_web_env()
    sys.stdout.write('\\nRESULT=' + str(popen.call_count))
"""
        rc, out, err = _run_in_subprocess(
            {"AWS_RL_ENV_POOL_SIZE": "8", "BACKEND_TYPE": "aws"},
            code,
        )
        assert rc == 0, err
        assert out.strip().splitlines()[-1] == "RESULT=0"

    def test_get_web_env_does_not_spawn_when_pool_size_1(self) -> None:
        """Legacy POOL_SIZE=1 path shares the lone pool MiniStack on :4566
        and never spawns a separate web MiniStack.
        """
        code = """
import sys
from unittest.mock import patch
import server.app as m
with patch('server.app.subprocess.Popen') as popen:
    m._get_web_env()
    sys.stdout.write('\\nRESULT=' + str(popen.call_count))
"""
        rc, out, err = _run_in_subprocess({"AWS_RL_ENV_POOL_SIZE": "1"}, code)
        assert rc == 0, err
        assert out.strip().splitlines()[-1] == "RESULT=0"

    def test_get_web_env_retries_after_spawn_failure(self) -> None:
        """If the first spawn fails (e.g., ministack not installed yet, or
        the bind timed out), _web_env stays None so a later request can
        retry instead of permanently caching the failure.
        """
        code = """
import sys
from unittest.mock import patch
import server.app as m
with patch('server.app._spawn_web_ministack', side_effect=RuntimeError('boom')):
    failed = False
    try:
        m._get_web_env()
    except RuntimeError:
        failed = True
    assert failed, 'expected first call to raise'
    assert m._web_env is None, '_web_env must stay None after spawn failure'
sys.stdout.write('\\nRESULT=ok')
"""
        rc, out, err = _run_in_subprocess({"AWS_RL_ENV_POOL_SIZE": "8"}, code)
        assert rc == 0, err
        assert out.strip().splitlines()[-1] == "RESULT=ok"

    def test_pool_factory_capacity_independent_of_web_env(self) -> None:
        """The web _env is a module-level singleton, NOT produced by the
        WebSocket factory. So a pool of 8 still hands out 8 distinct ports;
        the web env doesn't steal a slot. Critical for the user's "8 WS +
        web UI" goal.
        """
        pool, factory = make_env_factory(pool_size=8, base_port=4566)
        assert pool is not None
        envs = [factory() for _ in range(8)]
        assert pool.free_count == 0
        # 9th must fail — same as before this change
        with pytest.raises(RuntimeError, match="exhausted"):
            factory()
        # Sanity: all 8 ports distinct, none equal to 4565 (web port)
        ports = {int(e._backend._aws_infra_url.rsplit(":", 1)[-1]) for e in envs}
        assert len(ports) == 8
        assert 4565 not in ports
