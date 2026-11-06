# Tests for the SQLite sliding window limiter: window maths, eviction, reserve/commit/release, persistence,
# and that concurrent threads and separate processes can never push a tenant over its limit.

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ratelimiter import RateLimited, RequestTooLarge, SlidingWindowLimiter


class Clock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def limiter(tmp_path, clock):
    return SlidingWindowLimiter(str(tmp_path / "limits.db"), limit=50_000, window=60.0, clock=clock)


def test_allows_up_to_exact_limit(limiter):
    limiter.reserve("a", 30_000)
    r = limiter.reserve("a", 20_000)
    assert r.remaining == 0
    with pytest.raises(RateLimited) as exc:
        limiter.reserve("a", 1)
    assert exc.value.used == 50_000


def test_window_slides_instead_of_resetting(limiter, clock):
    limiter.reserve("a", 30_000)
    clock.now += 30
    limiter.reserve("a", 20_000)
    clock.now += 29.9
    with pytest.raises(RateLimited) as exc:
        limiter.reserve("a", 1)
    assert exc.value.retry_after == pytest.approx(0.1)
    clock.now += 0.1
    limiter.reserve("a", 30_000)
    with pytest.raises(RateLimited):
        limiter.reserve("a", 1)


def test_retry_after_waits_for_enough_tokens_to_expire(limiter, clock):
    limiter.reserve("a", 10_000)
    clock.now += 10
    limiter.reserve("a", 20_000)
    clock.now += 10
    limiter.reserve("a", 20_000)
    clock.now += 10
    with pytest.raises(RateLimited) as exc:
        limiter.reserve("a", 15_000)
    assert exc.value.retry_after == pytest.approx(40.0)


def test_rejected_request_does_not_consume_budget(limiter):
    limiter.reserve("a", 45_000)
    for _ in range(5):
        with pytest.raises(RateLimited):
            limiter.reserve("a", 10_000)
    assert limiter.used("a") == 45_000
    limiter.reserve("a", 5_000)


def test_expired_rows_are_evicted_for_every_tenant(limiter, clock):
    for tenant in ("a", "b", "c"):
        limiter.reserve(tenant, 1_000)
    assert limiter.row_count() == 3
    clock.now += 60
    limiter.reserve("d", 1)
    assert limiter.row_count() == 1
    assert limiter.used("a") == 0


def test_commit_corrects_estimate_to_actual(limiter):
    r = limiter.reserve("a", 5_000)
    assert limiter.commit(r, 1_200) == 50_000 - 1_200
    assert limiter.used("a") == 1_200
    r2 = limiter.reserve("a", 100)
    limiter.commit(r2, 900)
    assert limiter.used("a") == 2_100


def test_release_frees_the_reservation(limiter):
    r = limiter.reserve("a", 50_000)
    limiter.release(r)
    assert limiter.used("a") == 0
    limiter.reserve("a", 50_000)


def test_single_request_over_limit_is_too_large(limiter):
    with pytest.raises(RequestTooLarge):
        limiter.reserve("a", 50_001)
    assert limiter.row_count() == 0


def test_tenants_are_isolated(limiter):
    limiter.reserve("a", 50_000)
    limiter.reserve("b", 50_000)
    with pytest.raises(RateLimited):
        limiter.reserve("a", 1)


def test_state_persists_on_disk(tmp_path, clock):
    path = str(tmp_path / "limits.db")
    SlidingWindowLimiter(path, clock=clock).reserve("a", 40_000)
    reopened = SlidingWindowLimiter(path, clock=clock)
    assert reopened.used("a") == 40_000
    with pytest.raises(RateLimited):
        reopened.reserve("a", 20_000)
    assert Path(path).stat().st_size > 0


def test_concurrent_threads_never_exceed_limit(tmp_path):
    limiter = SlidingWindowLimiter(str(tmp_path / "limits.db"))
    accepted = []
    barrier = threading.Barrier(40)

    def worker():
        barrier.wait()
        try:
            limiter.reserve("t", 5_000)
            accepted.append(1)
        except RateLimited:
            pass

    threads = [threading.Thread(target=worker) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(accepted) == 10
    assert limiter.used("t") == 50_000


def test_concurrent_processes_share_one_limit(tmp_path):
    db = tmp_path / "limits.db"
    SlidingWindowLimiter(str(db))
    start_at = time.time() + 2.0
    script = textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(ROOT)!r})
        from ratelimiter import RateLimited, SlidingWindowLimiter
        limiter = SlidingWindowLimiter({str(db)!r})
        time.sleep(max(0, {start_at} - time.time()))
        ok = 0
        for _ in range(20):
            try:
                limiter.reserve("shared", 2_000)
                ok += 1
            except RateLimited:
                pass
        print(ok)
    """)
    procs = [subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True) for _ in range(4)]
    counts = [int(p.communicate(timeout=60)[0].strip()) for p in procs]
    assert sum(counts) == 25
    assert SlidingWindowLimiter(str(db)).used("shared") == 50_000
