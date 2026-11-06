# Token-aware sliding window rate limiter stored in an on-disk SQLite file. Each tenant has a token budget
# over a rolling 60 s window. Tokens are reserved up front, then corrected to actual usage or released.

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass


class RateLimited(Exception):
    def __init__(self, retry_after: float, used: int, limit: int):
        super().__init__(f"rate limited, retry after {retry_after:.2f}s")
        self.retry_after = retry_after
        self.used = used
        self.limit = limit


class RequestTooLarge(Exception):
    def __init__(self, tokens: int, limit: int):
        super().__init__(f"request needs {tokens} tokens, limit is {limit}")
        self.tokens = tokens
        self.limit = limit


@dataclass(frozen=True)
class Reservation:
    id: int
    tenant: str
    tokens: int
    remaining: int


class SlidingWindowLimiter:
    def __init__(
        self,
        path: str,
        limit: int = 50_000,
        window: float = 60.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = path
        self.limit = limit
        self.window = window
        self.clock = clock
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS token_usage ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " tenant TEXT NOT NULL,"
                " ts REAL NOT NULL,"
                " tokens INTEGER NOT NULL CHECK (tokens >= 0))"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_tenant_ts ON token_usage (tenant, ts)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_ts ON token_usage (ts)")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10.0, isolation_level=None)
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def reserve(self, tenant: str, tokens: int) -> Reservation:
        if tokens > self.limit:
            raise RequestTooLarge(tokens, self.limit)
        now = self.clock()
        with closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute("DELETE FROM token_usage WHERE ts <= ?", (now - self.window,))
                rows = conn.execute(
                    "SELECT ts, tokens FROM token_usage WHERE tenant = ? ORDER BY ts", (tenant,)
                ).fetchall()
                used = sum(t for _, t in rows)
                if used + tokens > self.limit:
                    conn.execute("COMMIT")
                    raise RateLimited(self._retry_after(rows, used, tokens, now), used, self.limit)
                cur = conn.execute(
                    "INSERT INTO token_usage (tenant, ts, tokens) VALUES (?, ?, ?)", (tenant, now, tokens)
                )
                conn.execute("COMMIT")
            except RateLimited:
                raise
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return Reservation(cur.lastrowid, tenant, tokens, self.limit - used - tokens)

    def _retry_after(self, rows: list[tuple[float, int]], used: int, tokens: int, now: float) -> float:
        freed = 0
        for ts, t in rows:
            freed += t
            if used - freed + tokens <= self.limit:
                return max(0.0, ts + self.window - now)
        return self.window

    def commit(self, reservation: Reservation, actual_tokens: int) -> int:
        with closing(self._connect()) as conn:
            conn.execute("UPDATE token_usage SET tokens = ? WHERE id = ?", (max(0, actual_tokens), reservation.id))
        return reservation.remaining + reservation.tokens - max(0, actual_tokens)

    def release(self, reservation: Reservation) -> None:
        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM token_usage WHERE id = ?", (reservation.id,))

    def used(self, tenant: str) -> int:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant = ? AND ts > ?",
                (tenant, self.clock() - self.window),
            ).fetchone()
        return int(row[0])

    def row_count(self) -> int:
        with closing(self._connect()) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM token_usage").fetchone()[0])
