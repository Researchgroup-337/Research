"""Stage C - cross-session behavioural store (local, on-device).

Privacy by design:
  * only pseudonymised IDs (HMAC) are stored - never usernames,
  * message TEXT is never stored - only scores, counts and timestamps,
  * optional encryption at rest with SQLCipher (pip install sqlcipher3-binary),
  * retention purge + right-to-erasure (delete_user).
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field

SCHEMA = """
CREATE TABLE IF NOT EXISTS pair_state (
    offender TEXT NOT NULL,
    target   TEXT NOT NULL,
    r_total  INTEGER NOT NULL DEFAULT 0,   -- all messages to target
    r_hostile INTEGER NOT NULL DEFAULT 0,  -- hostile messages to target (raw count)
    r_eff    REAL NOT NULL DEFAULT 0,      -- decayed hostile count
    s_hostile INTEGER NOT NULL DEFAULT 0,  -- distinct sessions with >=1 hostile message (raw)
    s_eff    REAL NOT NULL DEFAULT 0,      -- decayed session recurrence
    last_session TEXT,                     -- last session counted in s
    first_ts REAL, last_ts REAL, last_hostile_ts REAL,
    last_threat_ts REAL,                   -- last explicit high-confidence threat
    t_ewma   REAL NOT NULL DEFAULT 0,
    t_max    REAL NOT NULL DEFAULT 0,
    recent_t TEXT NOT NULL DEFAULT '[]',
    risk     REAL NOT NULL DEFAULT 0,
    tier     TEXT NOT NULL DEFAULT 'Low',
    flags    TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (offender, target)
);
CREATE INDEX IF NOT EXISTS ix_pair_target ON pair_state(target);
CREATE TABLE IF NOT EXISTS target_hostile (
    target TEXT NOT NULL, offender TEXT NOT NULL, ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_th ON target_hostile(target, ts);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL, kind TEXT NOT NULL, offender TEXT, target TEXT,
    old_tier TEXT, new_tier TEXT, risk REAL, detail TEXT
);
"""


@dataclass
class PairState:
    offender: str
    target: str
    r_total: int = 0
    r_hostile: int = 0
    r_eff: float = 0.0
    s_hostile: int = 0
    s_eff: float = 0.0
    last_session: str | None = None
    first_ts: float | None = None
    last_ts: float | None = None
    last_hostile_ts: float | None = None
    last_threat_ts: float | None = None
    t_ewma: float = 0.0
    t_max: float = 0.0
    recent_t: list = field(default_factory=list)
    risk: float = 0.0
    tier: str = "Low"
    flags: list = field(default_factory=list)


COLS = list(PairState.__dataclass_fields__.keys())


class BehaviorStore:
    def __init__(self, path: str = ":memory:", encryption_key: str | None = None):
        self.encrypted = False
        if encryption_key:
            try:
                import sqlcipher3
                self.conn = sqlcipher3.connect(path, check_same_thread=False)
                self.conn.execute(f"PRAGMA key = '{encryption_key.replace(chr(39), '')}'")
                self.encrypted = True
            except ImportError:
                raise RuntimeError("encryption_key given but sqlcipher3 is not installed (pip install sqlcipher3-binary)")
        else:
            self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL" if path != ":memory:" else "PRAGMA journal_mode=MEMORY")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self._cache: dict[tuple[str, str], PairState] = {}

    # ------------------------------------------------------------ pairs
    def get_pair(self, offender: str, target: str) -> PairState:
        key = (offender, target)
        if key in self._cache:
            return self._cache[key]
        row = self.conn.execute(f"SELECT {','.join(COLS)} FROM pair_state WHERE offender=? AND target=?", key).fetchone()
        if row is None:
            st = PairState(offender, target)
        else:
            d = dict(zip(COLS, row))
            d["recent_t"] = json.loads(d["recent_t"])
            d["flags"] = json.loads(d["flags"])
            st = PairState(**d)
        self._cache[key] = st
        return st

    def put_pair(self, st: PairState, commit: bool = True) -> None:
        vals = [getattr(st, c) for c in COLS]
        vals[COLS.index("recent_t")] = json.dumps(st.recent_t)
        vals[COLS.index("flags")] = json.dumps(st.flags)
        self.conn.execute(f"INSERT OR REPLACE INTO pair_state ({','.join(COLS)}) VALUES ({','.join('?'*len(COLS))})", vals)
        if commit:
            self.conn.commit()

    def pairs_for_offender(self, offender: str) -> list[PairState]:
        rows = self.conn.execute("SELECT target FROM pair_state WHERE offender=?", (offender,)).fetchall()
        return [self.get_pair(offender, r[0]) for r in rows]

    def pairs_for_target(self, target: str) -> list[PairState]:
        rows = self.conn.execute("SELECT offender FROM pair_state WHERE target=?", (target,)).fetchall()
        return [self.get_pair(r[0], target) for r in rows]

    def all_pairs(self) -> list[PairState]:
        self.flush()
        rows = self.conn.execute("SELECT offender, target FROM pair_state").fetchall()
        return [self.get_pair(o, t) for o, t in rows]

    def top_pairs(self, limit: int = 50) -> list[PairState]:
        rows = self.conn.execute("SELECT offender, target FROM pair_state ORDER BY risk DESC LIMIT ?", (limit,)).fetchall()
        return [self.get_pair(o, t) for o, t in rows]

    # ------------------------------------------------------------ victim-centric pressure
    def add_target_hostile(self, target: str, offender: str, ts: float) -> None:
        self.conn.execute("INSERT INTO target_hostile VALUES (?,?,?)", (target, offender, ts))

    def distinct_hostile_senders(self, target: str, since_ts: float) -> int:
        return self.conn.execute("SELECT COUNT(DISTINCT offender) FROM target_hostile WHERE target=? AND ts>=?",
                                 (target, since_ts)).fetchone()[0]

    # ------------------------------------------------------------ alerts
    def add_alert(self, ts, kind, offender, target, old_tier, new_tier, risk, detail: dict) -> int:
        cur = self.conn.execute("INSERT INTO alerts (ts,kind,offender,target,old_tier,new_tier,risk,detail) VALUES (?,?,?,?,?,?,?,?)",
                                (ts, kind, offender, target, old_tier, new_tier, risk, json.dumps(detail)))
        return cur.lastrowid

    def recent_alerts(self, limit: int = 100) -> list[dict]:
        rows = self.conn.execute("SELECT id,ts,kind,offender,target,old_tier,new_tier,risk,detail FROM alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        keys = ["id", "ts", "kind", "offender", "target", "old_tier", "new_tier", "risk", "detail"]
        return [dict(zip(keys, r)) | {"detail": json.loads(r[-1])} for r in rows]

    # ------------------------------------------------------------ privacy operations
    def delete_user(self, user_hash: str) -> int:
        """Right to erasure: remove every record where the user is offender or target."""
        self.flush()
        n = self.conn.execute("DELETE FROM pair_state WHERE offender=? OR target=?", (user_hash, user_hash)).rowcount
        self.conn.execute("DELETE FROM target_hostile WHERE offender=? OR target=?", (user_hash, user_hash))
        self.conn.execute("DELETE FROM alerts WHERE offender=? OR target=?", (user_hash, user_hash))
        self.conn.commit()
        self._cache = {k: v for k, v in self._cache.items() if user_hash not in k}
        return n

    def purge(self, now: float, retention_days: int, pressure_window_days: float) -> int:
        """Retention policy: drop pairs inactive for longer than retention_days."""
        self.flush()
        cutoff = now - retention_days * 86400
        n = self.conn.execute("DELETE FROM pair_state WHERE last_ts < ?", (cutoff,)).rowcount
        self.conn.execute("DELETE FROM target_hostile WHERE ts < ?", (now - pressure_window_days * 86400,))
        self.conn.commit()
        self._cache.clear()
        return n

    def flush(self) -> None:
        for st in self._cache.values():
            self.put_pair(st, commit=False)
        self.conn.commit()

    def stats(self) -> dict:
        q = lambda s: self.conn.execute(s).fetchone()[0]  # noqa: E731
        self.flush()
        return {"pairs": q("SELECT COUNT(*) FROM pair_state"),
                "offenders": q("SELECT COUNT(DISTINCT offender) FROM pair_state"),
                "alerts": q("SELECT COUNT(*) FROM alerts"), "encrypted": self.encrypted,
                "tiers": dict(self.conn.execute("SELECT tier, COUNT(*) FROM pair_state GROUP BY tier").fetchall())}

    def close(self):
        self.flush()
        self.conn.close()


def now_ts() -> float:
    return time.time()
