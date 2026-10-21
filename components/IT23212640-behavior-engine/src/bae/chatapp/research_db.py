"""Research log for SafeChat Lab.

This is the DATASET being collected, kept separate from the engine's privacy-preserving
store. It holds message text because building a labelled cross-session corpus is the point
of the study; participants consent to this, use study codes (P01, P02, ...) rather than real
identities, and display names are never exported.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS participants (
    code TEXT PRIMARY KEY, display_name TEXT NOT NULL, token TEXT UNIQUE NOT NULL,
    is_bot INTEGER NOT NULL DEFAULT 0, consent_ts REAL, created_ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS rooms (
    room_id TEXT PRIMARY KEY, name TEXT NOT NULL, owner TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY, room_id TEXT NOT NULL, started_ts REAL NOT NULL, ended_ts REAL
);
CREATE TABLE IF NOT EXISTS assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT, participant TEXT NOT NULL, role TEXT NOT NULL,
    target TEXT, room_id TEXT, note TEXT, created_ts REAL NOT NULL, active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS messages (
    message_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, room_id TEXT NOT NULL, ts REAL NOT NULL,
    sender TEXT NOT NULL, text TEXT NOT NULL, reply_to TEXT, mentions TEXT NOT NULL DEFAULT '[]',
    target TEXT, target_method TEXT, from_bank TEXT,
    T REAL, p_threat REAL, labels TEXT, risk REAL, tier TEXT, alerts TEXT, engine_ts REAL
);
CREATE INDEX IF NOT EXISTS ix_msg_session ON messages(session_id, ts);
CREATE TABLE IF NOT EXISTS pair_labels (
    sender TEXT NOT NULL, target TEXT NOT NULL, label INTEGER NOT NULL, annotator TEXT,
    note TEXT, ts REAL NOT NULL, PRIMARY KEY (sender, target)
);
CREATE TABLE IF NOT EXISTS label_reviews (
    conflict_id TEXT PRIMARY KEY, kind TEXT NOT NULL, sender TEXT, target TEXT,
    decision TEXT NOT NULL, detail TEXT, annotator TEXT, ts REAL NOT NULL
);
"""
# Engine outputs logged per message in addition to the original columns (added to older databases
# on start-up; NULL for messages logged before they existed).
ENGINE_COLUMNS = {"hostile": "INTEGER", "pair_hostile": "INTEGER", "pair_sessions": "INTEGER", "flags": "TEXT",
                  # overall (user-level) status: the sender's highest tier / risk across all their targets
                  "user_tier": "TEXT", "user_risk": "REAL"}

ROLES = {
    "normal": "Chat normally about the stream.",
    "bystander": "Chat normally. If you see someone being targeted, you may react as you naturally would.",
    "banter": "Joke around with your assigned friend using rough but friendly language. Not harassment.",
    "harasser": "Over several sessions, keep targeting your assigned person. Use the message bank if you prefer not to write it yourself.",
    "subtle_harasser": "Keep targeting your assigned person across sessions, but stay mostly mild: sarcasm, put-downs, 'watching you'.",
    "escalator": "Start mild with your assigned person; become more hostile in later sessions.",
    "burst": "In ONE session only, send many hostile messages to your assigned person, then behave normally.",
    "troll": "Make occasional hostile remarks to different people; don't focus on one person.",
}
POSITIVE_ROLES = {"harasser", "subtle_harasser", "escalator"}
BANK_ROLES = {"harasser", "subtle_harasser", "escalator", "burst", "troll", "banter"}
# Roles that are only meaningful towards one assigned person: saving them without a target would
# silently turn every pair of that sender into a role-derived 0.
TARGET_ROLES = POSITIVE_ROLES | {"burst", "banter"}
# Roles whose pairs are never repeat harassment by role; a manual 1 on them needs review.
NEGATIVE_ROLES = {"normal", "bystander", "troll", "burst", "banter"}


class ResearchDB:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL" if path != ":memory:" else "PRAGMA journal_mode=MEMORY")
        self.conn.executescript(SCHEMA)
        have = {r[1] for r in self.conn.execute("PRAGMA table_info(messages)")}
        for col, typ in ENGINE_COLUMNS.items():
            if col not in have:
                self.conn.execute(f"ALTER TABLE messages ADD COLUMN {col} {typ}")
        self.conn.commit()

    def q(self, sql, args=()):
        return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def one(self, sql, args=()):
        r = self.conn.execute(sql, args).fetchone()
        return dict(r) if r else None

    # --------------------------------------------------------------- participants
    def add_participant(self, display_name: str, is_bot: bool = False, code: str | None = None) -> dict:
        n = self.conn.execute("SELECT COUNT(*) FROM participants").fetchone()[0] + 1
        code = code or (f"B{n:03d}" if is_bot else f"P{n:03d}")
        token = secrets.token_urlsafe(9)
        self.conn.execute("INSERT INTO participants VALUES (?,?,?,?,?,?)",
                          (code, display_name.strip()[:32], token, int(is_bot), time.time() if is_bot else None, time.time()))
        self.conn.commit()
        return {"code": code, "display_name": display_name, "token": token, "is_bot": is_bot}

    def by_token(self, token: str):
        return self.one("SELECT * FROM participants WHERE token=?", (token,))

    def consent(self, code: str):
        self.conn.execute("UPDATE participants SET consent_ts=? WHERE code=?", (time.time(), code))
        self.conn.commit()

    def participants(self):
        return self.q("SELECT code, display_name, token, is_bot, consent_ts FROM participants ORDER BY code")

    # --------------------------------------------------------------- rooms & sessions
    def add_room(self, name: str, owner: str) -> dict:
        n = self.conn.execute("SELECT COUNT(*) FROM rooms").fetchone()[0] + 1
        rid = f"room{n}"
        self.conn.execute("INSERT INTO rooms VALUES (?,?,?)", (rid, name[:40], owner))
        self.conn.commit()
        return {"room_id": rid, "name": name, "owner": owner}

    def rooms(self):
        rooms = self.q("SELECT * FROM rooms ORDER BY room_id")
        for r in rooms:
            r["active_session"] = self.active_session(r["room_id"])
        return rooms

    def room(self, rid):
        return self.one("SELECT * FROM rooms WHERE room_id=?", (rid,))

    def active_session(self, rid):
        r = self.one("SELECT session_id FROM sessions WHERE room_id=? AND ended_ts IS NULL ORDER BY started_ts DESC LIMIT 1", (rid,))
        return r["session_id"] if r else None

    def start_session(self, rid) -> str:
        cur = self.active_session(rid)
        if cur:
            return cur
        n = self.conn.execute("SELECT COUNT(*) FROM sessions WHERE room_id=?", (rid,)).fetchone()[0] + 1
        sid = f"{rid}_s{n:03d}"
        self.conn.execute("INSERT INTO sessions VALUES (?,?,?,NULL)", (sid, rid, time.time()))
        self.conn.commit()
        return sid

    def end_session(self, rid):
        sid = self.active_session(rid)
        if sid:
            self.conn.execute("UPDATE sessions SET ended_ts=? WHERE session_id=?", (time.time(), sid))
            self.conn.commit()
        return sid

    def lab_clock(self, session_id: str, now: float, days_per_session: float) -> float:
        """Virtual time for the engine. In a lab, sessions run minutes apart but stand for live
        streams days apart. Session k (1-based, across all rooms) is mapped to
        first_start + (k-1)*days_per_session days, plus the real time elapsed inside it.
        days_per_session = 0 disables the mapping (engine sees real wall-clock time)."""
        if days_per_session <= 0:
            return now
        rows = self.q("SELECT session_id, started_ts FROM sessions ORDER BY started_ts")
        first = rows[0]["started_ts"]
        k = next(i for i, r in enumerate(rows) if r["session_id"] == session_id)
        return first + k * days_per_session * 86400 + (now - rows[k]["started_ts"])

    def sessions(self):
        return self.q("""SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id=s.session_id) AS messages,
                         (SELECT COUNT(DISTINCT sender) FROM messages m WHERE m.session_id=s.session_id) AS senders
                         FROM sessions s ORDER BY started_ts""")

    # --------------------------------------------------------------- roles & labels
    def assign(self, participant, role, target=None, room_id=None, note=None):
        if role not in ROLES:
            raise ValueError(f"unknown role {role}")
        if role in TARGET_ROLES and not target:
            raise ValueError(f"Role '{role}' needs a target: choose who this participant is aimed at, then save again.")
        if target and target == participant:
            raise ValueError("A participant cannot be their own target.")
        self.conn.execute("UPDATE assignments SET active=0 WHERE participant=?", (participant,))
        self.conn.execute("INSERT INTO assignments (participant, role, target, room_id, note, created_ts) VALUES (?,?,?,?,?,?)",
                          (participant, role, target, room_id, note, time.time()))
        self.conn.commit()

    def role_of(self, code):
        return self.one("SELECT role, target, room_id, note FROM assignments WHERE participant=? AND active=1", (code,))

    def assignments(self):
        return self.q("SELECT participant, role, target, room_id, note, created_ts, active FROM assignments ORDER BY id")

    def set_label(self, sender, target, label: int, annotator="researcher", note=None):
        self.conn.execute("INSERT OR REPLACE INTO pair_labels VALUES (?,?,?,?,?,?)", (sender, target, int(label), annotator, note, time.time()))
        self.conn.commit()

    def labels(self):
        return self.q("SELECT * FROM pair_labels")

    def clear_label(self, sender, target) -> bool:
        cur = self.conn.execute("DELETE FROM pair_labels WHERE sender=? AND target=?", (sender, target))
        self.conn.commit()
        return cur.rowcount > 0

    # --------------------------------------------------------------- label integrity
    def label_conflicts(self) -> list[dict]:
        """Role assignments and manual labels that disagree, or roles changed after the participant
        had already chatted. Each conflict has a stable id that includes the timestamp of the label or
        assignment involved, so a later change raises a new conflict instead of reusing an old review."""
        manual = {(m["sender"], m["target"]): m for m in self.labels()}
        hist: dict[str, list[dict]] = {}
        for a in self.q("SELECT * FROM assignments ORDER BY id"):
            hist.setdefault(a["participant"], []).append(a)
        active = {p: next((a for a in reversed(h) if a["active"]), None) for p, h in hist.items()}
        first_msg = {r["sender"]: r["t"] for r in self.q("SELECT sender, MIN(ts) AS t FROM messages GROUP BY sender")}
        out = []
        for p, a in active.items():
            if not a:
                continue
            if a["role"] in POSITIVE_ROLES and not a["target"]:
                out.append({"id": f"missing_target:{p}:{a['created_ts']}", "kind": "role_missing_target", "sender": p,
                            "target": None, "role": a["role"],
                            "detail": f"Role '{a['role']}' has no target, so every pair of {p} gets role label 0."})
            if a["role"] in POSITIVE_ROLES and a["target"]:
                m = manual.get((p, a["target"]))
                if m and m["label"] == 0:
                    out.append({"id": f"positive_role_manual_0:{p}:{a['target']}:{m['ts']}", "kind": "positive_role_manual_0",
                                "sender": p, "target": a["target"], "role": a["role"], "manual_label": 0, "role_label": 1,
                                "detail": f"Role '{a['role']}' targets {a['target']} (role label 1), but a manual label 0 overrides it."})
        for (s, t), m in manual.items():
            if m["label"] != 1:
                continue
            a = active.get(s)
            role = a["role"] if a else "normal"
            if role in NEGATIVE_ROLES:
                out.append({"id": f"manual_1_negative_role:{s}:{t}:{m['ts']}", "kind": "manual_1_negative_role",
                            "sender": s, "target": t, "role": role, "manual_label": 1, "role_label": 0,
                            "detail": f"Manual label 1, but the sender's role is '{role}' (role label 0)."})
        for p, h in hist.items():
            if p not in first_msg:
                continue
            prev = None  # no assignment yet = implicit 'normal'
            for a in h:
                if a["created_ts"] > first_msg[p]:
                    old = (prev["role"], prev["target"]) if prev else ("normal", None)
                    if old != (a["role"], a["target"]):
                        n = self.one("SELECT COUNT(*) AS n FROM messages WHERE sender=? AND ts < ?", (p, a["created_ts"]))["n"]
                        out.append({"id": f"role_changed_after_messages:{p}:{a['created_ts']}", "kind": "role_changed_after_messages",
                                    "sender": p, "target": a["target"], "role": a["role"],
                                    "old_role": old[0], "old_target": old[1], "new_role": a["role"], "new_target": a["target"],
                                    "changed_ts": a["created_ts"], "messages_before_change": n,
                                    "detail": f"Role changed from '{old[0]}' (target {old[1] or 'none'}) to '{a['role']}' "
                                              f"(target {a['target'] or 'none'}) after {n} messages were already logged."})
                prev = a
        return out

    def review_conflict(self, conflict: dict, decision: str, annotator="researcher", note=None):
        self.conn.execute("INSERT OR REPLACE INTO label_reviews VALUES (?,?,?,?,?,?,?,?)",
                          (conflict["id"], conflict["kind"], conflict.get("sender"), conflict.get("target"), decision,
                           json.dumps({**{k: v for k, v in conflict.items() if k != "id"}, "note": note}), annotator, time.time()))
        self.conn.commit()

    def reviews(self):
        return self.q("SELECT * FROM label_reviews ORDER BY ts")

    # --------------------------------------------------------------- messages
    def log(self, m: dict):
        self.conn.execute("""INSERT INTO messages (message_id, session_id, room_id, ts, sender, text, reply_to, mentions,
                             target, target_method, from_bank, T, p_threat, labels, risk, tier, alerts, engine_ts,
                             hostile, pair_hostile, pair_sessions, flags, user_tier, user_risk)
                             VALUES (:message_id,:session_id,:room_id,:ts,:sender,:text,:reply_to,:mentions,:target,
                             :target_method,:from_bank,:T,:p_threat,:labels,:risk,:tier,:alerts,:engine_ts,
                             :hostile,:pair_hostile,:pair_sessions,:flags,:user_tier,:user_risk)""",
                          {"hostile": None, "pair_hostile": None, "pair_sessions": None, "user_tier": None, "user_risk": None, **m,
                           "mentions": json.dumps(m.get("mentions", [])), "labels": json.dumps(m.get("labels", {})),
                           "alerts": json.dumps(m.get("alerts", [])),
                           "flags": json.dumps(m["flags"]) if m.get("flags") is not None else None})
        self.conn.commit()

    def history(self, session_id, limit=100):
        rows = self.q("""SELECT m.message_id, m.ts, m.sender, p.display_name, m.text, m.reply_to FROM messages m
                         JOIN participants p ON p.code=m.sender WHERE session_id=? ORDER BY ts DESC LIMIT ?""", (session_id, limit))
        return rows[::-1]

    def counts(self):
        c = lambda s: self.conn.execute(s).fetchone()[0]  # noqa: E731
        return {"participants": c("SELECT COUNT(*) FROM participants"), "sessions": c("SELECT COUNT(*) FROM sessions"),
                "messages": c("SELECT COUNT(*) FROM messages"),
                "pairs": c("SELECT COUNT(*) FROM (SELECT DISTINCT sender, target FROM messages)")}

    # --------------------------------------------------------------- ground truth
    def ground_truth_pairs(self):
        """Every observed (sender, target) pair with a label: manual label wins, else role-derived."""
        pairs = self.q("""SELECT sender, target, COUNT(*) AS n, MIN(ts) AS first_ts, MAX(ts) AS last_ts,
                          COUNT(DISTINCT session_id) AS sessions FROM messages GROUP BY sender, target""")
        manual = {(l["sender"], l["target"]): l for l in self.labels()}
        roles = {a["participant"]: a for a in self.q("SELECT * FROM assignments WHERE active=1")}
        for p in pairs:
            a = roles.get(p["sender"])
            p["role"] = a["role"] if a else "normal"
            role_label = int(bool(a and a["role"] in POSITIVE_ROLES and a["target"] == p["target"]))
            m = manual.get((p["sender"], p["target"]))
            p["label"] = m["label"] if m else role_label
            p["label_source"] = "manual" if m else "role_assignment"
        return pairs
