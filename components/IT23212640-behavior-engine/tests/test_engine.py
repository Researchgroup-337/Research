"""Unit + security tests.  Run:  pytest -q"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bae.config import EngineConfig  # noqa: E402
from bae.engine import BehavioralEngine  # noqa: E402
from bae.privacy import Pseudonymizer  # noqa: E402
from bae.risk import escalation_slope, tier_for  # noqa: E402
from bae.store import BehaviorStore  # noqa: E402
from bae.target_resolver import TargetResolver  # noqa: E402
from bae.text_norm import normalize, perturb  # noqa: E402

DAY = 86400.0
T0 = 1_782_844_200.0


def cfg():
    return EngineConfig.load()


def eng(**kw):
    return BehavioralEngine(cfg=cfg(), scorer=None, pseudonymizer=Pseudonymizer(b"k" * 32), **kw)


def harass(e, sender, target, sessions, per_session=3, T=0.8, start=T0, gap_days=2.0):
    rec = None
    for s in range(sessions):
        for m in range(per_session):
            rec = e.process_scored(f"{sender}-{s}-{m}", sender, target, f"sess{s}", start + s * gap_days * DAY + m * 60, T)
    return rec


# ---------------------------------------------------------------- text normalisation
@pytest.mark.parametrize("raw,expected", [
    ("I will k1ll you", "i will kill you"),
    ("I will k i l l you", "i will kill you"),
    ("I wіll kіll yоu", "i will kill you"),          # Cyrillic homoglyphs
    ("i will k​i​l​l you", "i will kill you"),
    ("see @Kasun at http://x.y", "see xxuser at xxurl"),
    ("year 2024 and 100 people", "year 2024 and 100 people"),  # plain numbers untouched
])
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_perturb_then_normalize_roundtrip():
    s = "i am going to find you and hurt you"
    for k in ["leet", "spaced", "zero_width", "homoglyph"]:
        assert normalize(perturb(s, k, rate=1.0)) == s


# ---------------------------------------------------------------- target resolution
def test_resolver_priority():
    r = TargetResolver()
    assert r.resolve("hey", "a", "s1", reply_to="b")[:2] == ("b", "reply")
    assert r.resolve("@cat you suck", "a", "s1")[:2] == ("cat", "mention")
    assert r.resolve("@a talking to myself", "a", "s1", channel_owner="own")[:2] == ("own", "broadcast")
    r.observe("s1", "kasun")
    assert r.resolve("kasun you are trash", "a", "s1")[:2] == ("kasun", "name")
    assert r.resolve("great stream", "a", "s1", channel_owner="streamer")[:2] == ("streamer", "broadcast")


# ---------------------------------------------------------------- privacy
def test_pseudonymisation_is_keyed_and_stable():
    p1, p2 = Pseudonymizer(b"a" * 32), Pseudonymizer(b"b" * 32)
    assert p1("Alice", "yt") == p1("alice ", "YT")
    assert p1("Alice", "yt") != p2("Alice", "yt")      # different key -> unlinkable
    assert "alice" not in p1("Alice", "yt")


def test_no_message_text_stored(tmp_path):
    e = BehavioralEngine(cfg=cfg(), scorer=None, db_path=str(tmp_path / "x.db"))
    e.process({"message_id": "1", "session_id": "s", "sender_id": "bob", "text": "SECRET-PHRASE @amy",
               "timestamp": T0, "content_scores": {"T": 0.9}})
    e.store.close()
    blob = (tmp_path / "x.db").read_bytes()
    assert b"SECRET-PHRASE" not in blob and b"bob" not in blob and b"amy" not in blob


def test_encrypted_store_unreadable_without_key(tmp_path):
    pytest.importorskip("sqlcipher3")
    path = str(tmp_path / "enc.db")
    s = BehaviorStore(path, encryption_key="correct horse")
    s.add_alert(T0, "x", "o", "t", "Low", "High", 0.9, {})
    s.close()
    import sqlite3
    with pytest.raises(sqlite3.DatabaseError):
        sqlite3.connect(path).execute("SELECT * FROM alerts").fetchall()
    s2 = BehaviorStore(path, encryption_key="correct horse")
    assert len(s2.recent_alerts()) == 1


def test_right_to_erasure():
    e = eng()
    harass(e, "bully", "victim", 4)
    e.store.flush()
    assert e.store.stats()["pairs"] == 1
    assert e.delete_user("bully", "sim") == 1
    assert e.store.stats()["pairs"] == 0


def test_retention_purge():
    e = eng()
    harass(e, "old", "v", 2, start=T0)
    harass(e, "new", "v", 2, start=T0 + 200 * DAY)
    assert e.purge(now=T0 + 205 * DAY) == 1


# ---------------------------------------------------------------- behaviour
def test_persistent_harasser_escalates_to_high():
    e = eng()
    rec = harass(e, "bully", "victim", sessions=6, per_session=3, T=0.75)
    assert rec["pair"]["sessions_targeted"] == 6
    assert rec["pair"]["tier"] in ("High", "Critical")


def test_single_toxic_message_stays_low():
    e = eng()
    rec = e.process_scored("m1", "rude", "x", "s1", T0, 0.8)
    assert rec["pair"]["tier"] == "Low"


def test_persistence_beats_one_session_burst_after_time():
    """A burst of 15 hostile messages in one session should rank below a user who returns
    across 6 sessions once some time has passed."""
    e = eng()
    harass(e, "burst", "v1", sessions=1, per_session=15, start=T0)
    harass(e, "persist", "v2", sessions=6, per_session=2, start=T0)
    snap = {r["offender"]: r["risk"] for r in e.snapshot(now=T0 + 14 * DAY)}
    hb, hp = e.pseudo("burst", "sim"), e.pseudo("persist", "sim")
    assert snap[hp] > snap[hb]


def test_decay_returns_reformed_user_to_low():
    e = eng()
    harass(e, "reformed", "v", sessions=6, per_session=3)
    end = T0 + 6 * 2 * DAY
    p_now = {r["offender"]: r["risk"] for r in e.snapshot(now=end)}
    p_later = {r["offender"]: r["risk"] for r in e.snapshot(now=end + 120 * DAY)}
    h = e.pseudo("reformed", "sim")
    assert p_later[h] < p_now[h]
    assert tier_for(p_later[h], e.cfg) == "Low"


def test_clean_messages_do_not_accumulate():
    e = eng()
    rec = harass(e, "fan", "streamer", sessions=20, per_session=5, T=0.05)
    assert rec["pair"]["hostile_messages"] == 0 and rec["pair"]["tier"] == "Low"


def test_explicit_threat_override_and_alert():
    e = eng()
    rec = e.process({"message_id": "1", "session_id": "s", "sender_id": "x", "text": "@y I will find you",
                     "timestamp": T0, "content_scores": {"T": 0.95, "threat": 0.97}})
    assert rec["pair"]["tier"] == "High"
    assert rec["alerts"] and rec["alerts"][0]["kind"] == "tier_escalation"
    rec2 = e.process({"message_id": "2", "session_id": "s", "sender_id": "x", "text": "@y ok",
                      "timestamp": T0 + 60, "content_scores": {"T": 0.1, "threat": 0.0}})
    assert rec2["pair"]["tier"] == "High" and not rec2["alerts"]   # sticky, no duplicate alert


def test_coordinated_harassment_alert():
    e = eng()
    alerts = []
    for i, u in enumerate(["a1", "a2", "a3", "a4"]):
        alerts += e.process_scored(f"m{i}", u, "victim", "s1", T0 + i * 60, 0.9)["alerts"]
    assert any(a["kind"] == "coordinated_harassment" for a in alerts)


def test_peer_signals_raise_effective_T():
    e = eng()
    rec = e.process({"message_id": "1", "session_id": "s", "sender_id": "x", "text": "@y click this",
                     "timestamp": T0, "content_scores": {"T": 0.1},
                     "peer_signals": {"url_threat": "malicious", "se_score": 0.2}})
    assert rec["message"]["effective_T"] >= 0.75 and rec["message"]["hostile"]
    assert "url_malicious" in rec["flags"]


def test_escalation_slope():
    assert escalation_slope([0.1, 0.3, 0.5, 0.7]) > 0.15
    assert escalation_slope([0.9, 0.5, 0.1]) == 0.0


def test_monotonic_in_hostility():
    """Security property: more hostile messages never lower risk."""
    e = eng()
    risks = []
    for i in range(12):
        risks.append(e.process_scored(f"m{i}", "b", "v", f"s{i // 3}", T0 + i * 3600, 0.8)["pair"]["risk"])
    assert all(b >= a - 1e-9 for a, b in zip(risks, risks[1:]))
