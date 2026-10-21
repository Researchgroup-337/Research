"""SafeChat Lab session report (engine outputs only), label-integrity check and role validation."""
from __future__ import annotations

import importlib
import io
import os
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bae.chatapp import session_report as sr  # noqa: E402
from bae.chatapp.research_db import ResearchDB  # noqa: E402
from bae.config import EngineConfig  # noqa: E402
from bae.engine import BehavioralEngine  # noqa: E402
from bae.experiment import IdentityPseudonymizer  # noqa: E402

H = {"X-Researcher-Key": "k-test"}
BASE = 1_790_000_000.0
SECRET_TEXT = "you are a worthless idiot"


def log_scenario(rdb: ResearchDB):
    """Known scenario in room1 (owner B001), 3 sessions:
       P001 -> B002: 6 hostile messages in each of sessions 1-3 (repeat harasser)
       session 3 only: P002 troll (1 hostile each to B002, B003, B004), P003 burst (6 hostile to B006),
       P004 one explicit threat to B002, P005 and P006 friendly broadcast messages.
       B002 then has 3 distinct hostile senders within 7 days (coordinated-harassment alert)."""
    cfg = EngineConfig.load()
    eng = BehavioralEngine(cfg=cfg, scorer=None, pseudonymizer=IdentityPseudonymizer())
    rdb.add_participant("Host", True, code="B001")
    for code, name in [("B002", "Nimal"), ("B003", "Amal"), ("B004", "Kamala"), ("B006", "Sunil"), ("B007", "Ruwan")]:
        rdb.add_participant(name, True, code=code)
    for code, name in [("P001", "Harry"), ("P002", "Trolly"), ("P003", "Bursty"), ("P004", "Threatener"),
                       ("P005", "Normie"), ("P006", "Friendly")]:
        rdb.add_participant(name, False, code=code)
    rdb.conn.execute("INSERT INTO rooms VALUES ('room1', 'Lab', 'B001')")
    plan = {1: [("P001", "B002", 0.8, 0.0)] * 6, 2: [("P001", "B002", 0.8, 0.0)] * 6,
            3: [("P001", "B002", 0.8, 0.0)] * 6 + [("P002", t, 0.7, 0.0) for t in ("B002", "B003", "B004")]
               + [("P003", "B006", 0.8, 0.0)] * 6 + [("P004", "B002", 0.95, 0.95)]
               + [("P005", None, 0.02, 0.0)] * 2 + [("P006", None, 0.02, 0.0)] * 2}
    n = 0
    for k, msgs in plan.items():
        sid, start = f"room1_s{k:03d}", BASE + k * 3600
        rdb.conn.execute("INSERT INTO sessions VALUES (?,?,?,?)", (sid, "room1", start, start + 600))
        for i, (snd, tgt, T, thr) in enumerate(msgs):
            n += 1
            text = (f"@{tgt} " if tgt else "") + (SECRET_TEXT if T >= 0.5 else "great stream")
            ets = BASE + k * 2 * 86400 + 10 * i
            rec = eng.process({"message_id": f"m{n}", "platform": "safechat", "session_id": sid, "channel_owner": "B001",
                               "sender_id": snd, "text": text, "timestamp": ets, "mentions": [tgt] if tgt else [],
                               "content_scores": {"T": T, "threat": thr}})
            rdb.log({"message_id": f"m{n}", "session_id": sid, "room_id": "room1", "ts": start + 10 * i, "sender": snd,
                     "text": text, "reply_to": None, "mentions": [tgt] if tgt else [], "target": rec["target"],
                     "target_method": rec["target_method"], "from_bank": None, "T": rec["message"]["T"],
                     "p_threat": rec["message"]["labels"].get("threat", 0.0), "labels": rec["message"]["labels"],
                     "risk": rec["pair"]["risk"], "tier": rec["pair"]["tier"], "alerts": rec["alerts"], "engine_ts": ets,
                     "hostile": int(rec["message"]["hostile"]), "pair_hostile": rec["pair"]["hostile_messages"],
                     "pair_sessions": rec["pair"]["sessions_targeted"], "flags": rec["flags"],
                     "user_tier": rec["user_max_tier"], "user_risk": rec["user_max_risk"]})
    rdb.conn.commit()
    rdb.assign("P001", "harasser", "B002")
    rdb.assign("P002", "troll")
    rdb.assign("P003", "burst", "B006")


@pytest.fixture()
def rdb(tmp_path):
    db = ResearchDB(str(tmp_path / "r.db"))
    log_scenario(db)
    return db


def test_categories_and_kpis(rdb):
    rep = sr.build_report(rdb, "room1_s003", EngineConfig.load(), content_model="test")
    c = rep["categories"]
    assert c["repeat_harasser"] == ["P001"]
    assert c["troll"] == ["P002"]
    assert c["burst_attacker"] == ["P003"]
    assert c["explicit_threat"] == ["P004"]
    assert c["normal"] == ["P005", "P006"]
    assert c["coordinated_target"] == ["B002"]
    k = rep["kpi"]
    assert (k["messages"], k["active_users"], k["hostile"], k["hostile_pct"]) == (20, 6, 16, 80.0)
    assert (k["repeat_harassers"], k["trolls"], k["burst_attackers"], k["explicit_threats"], k["users_targeted"]) == (1, 1, 1, 1, 4)
    html = sr.render_html(rep)
    for kid, v in [("messages", "20"), ("active_users", "6"), ("repeat_harassers", "1"), ("trolls", "1"),
                   ("burst_attackers", "1"), ("explicit_threats", "1"), ("users_targeted", "4")]:
        assert f'id="kpi-{kid}">{v}<' in html
    assert "Harry (P001)" in html and html.count("data:image/png;base64,") == 6


def test_earlier_session_and_escalation(rdb):
    rep1 = sr.build_report(rdb, "room1_s001", EngineConfig.load())
    assert rep1["categories"]["repeat_harasser"] == []          # one session only: not yet repeat
    rep3 = sr.build_report(rdb, "room1_s003", EngineConfig.load())
    assert rep3["prev_session"] == "room1_s002"
    harry = next(o for o in rep3["offenders"] if o["code"] == "P001")
    assert harry["prev_tier"] is not None and harry["change"] != "new"


def test_mask_text_and_codes_only(rdb):
    cfg = EngineConfig.load()
    masked = sr.render_html(sr.build_report(rdb, "room1_s003", cfg, mask_text=True))
    assert SECRET_TEXT not in masked and sr.HIDDEN in masked
    shown = sr.render_html(sr.build_report(rdb, "room1_s003", cfg, mask_text=False))
    assert SECRET_TEXT in shown
    codes = sr.build_report(rdb, "room1_s003", cfg, codes_only=True)
    assert "Harry" not in sr.render_html(codes)
    z = zipfile.ZipFile(io.BytesIO(sr.render_csv_zip(codes)))
    assert {"users.csv", "pairs.csv", "alerts.csv", "categories.csv"} <= set(z.namelist())
    assert b"Harry" not in z.read("users.csv") and SECRET_TEXT.encode() not in z.read("pairs.csv")
    assert sr.render_pdf(codes)[:4] == b"%PDF"


def test_without_truth_never_reads_roles_or_labels(rdb, monkeypatch):
    seen = []
    real_q, real_one = rdb.q, rdb.one
    monkeypatch.setattr(rdb, "q", lambda sql, args=(): (seen.append(sql), real_q(sql, args))[1])
    monkeypatch.setattr(rdb, "one", lambda sql, args=(): (seen.append(sql), real_one(sql, args))[1])

    def forbidden(*a, **k):
        raise AssertionError("ground truth read while include_truth=False")
    for name in ("ground_truth_pairs", "labels", "role_of", "assignments"):
        monkeypatch.setattr(rdb, name, forbidden)
    rep = sr.build_report(rdb, "room1_s003", EngineConfig.load(), include_truth=False)
    assert rep["truth"] is None
    assert seen and not any("assignments" in s or "pair_labels" in s for s in seen)
    assert "Researcher appendix" not in sr.render_html(rep)


def test_with_truth_adds_appendix(rdb):
    rep = sr.build_report(rdb, "room1_s003", EngineConfig.load(), include_truth=True)
    assert rep["truth"]["role_by_category"]["harasser"] == {"repeat_harasser": 1}
    assert rep["truth"]["confusion"].get("11") == 1                    # P001 -> B002 label 1, engine flagged
    assert "Researcher appendix: engine vs ground truth" in sr.render_html(rep)


# ---------------------------------------------------------------- server endpoints
@pytest.fixture()
def app(tmp_path, monkeypatch):
    for k, v in {"SAFECHAT_DIR": str(tmp_path), "BAE_RESEARCHER_KEY": "k-test", "SAFECHAT_DB": str(tmp_path / "r.db"),
                 "SAFECHAT_ENGINE_DB": str(tmp_path / "e.db"), "SAFECHAT_DAYS_PER_SESSION": "2", "BAE_SCORER": "tfidf"}.items():
        monkeypatch.setenv(k, v)
    import bae.chatapp.server as srv
    importlib.reload(srv)
    from fastapi.testclient import TestClient
    return TestClient(srv.app), srv


def test_report_endpoint_requires_key_and_formats(app):
    c, srv = app
    log_scenario(srv.rdb)
    url = "/api/admin/sessions/room1_s003/report"
    assert c.get(url).status_code == 401
    r = c.get(url, headers=H)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    assert 'id="kpi-messages">20<' in r.text and SECRET_TEXT not in r.text      # masked by default
    assert SECRET_TEXT in c.get(url + "?mask_text=false", headers=H).text
    assert c.get(url + "?format=pdf", headers=H).content[:4] == b"%PDF"
    assert "categories.csv" in zipfile.ZipFile(io.BytesIO(c.get(url + "?format=csv", headers=H).content)).namelist()
    assert c.get(url + "?format=doc", headers=H).status_code == 422
    assert c.get("/api/admin/sessions/nope/report", headers=H).status_code == 404


def test_end_session_auto_saves_report(app, tmp_path):
    c, srv = app
    host = c.post("/api/admin/participants", json={"display_name": "Host", "is_bot": True}, headers=H).json()
    rid = c.post("/api/admin/rooms", json={"name": "Lab", "owner": host["code"]}, headers=H).json()["room_id"]
    sid = c.post(f"/api/admin/rooms/{rid}/start", headers=H).json()["session_id"]
    r = c.post(f"/api/admin/rooms/{rid}/end?save_report=true", headers=H).json()
    assert r["ended"] == sid and (tmp_path / "reports" / f"{sid}.html").exists()
    sid2 = c.post(f"/api/admin/rooms/{rid}/start", headers=H).json()["session_id"]
    assert "report" not in c.post(f"/api/admin/rooms/{rid}/end", headers=H).json()      # off by default
    assert not (tmp_path / "reports" / f"{sid2}.html").exists()


def test_save_role_refuses_missing_target(app):
    c, _ = app
    a = c.post("/api/admin/participants", json={"display_name": "Alice"}, headers=H).json()
    b = c.post("/api/admin/participants", json={"display_name": "Bob", "is_bot": True}, headers=H).json()
    for role in ("harasser", "subtle_harasser", "escalator", "burst", "banter"):
        r = c.post("/api/admin/assignments", json={"participant": a["code"], "role": role, "target": None}, headers=H)
        assert r.status_code == 422 and "needs a target" in r.json()["detail"]
    assert c.post("/api/admin/assignments", json={"participant": a["code"], "role": "normal"}, headers=H).status_code == 200
    assert c.post("/api/admin/assignments", json={"participant": a["code"], "role": "harasser", "target": b["code"]},
                  headers=H).status_code == 200


def _msg(srv, mid, snd, tgt, ts):
    srv.rdb.log({"message_id": mid, "session_id": "room1_s001", "room_id": "room1", "ts": ts, "sender": snd, "text": "hi",
                 "reply_to": None, "mentions": [], "target": tgt, "target_method": "mention", "from_bank": None, "T": 0.1,
                 "p_threat": 0.0, "labels": {}, "risk": 0.0, "tier": "Low", "alerts": [], "engine_ts": ts})


def test_label_check_conflicts_and_resolution(app):
    import time
    c, srv = app
    A, B, C, D = (c.post("/api/admin/participants", json={"display_name": n}, headers=H).json()["code"]
                  for n in ("Ann", "Ben", "Cal", "Dee"))
    assert c.get("/api/admin/label-check").status_code == 401
    # (a) positive role but manual 0
    c.post("/api/admin/assignments", json={"participant": A, "role": "harasser", "target": B}, headers=H)
    c.post("/api/admin/labels", json={"sender": A, "target": B, "label": 0}, headers=H)
    # (b) manual 1 on a normal sender
    _msg(srv, "x1", C, B, time.time() - 50)
    c.post("/api/admin/labels", json={"sender": C, "target": B, "label": 1}, headers=H)
    # (c) role changed after messages were logged
    _msg(srv, "x2", D, B, time.time() - 40)
    c.post("/api/admin/assignments", json={"participant": D, "role": "troll"}, headers=H)
    lc = c.get("/api/admin/label-check", headers=H).json()
    kinds = {x["kind"]: x for x in lc["conflicts"]}
    assert set(kinds) == {"positive_role_manual_0", "manual_1_negative_role", "role_changed_after_messages"}
    assert kinds["role_changed_after_messages"]["old_role"] == "normal" and kinds["role_changed_after_messages"]["new_role"] == "troll"
    assert not kinds["role_changed_after_messages"]["can_use_role"]
    # nothing has been changed silently
    assert {(l["sender"], l["target"]) for l in srv.rdb.labels()} == {(A, B), (C, B)}

    res = lambda cid, a: c.post("/api/admin/label-check/resolve", json={"conflict_id": cid, "action": a}, headers=H)  # noqa: E731
    assert res(kinds["positive_role_manual_0"]["id"], "use_role").json()["decision"] == "used_role_label"
    assert (A, B) not in {(l["sender"], l["target"]) for l in srv.rdb.labels()}
    assert res(kinds["manual_1_negative_role"]["id"], "keep_manual").json()["decision"] == "kept_manual_label"
    assert res(kinds["role_changed_after_messages"]["id"], "use_role").status_code == 422
    assert res(kinds["role_changed_after_messages"]["id"], "keep_manual").json()["decision"] == "reviewed"
    lc = c.get("/api/admin/label-check", headers=H).json()
    assert lc["conflicts"] == [] and len(lc["reviewed"]) == 3
    z = zipfile.ZipFile(io.BytesIO(c.get("/api/admin/export", headers=H).content))
    pairs = z.read("pairs.csv").decode()
    assert "label_review" in pairs.splitlines()[0] and "manual_1_negative_role:kept_manual_label" in pairs
    assert "label_reviews.csv" in z.namelist()


# ---------------------------------------------------------------- overall (user-level) status
def test_report_shows_overall_tier(rdb):
    rep = sr.build_report(rdb, "room1_s003", EngineConfig.load())
    harry = next(o for o in rep["offenders"] if o["code"] == "P001")
    assert harry["overall_tier"] in ("High", "Critical")
    assert "Overall tier" in sr.render_html(rep)
    assert "overall_tier" in zipfile.ZipFile(io.BytesIO(sr.render_csv_zip(rep))).read("users.csv").decode().splitlines()[0]


def test_old_database_without_new_columns_still_opens(tmp_path):
    import sqlite3
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)  # messages table exactly as before either change set
    con.executescript("""CREATE TABLE messages (message_id TEXT PRIMARY KEY, session_id TEXT NOT NULL, room_id TEXT NOT NULL,
        ts REAL NOT NULL, sender TEXT NOT NULL, text TEXT NOT NULL, reply_to TEXT, mentions TEXT NOT NULL DEFAULT '[]',
        target TEXT, target_method TEXT, from_bank TEXT, T REAL, p_threat REAL, labels TEXT, risk REAL, tier TEXT,
        alerts TEXT, engine_ts REAL);
        INSERT INTO messages (message_id, session_id, room_id, ts, sender, text, T, tier) VALUES ('old1','s1','room1',1,'P001','hi',0.1,'Low');""")
    con.commit()
    con.close()
    db = ResearchDB(str(path))
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info(messages)")}
    assert {"user_tier", "user_risk", "hostile", "pair_hostile", "pair_sessions", "flags"} <= cols
    assert db.one("SELECT user_tier, hostile FROM messages WHERE message_id='old1'") == {"user_tier": None, "hostile": None}
    db.log({"message_id": "new1", "session_id": "s1", "room_id": "room1", "ts": 2, "sender": "P001", "text": "x",
            "reply_to": None, "mentions": [], "target": "B002", "target_method": "mention", "from_bank": None, "T": 0.9,
            "p_threat": 0.0, "labels": {}, "risk": 0.7, "tier": "High", "alerts": [], "engine_ts": 2,
            "user_tier": "High", "user_risk": 0.7})
    assert db.one("SELECT user_tier, user_risk FROM messages WHERE message_id='new1'") == {"user_tier": "High", "user_risk": 0.7}
    ResearchDB(str(path))  # opening again is a no-op


def test_server_logs_user_tier_and_pairs_overall(app):
    c, srv = app
    host = c.post("/api/admin/participants", json={"display_name": "Host", "is_bot": True}, headers=H).json()
    a = c.post("/api/admin/participants", json={"display_name": "Alice"}, headers=H).json()
    b = c.post("/api/admin/participants", json={"display_name": "Bob", "is_bot": True}, headers=H).json()
    rid = c.post("/api/admin/rooms", json={"name": "Lab", "owner": host["code"]}, headers=H).json()["room_id"]
    c.post("/api/join", json={"token": a["token"], "consent": True})
    c.post(f"/api/admin/rooms/{rid}/start", headers=H)
    with c.websocket_connect(f"/ws/{rid}?token={a['token']}") as ws:
        ws.receive_json()
        ws.receive_json()
        for t in ("@Bob you are pathetic", "nice stream everyone"):
            ws.send_json({"type": "msg", "text": t})
            while ws.receive_json()["type"] != "msg":
                pass
    rows = srv.rdb.q("SELECT user_tier, user_risk, hostile, pair_sessions FROM messages")
    assert len(rows) == 2 and all(r["user_tier"] in ("Low", "Medium", "High", "Critical") and r["user_risk"] is not None for r in rows)
    feed = c.get("/api/admin/feed", headers=H).json()
    assert all("user_tier" in f and "user_risk" in f for f in feed)
    pairs = c.get("/api/admin/pairs", headers=H).json()
    mine = [p for p in pairs if p["sender"] == a["code"]]
    assert len(mine) == 2
    top = max(mine, key=lambda p: p["risk_now"])
    for p in mine:   # the same overall status on every pair of the sender = their highest pair
        assert p["sender_overall_tier"] == top["tier_now"] and p["sender_overall_risk"] == top["risk_now"]
        assert p["sender_targets_high"] == sum(q["tier_now"] in ("High", "Critical") for q in mine)
    z = zipfile.ZipFile(io.BytesIO(c.get("/api/admin/export", headers=H).content))
    assert "user_tier,user_risk" in z.read("messages.csv").decode().splitlines()[0]
