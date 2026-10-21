"""SafeChat Lab tests: consent gate, WebSocket chat, target resolution, research log, export."""
from __future__ import annotations

import importlib
import io
import os
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
H = {"X-Researcher-Key": "k-test"}


@pytest.fixture()
def app(tmp_path):
    os.environ.update({"SAFECHAT_DIR": str(tmp_path), "BAE_RESEARCHER_KEY": "k-test",
                       "SAFECHAT_DB": str(tmp_path / "r.db"), "SAFECHAT_ENGINE_DB": str(tmp_path / "e.db"),
                       "SAFECHAT_DAYS_PER_SESSION": "2"})
    import bae.chatapp.server as srv
    importlib.reload(srv)
    from fastapi.testclient import TestClient
    return TestClient(srv.app)


def setup_world(c):
    host = c.post("/api/admin/participants", json={"display_name": "Host", "is_bot": True}, headers=H).json()
    a = c.post("/api/admin/participants", json={"display_name": "Alice"}, headers=H).json()
    b = c.post("/api/admin/participants", json={"display_name": "Bob", "is_bot": True}, headers=H).json()
    room = c.post("/api/admin/rooms", json={"name": "Lab", "owner": host["code"]}, headers=H).json()
    c.post("/api/admin/assignments", json={"participant": a["code"], "role": "harasser", "target": b["code"]}, headers=H)
    return host, a, b, room


def test_researcher_auth(app):
    assert app.get("/api/admin/stats").status_code == 401
    assert app.get("/api/admin/stats", headers=H).status_code == 200


def test_consent_required(app):
    _, a, _, room = setup_world(app)
    assert app.post("/api/join", json={"token": a["token"], "consent": False}).status_code == 403
    r = app.post("/api/join", json={"token": a["token"], "consent": True}).json()
    assert r["role"] == "harasser" and r["target_name"] == "Bob" and r["bank"]


def test_chat_logs_target_and_exports(app):
    host, a, b, room = setup_world(app)
    app.post("/api/join", json={"token": a["token"], "consent": True})
    rid = room["room_id"]
    with app.websocket_connect(f"/ws/{rid}?token={a['token']}") as ws:
        assert ws.receive_json()["type"] == "hello"
        ws.receive_json()  # members
        ws.send_json({"type": "msg", "text": "too early"})
        assert ws.receive_json()["type"] == "error"  # no live session yet
        app.post(f"/api/admin/rooms/{rid}/start", headers=H)
        ws.receive_json()  # session live
        ws.send_json({"type": "msg", "text": "@Bob you are pathetic"})
        m = ws.receive_json()
        assert m["type"] == "msg" and m["sender"] == a["code"]
        ws.send_json({"type": "msg", "text": "nice stream everyone"})
        ws.receive_json()
    feed = app.get("/api/admin/feed", headers=H).json()
    by_text = {f["text"]: f for f in feed}
    assert by_text["@Bob you are pathetic"]["target"] == b["code"]
    assert by_text["@Bob you are pathetic"]["target_method"] == "mention"
    assert by_text["nice stream everyone"]["target"] == host["code"]          # broadcast -> room owner
    pairs = {(p["sender"], p["target"]): p for p in app.get("/api/admin/pairs", headers=H).json()}
    assert pairs[(a["code"], b["code"])]["label"] == 1
    assert pairs[(a["code"], host["code"])]["label"] == 0
    z = zipfile.ZipFile(io.BytesIO(app.get("/api/admin/export", headers=H).content))
    assert {"messages.csv", "pairs.csv", "participants.csv", "sessions.csv", "README.txt"} <= set(z.namelist())
    assert b"Alice" not in z.read("participants.csv")                          # display names never exported
    z2 = zipfile.ZipFile(io.BytesIO(app.get("/api/admin/export?include_text=false", headers=H).content))
    assert b"pathetic" not in z2.read("messages.csv")


def test_ws_rejects_bad_token(app):
    _, _, _, room = setup_world(app)
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with app.websocket_connect(f"/ws/{room['room_id']}?token=nope") as ws:
            ws.receive_json()


def test_lab_clock_spaces_sessions(app):
    _, a, _, room = setup_world(app)
    import bae.chatapp.server as srv
    rid = room["room_id"]
    s1 = app.post(f"/api/admin/rooms/{rid}/start", headers=H).json()["session_id"]
    app.post(f"/api/admin/rooms/{rid}/end", headers=H)
    s2 = app.post(f"/api/admin/rooms/{rid}/start", headers=H).json()["session_id"]
    import time
    now = time.time()
    gap = srv.rdb.lab_clock(s2, now, 2) - srv.rdb.lab_clock(s1, now, 2)
    assert abs(gap - 2 * 86400) < 5
