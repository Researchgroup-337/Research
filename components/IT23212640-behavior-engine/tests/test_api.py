"""API tests: authentication, validation, scoring, erasure."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture()
def client(tmp_path):
    os.environ["BAE_API_KEY"] = "test-key"
    os.environ["BAE_DB_PATH"] = str(tmp_path / "api.db")
    os.environ["BAE_RATE"] = "1000"
    os.environ.pop("BAE_DB_KEY", None)
    import bae.api as api
    importlib.reload(api)
    from fastapi.testclient import TestClient
    return TestClient(api.app)


EV = {"message_id": "m1", "platform": "youtube", "session_id": "s1", "channel_owner": "streamer",
      "sender_id": "troll", "text": "@kasun I know where you live, you are worthless", "timestamp": 1782844200.0}


def test_auth_required(client):
    assert client.post("/v1/score", json=EV).status_code == 401
    assert client.post("/v1/score", json=EV, headers={"X-API-Key": "wrong"}).status_code == 401


def test_score_and_contract(client):
    r = client.post("/v1/score", json=EV, headers={"X-API-Key": "test-key"})
    assert r.status_code == 200
    body = r.json()
    for k in ("offender", "target", "message", "pair", "alerts", "user_max_tier", "latency_ms"):
        assert k in body
    assert body["target_method"] == "mention"
    assert body["pair"]["tier"] in ("Low", "Medium", "High", "Critical")
    assert "troll" not in body["offender"]


def test_validation_rejects_bad_input(client):
    bad = dict(EV, text="x" * 6000)
    assert client.post("/v1/score", json=bad, headers={"X-API-Key": "test-key"}).status_code == 422
    bad2 = dict(EV, peer_signals={"hate_score": 7})
    assert client.post("/v1/score", json=bad2, headers={"X-API-Key": "test-key"}).status_code == 422


def test_erasure_endpoint(client):
    h = {"X-API-Key": "test-key"}
    client.post("/v1/score", json=EV, headers=h)
    r = client.delete("/v1/users/youtube/troll", headers=h)
    assert r.status_code == 200 and r.json()["deleted_pairs"] >= 1


def test_health_and_dashboard(client):
    assert client.get("/health").json()["status"] == "ok"
    assert "Moderator Risk Console" in client.get("/dashboard").text
