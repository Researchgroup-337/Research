"""FastAPI service: the integration point for the rest of J26-CS-337.

Run:   BAE_API_KEY=change-me BAE_DB_KEY=db-secret uvicorn bae.api:app --app-dir src --host 127.0.0.1 --port 8000

Security controls
  * every /v1 endpoint needs the X-API-Key header (constant-time comparison)
  * per-key token-bucket rate limit (default 50 req/s, burst 200) against flooding
  * request bodies validated by pydantic (types, ranges, max text length 5,000 chars)
  * database encrypted at rest when BAE_DB_KEY is set (SQLCipher)
  * binds to 127.0.0.1 by default - local processing, nothing leaves the machine
  * erasure endpoint (DELETE /v1/users/...) and retention purge for data-protection compliance
"""
from __future__ import annotations

import hmac
import os
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse

from .config import ROOT, EngineConfig
from .engine import BehavioralEngine
from .schemas import ChatEvent

API_KEY = os.environ.get("BAE_API_KEY", "dev-key-change-me")
DB_PATH = os.environ.get("BAE_DB_PATH", str(ROOT / "data" / "bae.db"))
DB_KEY = os.environ.get("BAE_DB_KEY") or None
SCORER = os.environ.get("BAE_SCORER", "auto")
RATE, BURST = float(os.environ.get("BAE_RATE", 50)), float(os.environ.get("BAE_BURST", 200))

app = FastAPI(title="Behavioral Analytics Engine", version="1.0.0",
              description="Real-time repeat-harassment detection (J26-CS-337, IT23212640)")
_engine: BehavioralEngine | None = None
_lock = threading.Lock()
_buckets: dict[str, list[float]] = {}


def engine() -> BehavioralEngine:
    global _engine
    if _engine is None:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        _engine = BehavioralEngine(cfg=EngineConfig.load(), scorer=SCORER, db_path=DB_PATH, encryption_key=DB_KEY)
    return _engine


def auth(x_api_key: str = Header(default="")) -> str:
    if not hmac.compare_digest(x_api_key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
    now = time.monotonic()
    tokens, last = _buckets.get(x_api_key, [BURST, now])
    tokens = min(BURST, tokens + (now - last) * RATE)
    if tokens < 1:
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    _buckets[x_api_key] = [tokens - 1, now]
    return x_api_key


@app.get("/health")
def health():
    e = engine()
    return {"status": "ok", "content_model": e.scorer.name if e.scorer else None,
            "encrypted_store": e.store.encrypted, "half_life_days": e.cfg.half_life_days}


@app.post("/v1/score", dependencies=[Depends(auth)])
def score(event: ChatEvent):
    with _lock:  # SQLite writer: one message at a time keeps state consistent
        return engine().process(event)


@app.post("/v1/score/batch", dependencies=[Depends(auth)])
def score_batch(events: list[ChatEvent]):
    if len(events) > 500:
        raise HTTPException(413, "max 500 events per batch")
    with _lock:
        return [engine().process(e) for e in sorted(events, key=lambda e: e.timestamp if isinstance(e.timestamp, float) else e.timestamp.timestamp())]


@app.get("/v1/users/{platform}/{user_id}/risk", dependencies=[Depends(auth)])
def user_risk(platform: str, user_id: str):
    with _lock:
        return engine().user_report(user_id, platform)


@app.get("/v1/top", dependencies=[Depends(auth)])
def top(limit: int = 25):
    with _lock:
        e = engine()
        e.store.flush()
        return [{"offender": p.offender, "target": p.target, "risk": round(p.risk, 4), "tier": p.tier,
                 "hostile_messages": p.r_hostile, "sessions_targeted": p.s_hostile, "last_ts": p.last_ts,
                 "flags": p.flags} for p in e.store.top_pairs(min(limit, 200))]


@app.get("/v1/alerts", dependencies=[Depends(auth)])
def alerts(limit: int = 50):
    with _lock:
        return engine().store.recent_alerts(min(limit, 500))


@app.get("/v1/stats", dependencies=[Depends(auth)])
def stats():
    with _lock:
        return engine().store.stats()


@app.delete("/v1/users/{platform}/{user_id}", dependencies=[Depends(auth)])
def erase_user(platform: str, user_id: str):
    with _lock:
        return {"deleted_pairs": engine().delete_user(user_id, platform)}


@app.post("/v1/admin/purge", dependencies=[Depends(auth)])
def purge():
    with _lock:
        return {"purged_pairs": engine().purge()}


@app.get("/dashboard")
def dashboard(request: Request):
    return FileResponse(Path(__file__).parent / "static" / "dashboard.html")
