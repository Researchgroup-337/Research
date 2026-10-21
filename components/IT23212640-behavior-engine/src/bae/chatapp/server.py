"""SafeChat Lab: test live-chat application with session tracking.

    BAE_RESEARCHER_KEY=lab-key uvicorn bae.chatapp.server:app --app-dir src --host 0.0.0.0 --port 8100

  /                participants join with their study token, give consent and chat (WebSocket)
  /researcher      researcher console: participants, roles, rooms, start/end sessions,
                   live risk tiers, pair labelling, dataset export

Every message goes through the real Behavioral Analytics Engine (target resolution,
content model, cross-session history, TPI, tiers) and is logged, with the engine's output,
to a research database. Exports use the same schema as the synthetic corpus, so
scripts/evaluate_collected.py can evaluate the engine on human-generated sessions.
Participants never see risk scores (blind study); only the researcher console does.
"""
from __future__ import annotations

import asyncio
import csv
import hmac
import io
import json
import os
import random
import re
import time
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from ..config import ROOT, EngineConfig
from ..engine import BehavioralEngine
from ..risk import pair_features, risk_probability, tier_for
from . import session_report
from .research_db import BANK_ROLES, ROLES, ResearchDB

PLATFORM = "safechat"
KEY = os.environ.get("BAE_RESEARCHER_KEY", "lab-key-change-me")
# Lab clock: each session stands for a live stream this many days after the previous one.
DAYS_PER_SESSION = float(os.environ.get("SAFECHAT_DAYS_PER_SESSION", "2"))
DATA_DIR = Path(os.environ.get("SAFECHAT_DIR", ROOT / "data" / "safechat"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="SafeChat Lab", version="1.0.0")
rdb = ResearchDB(os.environ.get("SAFECHAT_DB", str(DATA_DIR / "research.db")))
engine = BehavioralEngine(cfg=EngineConfig.load(), scorer=os.environ.get("BAE_SCORER", "auto"),
                          db_path=os.environ.get("SAFECHAT_ENGINE_DB", str(DATA_DIR / "engine.db")),
                          encryption_key=os.environ.get("BAE_DB_KEY") or None)
_bank_path = ROOT / "data" / "message_bank.json"
BANK = json.loads(_bank_path.read_text(encoding="utf-8"))["buckets"] if _bank_path.exists() else {}
_hash2code: dict[str, str] = {}
_lock = asyncio.Lock()


def _code_of(h: str) -> str:
    if h not in _hash2code:
        for p in rdb.participants():
            _hash2code[engine.pseudo(p["code"], PLATFORM)] = p["code"]
    return _hash2code.get(h, h)


def researcher(x_researcher_key: str = Header(default="")):
    if not hmac.compare_digest(x_researcher_key.encode(), KEY.encode()):
        raise HTTPException(401, "invalid researcher key")


# ------------------------------------------------------------------ live rooms
class Hub:
    def __init__(self):
        self.rooms: dict[str, dict[str, WebSocket]] = {}

    def members(self, rid):
        names = {p["code"]: p["display_name"] for p in rdb.participants()}
        return [{"code": c, "name": names.get(c, c)} for c in self.rooms.get(rid, {})]

    async def send_all(self, rid, payload):
        dead = []
        for code, ws in list(self.rooms.get(rid, {}).items()):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(code)
        for c in dead:
            self.rooms.get(rid, {}).pop(c, None)


hub = Hub()


def resolve_mentions(text: str, rid: str, sender: str) -> list[str]:
    """Map @DisplayName (or a bare display name) in the text to study codes of people in the room."""
    by_name = {}
    for p in rdb.participants():
        by_name[p["display_name"].lower()] = p["code"]
        by_name[p["code"].lower()] = p["code"]
    out = []
    for m in re.findall(r"@([\w.\-]{2,32})", text):
        c = by_name.get(m.lower())
        if c and c != sender and c not in out:
            out.append(c)
    if not out:
        present = {m["name"].lower(): m["code"] for m in hub.members(rid)}
        for w in re.findall(r"[\w.\-]{3,32}", text):
            c = present.get(w.lower())
            if c and c != sender:
                out.append(c)
                break
    return out


async def handle_message(rid: str, sender: dict, data: dict):
    sid = rdb.active_session(rid)
    if not sid:
        return {"type": "error", "detail": "No live session in this room right now."}
    text = str(data.get("text", "")).strip()[:500]
    if not text:
        return None
    reply_to = data.get("reply_to") or None
    if reply_to == sender["code"]:
        reply_to = None
    mentions = resolve_mentions(text, rid, sender["code"])
    room = rdb.room(rid)
    ts = time.time()
    ets = rdb.lab_clock(sid, ts, DAYS_PER_SESSION)
    mid = f"m{uuid.uuid4().hex[:12]}"
    async with _lock:
        rec = engine.process({"message_id": mid, "platform": PLATFORM, "session_id": sid, "channel_owner": room["owner"],
                              "sender_id": sender["code"], "text": text, "timestamp": ets,
                              "reply_to": reply_to, "mentions": mentions})
    target = _code_of(rec["target"])
    rdb.log({"message_id": mid, "session_id": sid, "room_id": rid, "ts": ts, "sender": sender["code"], "text": text,
             "reply_to": reply_to, "mentions": mentions, "target": target, "target_method": rec["target_method"],
             "from_bank": data.get("from_bank"), "T": rec["message"]["T"],
             "p_threat": rec["message"]["labels"].get("threat", 0.0), "labels": rec["message"]["labels"],
             "risk": rec["pair"]["risk"], "tier": rec["pair"]["tier"], "alerts": rec["alerts"], "engine_ts": ets,
             "hostile": int(rec["message"]["hostile"]), "pair_hostile": rec["pair"]["hostile_messages"],
             "pair_sessions": rec["pair"]["sessions_targeted"], "flags": rec["flags"],
             "user_tier": rec["user_max_tier"], "user_risk": rec["user_max_risk"]})
    await hub.send_all(rid, {"type": "msg", "message_id": mid, "ts": ts, "sender": sender["code"],
                             "name": sender["display_name"], "text": text, "reply_to": reply_to})
    return None


@app.websocket("/ws/{rid}")
async def ws_room(ws: WebSocket, rid: str, token: str = ""):
    p = rdb.by_token(token)
    if not p or not rdb.room(rid) or not p["consent_ts"]:
        await ws.close(code=4401)
        return
    await ws.accept()
    hub.rooms.setdefault(rid, {})[p["code"]] = ws
    sid = rdb.active_session(rid)
    await ws.send_json({"type": "hello", "me": {"code": p["code"], "name": p["display_name"]}, "session": sid,
                        "history": rdb.history(sid) if sid else []})
    await hub.send_all(rid, {"type": "members", "members": hub.members(rid)})
    try:
        while True:
            data = await ws.receive_json()
            if data.get("type") == "msg":
                err = await handle_message(rid, p, data)
                if err:
                    await ws.send_json(err)
    except WebSocketDisconnect:
        pass
    finally:
        hub.rooms.get(rid, {}).pop(p["code"], None)
        await hub.send_all(rid, {"type": "members", "members": hub.members(rid)})


# ------------------------------------------------------------------ participant REST
class JoinReq(BaseModel):
    token: str
    consent: bool = False


@app.post("/api/join")
def join(req: JoinReq):
    p = rdb.by_token(req.token.strip())
    if not p:
        raise HTTPException(401, "Unknown study token")
    if req.consent and not p["consent_ts"]:
        rdb.consent(p["code"])
        p = rdb.by_token(req.token.strip())
    if not p["consent_ts"]:
        raise HTTPException(403, "Consent is required to take part")
    role = rdb.role_of(p["code"]) or {"role": "normal", "target": None}
    names = {x["code"]: x["display_name"] for x in rdb.participants()}
    return {"code": p["code"], "name": p["display_name"], "rooms": rdb.rooms(),
            "role": role["role"], "role_text": ROLES[role["role"]],
            "target_name": names.get(role.get("target")) if role.get("target") else None,
            "bank": role["role"] in BANK_ROLES}


@app.get("/api/bank")
def bank(token: str, bucket: str = "mild"):
    p = rdb.by_token(token)
    role = (rdb.role_of(p["code"]) or {}).get("role") if p else None
    if role not in BANK_ROLES or bucket not in BANK:
        raise HTTPException(403, "message bank not available for this role")
    return {"bucket": bucket, "text": random.choice(BANK[bucket])}


# ------------------------------------------------------------------ researcher REST
class NewParticipant(BaseModel):
    display_name: str = Field(..., min_length=2, max_length=32)
    is_bot: bool = False


class NewRoom(BaseModel):
    name: str = Field(..., min_length=2, max_length=40)
    owner: str


class Assignment(BaseModel):
    participant: str
    role: str
    target: str | None = None
    note: str | None = None


class Label(BaseModel):
    sender: str
    target: str
    label: int = Field(..., ge=0, le=1)
    note: str | None = None


R = [Depends(researcher)]


@app.post("/api/admin/participants", dependencies=R)
def add_participant(req: NewParticipant):
    p = rdb.add_participant(req.display_name, req.is_bot)
    _hash2code[engine.pseudo(p["code"], PLATFORM)] = p["code"]
    return p


def engine_now() -> float:
    """Current time on the lab clock (latest session, real time elapsed in it)."""
    rows = rdb.q("SELECT session_id, started_ts, ended_ts FROM sessions ORDER BY started_ts DESC LIMIT 1")
    if not rows:
        return time.time()
    r = rows[0]
    return rdb.lab_clock(r["session_id"], r["ended_ts"] or time.time(), DAYS_PER_SESSION)


@app.get("/api/admin/participants", dependencies=R)
def list_participants():
    roles = {a["participant"]: a for a in rdb.q("SELECT * FROM assignments WHERE active=1")}
    out = []
    now = engine_now()
    for p in rdb.participants():
        h = engine.pseudo(p["code"], PLATFORM)
        tier, risk, targets = "Low", 0.0, 0
        for st in engine.store.pairs_for_offender(h):
            f = pair_features(st, now, engine.cfg)
            r = risk_probability(f, engine.cfg)
            if r > risk:
                risk, tier = r, tier_for(r, engine.cfg)
            targets += st.r_hostile > 0
        a = roles.get(p["code"], {})
        out.append({**p, "role": a.get("role", "normal"), "target": a.get("target"), "risk": round(risk, 4),
                    "tier": tier, "online": any(p["code"] in m for m in hub.rooms.values()), "targets_harassed": targets})
    return out


@app.post("/api/admin/rooms", dependencies=R)
def add_room(req: NewRoom):
    return rdb.add_room(req.name, req.owner)


@app.get("/api/admin/rooms", dependencies=R)
def list_rooms():
    return rdb.rooms()


@app.post("/api/admin/rooms/{rid}/start", dependencies=R)
async def start(rid: str):
    sid = rdb.start_session(rid)
    await hub.send_all(rid, {"type": "session", "state": "live", "session": sid})
    return {"session_id": sid}


@app.post("/api/admin/rooms/{rid}/end", dependencies=R)
async def end(rid: str, save_report: bool = False):
    sid = rdb.end_session(rid)
    await hub.send_all(rid, {"type": "session", "state": "ended", "session": sid})
    out = {"ended": sid}
    if save_report and sid:  # "Auto-save report when a session ends" (researcher console checkbox)
        out["report"] = await asyncio.to_thread(save_session_report, sid)
    return out


REPORTS_DIR = DATA_DIR / "reports"


def make_report(session_id: str, mask_text: bool, include_truth: bool, codes_only: bool) -> dict:
    return session_report.build_report(rdb, session_id, engine.cfg, content_model=engine.scorer.name if engine.scorer else None,
                                       mask_text=mask_text, include_truth=include_truth, codes_only=codes_only)


def save_session_report(session_id: str) -> str:
    """Writes data/safechat/reports/<session_id>.html (message text hidden, as for any exported file)."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / f"{session_id}.html"
    path.write_text(session_report.render_html(make_report(session_id, True, False, False)), encoding="utf-8")
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


@app.get("/api/admin/sessions/{session_id}/report", dependencies=R)
def get_session_report(session_id: str, format: str = "html", mask_text: bool = True, include_truth: bool = False,
                       codes_only: bool = False):
    """Session report built only from engine outputs. include_truth adds the separate researcher appendix."""
    if format not in ("html", "pdf", "csv"):
        raise HTTPException(422, "format must be html, pdf or csv")
    try:
        rep = make_report(session_id, mask_text, include_truth, codes_only)
    except KeyError:
        raise HTTPException(404, f"unknown session {session_id}")
    if format == "html":
        return Response(session_report.render_html(rep), media_type="text/html; charset=utf-8",
                        headers={"Content-Disposition": f'inline; filename="session_report_{session_id}.html"'})
    if format == "pdf":
        return Response(session_report.render_pdf(rep), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="session_report_{session_id}.pdf"'})
    return Response(session_report.render_csv_zip(rep), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="session_report_{session_id}_csv.zip"'})


@app.get("/api/admin/sessions", dependencies=R)
def sessions():
    return rdb.sessions()


@app.post("/api/admin/assignments", dependencies=R)
def assign(req: Assignment):
    try:
        rdb.assign(req.participant, req.role, req.target, None, req.note)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return {"ok": True}


@app.get("/api/admin/roles", dependencies=R)
def roles():
    return ROLES


@app.post("/api/admin/labels", dependencies=R)
def label(req: Label):
    rdb.set_label(req.sender, req.target, req.label, note=req.note)
    return {"ok": True}


class Resolve(BaseModel):
    conflict_id: str
    action: str = Field(..., pattern="^(use_role|keep_manual)$")
    note: str | None = None


@app.get("/api/admin/label-check", dependencies=R)
def label_check():
    """Conflicts between role assignments and manual labels (never changed silently)."""
    reviewed = {r["conflict_id"]: r for r in rdb.reviews()}
    conflicts = rdb.label_conflicts()
    for c in conflicts:
        # removing the manual override only makes sense when a manual label is involved
        c["can_use_role"] = c["kind"] in ("positive_role_manual_0", "manual_1_negative_role")
    return {"conflicts": [c for c in conflicts if c["id"] not in reviewed],
            "reviewed": [{**json.loads(r["detail"] or "{}"), "id": r["conflict_id"], "decision": r["decision"], "ts": r["ts"]}
                         for r in reviewed.values()]}


@app.post("/api/admin/label-check/resolve", dependencies=R)
def label_resolve(req: Resolve):
    c = next((c for c in rdb.label_conflicts() if c["id"] == req.conflict_id), None)
    if c is None:
        raise HTTPException(404, "conflict not found (it may already be resolved)")
    if req.action == "use_role":
        if c["kind"] not in ("positive_role_manual_0", "manual_1_negative_role"):
            raise HTTPException(422, "this conflict has no manual label to remove; use keep_manual to mark it reviewed")
        rdb.clear_label(c["sender"], c["target"])
        decision = "used_role_label"
    else:
        decision = "kept_manual_label" if c["kind"] in ("positive_role_manual_0", "manual_1_negative_role") else "reviewed"
    rdb.review_conflict(c, decision, note=req.note)
    return {"ok": True, "decision": decision}


@app.get("/api/admin/pairs", dependencies=R)
def pairs():
    gt = {(p["sender"], p["target"]): p for p in rdb.ground_truth_pairs()}
    engine.store.flush()
    now = engine_now()
    out = []
    for st in engine.store.all_pairs():
        s, t = _code_of(st.offender), _code_of(st.target)
        f = pair_features(st, now, engine.cfg)
        r = risk_probability(f, engine.cfg)
        g = gt.get((s, t), {})
        out.append({"sender": s, "target": t, "risk_now": round(r, 4), "tier_now": tier_for(r, engine.cfg),
                    "tier_last": st.tier, "hostile_messages": st.r_hostile, "sessions_targeted": st.s_hostile,
                    "messages": st.r_total, "label": g.get("label"), "label_source": g.get("label_source"),
                    "role": g.get("role")})
    # Overall status of a user = their highest pair tier/risk right now (across all their targets)
    order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    overall: dict[str, dict] = {}
    for d in out:
        o = overall.setdefault(d["sender"], {"tier": "Low", "risk": 0.0, "targets_high": 0})
        if d["risk_now"] > o["risk"]:
            o["risk"] = d["risk_now"]
        if order[d["tier_now"]] > order[o["tier"]]:
            o["tier"] = d["tier_now"]
        o["targets_high"] += order[d["tier_now"]] >= 2
    for d in out:
        o = overall[d["sender"]]
        d["sender_overall_tier"], d["sender_overall_risk"] = o["tier"], round(o["risk"], 4)
        d["sender_targets_high"] = o["targets_high"]
    out.sort(key=lambda d: -d["risk_now"])
    return out


@app.get("/api/admin/feed", dependencies=R)
def feed(limit: int = 60):
    rows = rdb.q("""SELECT m.message_id, m.ts, m.session_id, m.sender, m.target, m.target_method, m.text, m.T,
                    m.tier, m.risk, m.user_tier, m.user_risk, m.alerts FROM messages m ORDER BY ts DESC LIMIT ?""", (min(limit, 300),))
    for r in rows:
        r["alerts"] = json.loads(r["alerts"] or "[]")
    return rows


@app.get("/api/admin/stats", dependencies=R)
def stats():
    return {**rdb.counts(), "days_per_session": DAYS_PER_SESSION, "content_model": engine.scorer.name if engine.scorer else None,
            "engine": engine.store.stats()}


@app.get("/api/admin/export", dependencies=R)
def export(include_text: bool = True):
    """Dataset export (zip of CSVs). Display names are never exported - only study codes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        def write(name, rows, cols):
            s = io.StringIO()
            w = csv.DictWriter(s, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
            z.writestr(name, s.getvalue())
        msgs = rdb.q("SELECT * FROM messages ORDER BY ts")
        for m in msgs:
            m["iso_time"] = datetime.fromtimestamp(m["ts"], timezone.utc).isoformat()
            if not include_text:
                m["text"] = ""
        write("messages.csv", msgs, ["message_id", "session_id", "room_id", "ts", "engine_ts", "iso_time", "sender", "text", "reply_to",
                                     "mentions", "target", "target_method", "from_bank", "T", "p_threat", "risk", "tier",
                                     "user_tier", "user_risk"])
        reviews = rdb.reviews()
        by_pair: dict = {}
        for r in reviews:
            by_pair.setdefault((r["sender"], r["target"]), []).append(f"{r['kind']}:{r['decision']}")
        gt = rdb.ground_truth_pairs()
        for p in gt:
            p["label_review"] = ";".join(by_pair.get((p["sender"], p["target"]), []))
        write("pairs.csv", gt, ["sender", "target", "n", "sessions", "first_ts", "last_ts",
                                "role", "label", "label_source", "label_review"])
        write("label_reviews.csv", reviews, ["conflict_id", "kind", "sender", "target", "decision", "detail", "annotator", "ts"])
        roles = {a["participant"]: a for a in rdb.q("SELECT * FROM assignments WHERE active=1")}
        write("participants.csv", [{"code": p["code"], "is_bot": p["is_bot"], "role": roles.get(p["code"], {}).get("role", "normal"),
                                    "target": roles.get(p["code"], {}).get("target"), "consented": bool(p["consent_ts"])}
                                   for p in rdb.participants()], ["code", "is_bot", "role", "target", "consented"])
        write("sessions.csv", rdb.sessions(), ["session_id", "room_id", "started_ts", "ended_ts", "messages", "senders"])
        write("assignments_history.csv", rdb.assignments(), ["participant", "role", "target", "note", "created_ts", "active"])
        z.writestr("README.txt", EXPORT_README)
    buf.seek(0)
    name = f"safechat_dataset_{datetime.now():%Y%m%d_%H%M}.zip"
    return StreamingResponse(buf, media_type="application/zip", headers={"Content-Disposition": f"attachment; filename={name}"})


EXPORT_README = """SafeChat Lab - cross-session harassment dataset export
messages.csv   one row per chat message. sender/target are study codes (P001 = human, B001 = bot).
               target = who the message was aimed at, resolved by the engine (reply > @mention > name > room owner);
               target_method says which. from_bank = message-bank bucket if the participant used a suggested message.
               T, p_threat = content model output; risk, tier = engine output at that moment.
               ts = real wall-clock time; engine_ts = lab-clock time the engine used (each session is
               mapped SAFECHAT_DAYS_PER_SESSION days after the previous one).
pairs.csv      every observed (sender, target) pair. label 1 = repeat harassment. label_source = manual annotation
               or role_assignment (a harasser/subtle_harasser/escalator role aimed at that target).
               label_review = label-integrity conflicts on this pair and how they were resolved.
label_reviews.csv  every reviewed label conflict (Pairs & labels -> label-check panel).
participants.csv  roles only - display names are never exported.
sessions.csv   chat sessions (one live stream = one session).
Evaluate with:  python scripts/evaluate_collected.py --export <this zip>
"""


# ------------------------------------------------------------------ pages
@app.get("/")
def chat_page():
    return FileResponse(STATIC / "chat.html")


@app.get("/researcher")
def researcher_page():
    return FileResponse(STATIC / "researcher.html")


@app.get("/health")
def health():
    return {"status": "ok", "content_model": engine.scorer.name if engine.scorer else None, **rdb.counts()}
