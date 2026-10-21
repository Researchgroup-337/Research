"""Bot simulator for SafeChat Lab: fills the test chat with scripted users over several
sessions. Real participants can join the same room at the same time, giving a hybrid dataset.

    python scripts/run_chat_bots.py --sessions 6 --minutes 2      # bots run 6 sessions on their own
    python scripts/run_chat_bots.py --follow                      # bots join the sessions YOU start
    python scripts/run_chat_bots.py --follow --room "test"        # ... in a room of your choice

The first run creates the bots, their roles (ground truth) and a room; later runs reuse
them (saved in data/safechat/bots.json) so the same bots return across sessions, which is
exactly the cross-session behaviour the engine has to detect.
--victim P001 makes the harasser bots target a human participant (e.g. a researcher persona).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

import httpx
import websockets

ROOT = Path(__file__).resolve().parents[1]
BANK = json.loads((ROOT / "data" / "message_bank.json").read_text(encoding="utf-8"))["buckets"]
STATE = ROOT / "data" / "safechat" / "bots.json"

# persona -> (count, bucket mix for messages aimed at the target/friend)
PERSONAS = {
    "normal": (9, None), "banter": (2, None), "harasser": (2, None), "subtle_harasser": (1, None),
    "escalator": (1, None), "troll": (1, None), "burst": (1, None),
}
MIX = {
    "normal": {"clean": .96, "mild": .04}, "banter": {"clean": .55, "mild": .45},
    "harasser": {"clean": .25, "mild": .45, "severe": .2, "threat": .1},
    "subtle_harasser": {"clean": .55, "mild": .4, "severe": .05},
    "attack": {"clean": .1, "mild": .45, "severe": .3, "threat": .15},
}
NAMES = ["Nimal", "Kasun", "Sachini", "Dilan", "Ishara", "Ruwan", "Tharushi", "Amaya", "Chamod", "Hiruni",
         "Pasan", "Sanduni", "Yasith", "Nethmi", "Kavindu", "Dinithi", "Malith", "Senuri", "Lahiru", "Oshadi"]


def pick(mix):
    r, acc = random.random(), 0.0
    for k, p in mix.items():
        acc += p
        if r <= acc:
            return k
    return "clean"


def setup(c: httpx.Client, victim: str | None) -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    bots = []
    host = c.post("/api/admin/participants", json={"display_name": "StreamHost", "is_bot": True}).json()
    names = iter(NAMES)
    for persona, (n, _) in PERSONAS.items():
        for _ in range(n):
            b = c.post("/api/admin/participants", json={"display_name": next(names), "is_bot": True}).json()
            b["persona"] = persona
            bots.append(b)
    normals = [b for b in bots if b["persona"] == "normal"]
    for b in bots:
        target = None
        if b["persona"] in ("harasser", "subtle_harasser", "escalator", "burst"):
            target = victim if (victim and b["persona"] == "harasser") else random.choice(normals)["code"]
        elif b["persona"] == "banter":
            target = next(x["code"] for x in bots if x["persona"] == "banter" and x is not b)
        b["target"] = target
        c.post("/api/admin/assignments", json={"participant": b["code"], "role": b["persona"], "target": target})
    for b in normals:  # friends for ordinary chatter
        b["friends"] = [x["code"] for x in random.sample(normals, 3) if x is not b]
    room = c.post("/api/admin/rooms", json={"name": "Lab Stream", "owner": host["code"]}).json()
    state = {"host": host, "bots": bots, "room": room["room_id"], "sessions_run": 0}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


async def bot_loop(url, rid, bot, names, live, session_idx, n_sessions, mean_gap):
    """live() -> True while the bots should keep chatting in this session."""
    p = bot["persona"]
    # attendance: harassers keep coming back; others drop in and out
    attend = {"harasser": .9, "subtle_harasser": .85, "escalator": .9}.get(p, .7)
    if p == "burst":
        attend = 1.0 if session_idx == bot.get("burst_session", 1) else .5
    if random.random() > attend:
        return 0
    ws_url = url.replace("http", "ws", 1) + f"/ws/{rid}?token={bot['token']}"
    sent = 0
    burst_now = p == "burst" and session_idx == bot.get("burst_session", 1)
    burst_left = random.randint(15, 25) if burst_now else 0
    async with websockets.connect(ws_url, max_queue=None, close_timeout=2) as ws:
        await ws.recv()  # hello

        async def reader():  # keep consuming broadcasts so keep-alive pings are answered
            try:
                async for _ in ws:
                    pass
            except Exception:
                pass
        rtask = asyncio.create_task(reader())
        while live():
            gap = mean_gap / 4 if (p == "burst" and session_idx == bot.get("burst_session", 1)) else mean_gap
            wait = random.expovariate(1 / gap)
            while wait > 0 and live():  # sleep in short steps so bots leave quickly when the session ends
                await asyncio.sleep(min(1.0, wait))
                wait -= 1.0
            if not live():
                break
            target, reply, bucket = None, False, "clean"
            r = random.random()
            if p in ("harasser", "subtle_harasser", "escalator") and r < .7:
                target = bot["target"]
                if p == "escalator":
                    prog = session_idx / max(1, n_sessions - 1)
                    mix = {"clean": .55 - .45 * prog, "mild": .4 - .05 * prog, "severe": .05 + .3 * prog, "threat": .2 * prog}
                else:
                    mix = MIX[p]
                bucket = pick(mix)
            elif burst_now and burst_left > 0 and r < .8:
                target, bucket = bot["target"], pick(MIX["attack"])
                burst_left -= 1
            elif p == "troll" and r < .5:
                target, bucket = random.choice([c for c in names if c != bot["code"]]), pick(MIX["attack"])
            elif p == "banter" and r < .6:
                target, bucket = bot["target"], pick(MIX["banter"])
            elif r < .35 and bot.get("friends"):
                target, bucket = random.choice(bot["friends"]), pick(MIX["normal"])
            else:
                bucket = pick(MIX["normal"])
            text = random.choice(BANK[bucket])
            msg = {"type": "msg", "text": text, "from_bank": bucket}
            if target:
                if random.random() < .5:
                    msg["reply_to"] = target
                elif random.random() < .75:  # some aimed messages carry no mention (resolution noise)
                    msg["text"] = f"@{names[target]} {text}"
            try:
                await ws.send(json.dumps(msg))
            except websockets.ConnectionClosed:
                break
            sent += 1
        rtask.cancel()
    return sent


def find_room(c: httpx.Client, wanted: str | None, default_rid: str) -> dict:
    rooms = c.get("/api/admin/rooms").json()
    if not wanted:
        return next(r for r in rooms if r["room_id"] == default_rid)
    for r in rooms:
        if wanted.lower() in (r["room_id"].lower(), r["name"].lower()):
            return r
    sys.exit(f"Room '{wanted}' not found. Rooms: " + ", ".join(f"{r['name']} ({r['room_id']})" for r in rooms))


async def follow(args, c: httpx.Client, state: dict, names: dict):
    """Researcher-driven mode: wait for the researcher to start a session in the room, chat while it
    is live, stop when the researcher ends it, then wait for the next one. Ctrl+C to quit."""
    room = find_room(c, args.room, state["room"])
    rid = room["room_id"]
    print(f"Bots are following room '{room['name']}' ({rid}). Start/End sessions in the researcher console.")
    print("Press Ctrl+C to stop the bots.\n")
    while True:
        sid = None
        waiting_msg = False
        while not sid:
            sid = find_room(c, rid, rid)["active_session"]
            if not sid:
                if not waiting_msg:
                    print("  waiting for the researcher to click 'Start session' ...")
                    waiting_msg = True
                await asyncio.sleep(2)
        names = {p["code"]: p["display_name"] for p in c.get("/api/admin/participants").json()}
        idx = state["sessions_run"]
        flag = {"on": True}

        async def watcher():
            while flag["on"]:
                await asyncio.sleep(2)
                try:
                    cur = (await asyncio.to_thread(find_room, c, rid, rid))["active_session"]
                except Exception:
                    cur = None
                if cur != sid:
                    flag["on"] = False
        wtask = asyncio.create_task(watcher())
        print(f"  session {sid} is LIVE - bots joining")
        counts = await asyncio.gather(*[bot_loop(args.url, rid, b, names, lambda: flag["on"], idx,
                                                 max(args.sessions, idx + 1), args.gap) for b in state["bots"]])
        wtask.cancel()
        state["sessions_run"] += 1
        STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"  session {sid} ended: {sum(counts)} bot messages\n")


async def run(args):
    c = httpx.Client(base_url=args.url, headers={"X-Researcher-Key": args.key}, timeout=30)
    state = setup(c, args.victim)
    if args.follow:
        for b in state["bots"]:
            b.setdefault("burst_session", random.randint(1, max(1, args.sessions - 1)))
        return await follow(args, c, state, {})
    names = {p["code"]: p["display_name"] for p in c.get("/api/admin/participants").json()}
    rid = state["room"]
    for b in state["bots"]:
        b.setdefault("burst_session", random.randint(1, max(1, args.sessions - 1)))
    for s in range(args.sessions):
        idx = state["sessions_run"]
        sid = c.post(f"/api/admin/rooms/{rid}/start").json()["session_id"]
        end_ts = time.time() + args.minutes * 60
        counts = await asyncio.gather(*[bot_loop(args.url, rid, b, names, lambda: time.time() < end_ts, idx, args.sessions + state["sessions_run"] - s, args.gap)
                                        for b in state["bots"]])
        c.post(f"/api/admin/rooms/{rid}/end")
        state["sessions_run"] += 1
        STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"session {sid}: {sum(counts)} bot messages")
        if s < args.sessions - 1:
            await asyncio.sleep(args.pause)
    pairs = c.get("/api/admin/pairs").json()
    print("\nTop pairs now:")
    for p in pairs[:10]:
        print(f"  {p['sender']:>5s} -> {p['target']:<5s} {p['tier_now']:<8s} risk={p['risk_now']:.2f} hostile={p['hostile_messages']:>2d} "
              f"sessions={p['sessions_targeted']} label={p['label']} role={p['role']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--key", default="lab-key-change-me")
    ap.add_argument("--sessions", type=int, default=6)
    ap.add_argument("--minutes", type=float, default=2.0, help="length of each session")
    ap.add_argument("--gap", type=float, default=15.0, help="mean seconds between a bot's messages")
    ap.add_argument("--pause", type=float, default=5.0, help="seconds between sessions")
    ap.add_argument("--victim", help="study code of a HUMAN participant for the harasser bots to target")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--follow", action="store_true",
                    help="do NOT start/end sessions: join whenever the researcher starts a session, leave when it ends")
    ap.add_argument("--room", help="room name or id for --follow (default: the bots' 'Lab Stream' room)")
    args = ap.parse_args()
    random.seed(args.seed)
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nBots stopped.")


if __name__ == "__main__":
    sys.exit(main())
