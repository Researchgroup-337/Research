"""Session Report for SafeChat Lab: what happened in one chat session, for the streamer/researcher.

GOLDEN RULE (research validity): every number, chart and category in the report is computed ONLY
from the engine outputs logged with each message: content score T, P(threat), the pair's risk and
tier after the message, the engine's hostile flag, hostile-message and hostile-session counts, flags
and alerts. Role assignments and ground-truth labels are never read for the report itself, because
a real streamer would not know them. They appear only in the optional, clearly separated
"Researcher appendix" (include_truth=True), built by `load_truth()`.

Messages logged before the engine fields `hostile`, `pair_hostile` and `pair_sessions` were added to
the research log are completed with the engine's own rule (risk.update_pair): a message is hostile
when T >= hostile_threshold, and a pair's hostile-session count goes up on the first hostile
message of each new session.

Entry points:
    build_report(rdb, session_id, cfg, ...) -> dict
    render_html(report) / render_pdf(report) / render_csv_zip(report)
"""
from __future__ import annotations

import base64
import csv
import html
import io
import json
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime

from ..risk import TIER_RANK, TIERS

# ---------------------------------------------------------------- category thresholds
REPEAT_MIN_TIER = "High"
"""Repeat harasser: the pair's tier at the end of the session is at least this..."""
REPEAT_MIN_SESSIONS = 2
"""...AND the engine has counted hostility towards that target in at least this many sessions."""
BURST_MIN_HOSTILE = 5
"""Burst attacker: at least this many hostile messages to ONE target within this session..."""
BURST_MAX_SESSIONS = 1
"""...and that pair's hostile-session count is still this value (hostility only in this session)."""
TROLL_MIN_TARGETS = 3
"""Troll: hostile messages to at least this many DIFFERENT targets in this session..."""
TROLL_MAX_TIER = "Medium"
"""...with none of the sender's pairs above this tier at the end of the session (no pair reaching High)."""
COORDINATED_ALERT = "coordinated_harassment"
"""Victim side: targets for which the engine raised this alert during the session
(>= pressure_min_senders distinct hostile senders within pressure_window_days)."""
# Explicit threat: any message with P(threat) >= cfg.threat_override_p (the engine's override threshold).

CATEGORIES = ["repeat_harasser", "explicit_threat", "burst_attacker", "troll", "normal"]
CATEGORY_NAMES = {"repeat_harasser": "Repeat harasser", "explicit_threat": "Explicit threat",
                  "burst_attacker": "Burst attacker", "troll": "Troll", "normal": "Normal",
                  "coordinated_target": "Coordinated harassment (victim)"}
DEFINITIONS = {
    "repeat_harasser": f"a pair at {REPEAT_MIN_TIER} or Critical at the end of the session whose target was attacked "
                       f"in at least {REPEAT_MIN_SESSIONS} sessions.",
    "explicit_threat": "sent at least one message with P(threat) at or above the engine's explicit-threat override "
                       "threshold ({p}).",
    "burst_attacker": f"{BURST_MIN_HOSTILE} or more hostile messages to one target in this session, with no hostility "
                      "towards that target in any earlier session.",
    "troll": f"hostile messages to {TROLL_MIN_TARGETS} or more different people in this session, without any single "
             "pair reaching High.",
    "coordinated_target": "a person for whom the engine raised a coordinated-harassment alert in this session "
                          "(several different hostile senders within a few days).",
    "normal": "active in the session and in none of the offender categories above.",
}
OKABE = {"blue": "#0072B2", "orange": "#E69F00", "vermillion": "#D55E00", "sky": "#56B4E9", "green": "#009E73",
         "purple": "#CC79A7", "grey": "#8C8C8C", "ink": "#222222"}
TIER_COLORS = {"Low": "#cde2fb", "Medium": "#86b6ef", "High": "#2a78d6", "Critical": "#0d366b"}
HIDDEN = "[hidden]"


# ---------------------------------------------------------------- data access (engine outputs only)
MSG_COLS = ("message_id, session_id, room_id, ts, sender, target, target_method, text, T, p_threat, risk, tier, "
            "alerts, hostile, pair_hostile, pair_sessions, flags, user_tier")


def load_session(rdb, session_id: str) -> dict:
    """Reads sessions, rooms, the message log (engine outputs) and display names. Nothing else."""
    s = rdb.one("SELECT session_id, room_id, started_ts, ended_ts FROM sessions WHERE session_id=?", (session_id,))
    if not s:
        raise KeyError(f"unknown session {session_id}")
    room = rdb.one("SELECT room_id, name, owner FROM rooms WHERE room_id=?", (s["room_id"],)) or \
        {"room_id": s["room_id"], "name": s["room_id"], "owner": None}
    prev = rdb.one("SELECT session_id, room_id, started_ts, ended_ts FROM sessions WHERE room_id=? AND started_ts<? "
                   "ORDER BY started_ts DESC LIMIT 1", (s["room_id"], s["started_ts"]))
    end = s["ended_ts"] or time.time()
    hist = rdb.q(f"SELECT {MSG_COLS} FROM messages WHERE ts<=? ORDER BY ts, rowid", (end,))
    for r in hist:
        r["alerts"] = json.loads(r["alerts"] or "[]")
        r["flags"] = json.loads(r["flags"]) if r.get("flags") else None
    names = {r["code"]: r["display_name"] for r in rdb.q("SELECT code, display_name FROM participants")}
    return {"session": s, "room": room, "prev": prev, "end_ts": end, "history": hist, "names": names}


def load_truth(rdb) -> dict:
    """Researcher appendix ONLY: role assignments and ground-truth pair labels."""
    roles = {a["participant"]: a for a in rdb.q("SELECT participant, role, target FROM assignments WHERE active=1")}
    labels = {(p["sender"], p["target"]): p for p in rdb.ground_truth_pairs()}
    return {"roles": roles, "labels": labels}


# ---------------------------------------------------------------- computation
def _is_hostile(r: dict, threshold: float) -> bool:
    if r.get("hostile") is not None:
        return bool(r["hostile"])
    return (r.get("T") or 0.0) >= threshold


def pair_states(rows: list[dict], threshold: float) -> dict:
    """State of each (sender, target) pair after the last of `rows` (time-ordered): the engine's tier and
    risk after its last message, hostile-message count and hostile-session count."""
    st: dict = {}
    for r in rows:
        p = st.setdefault((r["sender"], r["target"]), {"hostile": 0, "sessions": 0, "_last_s": None, "tier": "Low",
                                                       "risk": 0.0, "messages": 0, "last_ts": None})
        p["messages"] += 1
        if _is_hostile(r, threshold):
            p["hostile"] += 1
            if p["_last_s"] != r["session_id"]:
                p["sessions"] += 1
                p["_last_s"] = r["session_id"]
        if r.get("pair_hostile") is not None:
            p["hostile"] = r["pair_hostile"]
        if r.get("pair_sessions") is not None:
            p["sessions"] = r["pair_sessions"]
        p["tier"] = r.get("tier") or "Low"
        p["risk"] = r.get("risk") or 0.0
        p["last_ts"] = r["ts"]
    return st


def categorise(rows: list[dict], end_state: dict, cfg) -> dict:
    """Behaviour categories for one session. rows = this session's messages; end_state = pair_states() at
    the end of the session. Returns {category: set(codes)} plus 'coordinated_target'."""
    thr = cfg.hostile_threshold
    hostile_to = defaultdict(Counter)
    active_pairs = defaultdict(set)
    senders = set()
    for r in rows:
        senders.add(r["sender"])
        active_pairs[r["sender"]].add(r["target"])
        if _is_hostile(r, thr):
            hostile_to[r["sender"]][r["target"]] += 1
    tier_of = lambda s, t: end_state.get((s, t), {}).get("tier", "Low")  # noqa: E731
    sess_of = lambda s, t: end_state.get((s, t), {}).get("sessions", 0)  # noqa: E731
    cats = {c: set() for c in CATEGORIES}
    for s in senders:
        for t in active_pairs[s]:
            if TIER_RANK[tier_of(s, t)] >= TIER_RANK[REPEAT_MIN_TIER] and sess_of(s, t) >= REPEAT_MIN_SESSIONS:
                cats["repeat_harasser"].add(s)
        for t, n in hostile_to[s].items():
            if n >= BURST_MIN_HOSTILE and sess_of(s, t) == BURST_MAX_SESSIONS:
                cats["burst_attacker"].add(s)
        if len(hostile_to[s]) >= TROLL_MIN_TARGETS and \
                all(TIER_RANK[tier_of(s, t)] <= TIER_RANK[TROLL_MAX_TIER] for t in active_pairs[s]):
            cats["troll"].add(s)
    for r in rows:
        if (r.get("p_threat") or 0.0) >= cfg.threat_override_p:
            cats["explicit_threat"].add(r["sender"])
    cats["normal"] = senders - set().union(*(cats[c] for c in CATEGORIES if c != "normal"))
    cats["coordinated_target"] = {r["target"] for r in rows for a in r["alerts"] if a.get("kind") == COORDINATED_ALERT}
    return cats


def build_report(rdb, session_id: str, cfg, *, content_model: str | None = None, mask_text: bool = True,
                 include_truth: bool = False, codes_only: bool = False) -> dict:
    d = load_session(rdb, session_id)
    s, hist, names = d["session"], d["history"], d["names"]
    thr = cfg.hostile_threshold
    who = (lambda c: c or "–") if codes_only else (lambda c: f"{names[c]} ({c})" if c in names else (c or "–"))
    rows = [r for r in hist if r["session_id"] == session_id]
    end_state = pair_states(hist, thr)
    prev = d["prev"]
    prev_state = pair_states([r for r in hist if prev and r["ts"] <= (prev["ended_ts"] or s["started_ts"])], thr) \
        if prev else {}
    cats = categorise(rows, end_state, cfg)
    user_cats = defaultdict(list)
    for c in CATEGORIES + ["coordinated_target"]:
        for u in cats[c]:
            user_cats[u].append(c)

    hostile_rows = [r for r in rows if _is_hostile(r, thr)]
    threat_rows = [r for r in rows if (r.get("p_threat") or 0.0) >= cfg.threat_override_p]
    alerts = [(r, a) for r in rows for a in r["alerts"]]
    active_pairs = sorted({(r["sender"], r["target"]) for r in rows})
    senders = sorted({r["sender"] for r in rows})

    def highest(state, sender, only=None):
        ts = [v["tier"] for (a, t), v in state.items() if a == sender and (only is None or (a, t) in only)]
        return max(ts, key=TIER_RANK.get) if ts else None

    kpi = {"messages": len(rows), "active_users": len(senders), "hostile": len(hostile_rows),
           "hostile_pct": round(100 * len(hostile_rows) / len(rows), 1) if rows else 0.0, "alerts": len(alerts),
           "repeat_harassers": len(cats["repeat_harasser"]), "trolls": len(cats["troll"]),
           "burst_attackers": len(cats["burst_attacker"]), "explicit_threats": len(threat_rows),
           "explicit_threat_senders": len(cats["explicit_threat"]),
           "users_targeted": len({r["target"] for r in hostile_rows}),
           "coordinated_targets": len(cats["coordinated_target"])}

    # Overall tier = the engine's user_max_tier (highest tier across all of the user's targets), as logged with
    # the user's last message up to the end of this session; rows logged before user_tier existed fall back to
    # the highest pair tier across all of the user's pairs.
    logged_overall = {r["sender"]: r["user_tier"] for r in hist if r.get("user_tier")}
    sent = Counter(r["sender"] for r in hostile_rows)
    offenders = []
    for u, n in sent.most_common():
        now_t = highest(end_state, u, set(active_pairs))
        before = highest(prev_state, u)
        if before is None:
            change = "new"
        else:
            diff = TIER_RANK[now_t] - TIER_RANK[before]
            change = f"{before} → {now_t}" + (" ▲" if diff > 0 else " ▼" if diff < 0 else "")
        offenders.append({"code": u, "name": who(u), "hostile": n,
                          "targets": len({r["target"] for r in hostile_rows if r["sender"] == u}),
                          "highest_tier": now_t, "categories": [c for c in user_cats[u] if c != "coordinated_target"],
                          "change": change, "prev_tier": before,
                          "overall_tier": logged_overall.get(u) or highest(end_state, u)})
    received = Counter(r["target"] for r in hostile_rows)
    targeted = [{"code": t, "name": who(t), "hostile_received": n,
                 "senders": len({r["sender"] for r in hostile_rows if r["target"] == t}),
                 "coordinated": t in cats["coordinated_target"]} for t, n in received.most_common()]
    alert_rows = [{"ts": r["ts"], "kind": a.get("kind"), "from": a.get("from"), "to": a.get("to"),
                   "sender": r["sender"], "target": r["target"], "senders": a.get("senders")} for r, a in alerts]
    pairs = []
    for (u, t) in active_pairs:
        e, p = end_state[(u, t)], prev_state.get((u, t))
        in_s = [r for r in rows if r["sender"] == u and r["target"] == t]
        pairs.append({"sender": u, "target": t, "messages": len(in_s),
                      "hostile": sum(_is_hostile(r, thr) for r in in_s), "tier": e["tier"], "risk": e["risk"],
                      "prev_tier": p["tier"] if p else None, "hostile_total": e["hostile"], "sessions": e["sessions"],
                      "max_T": max((r["T"] or 0.0) for r in in_s),
                      "explicit_threat": any((r.get("p_threat") or 0.0) >= cfg.threat_override_p for r in in_s)})
    escalations = [p for p in pairs if TIER_RANK[p["tier"]] > TIER_RANK[p["prev_tier"] or "Low"]]
    escalations.sort(key=lambda p: (-TIER_RANK[p["tier"]], -p["risk"]))
    evidence = []
    for p in sorted(pairs, key=lambda p: (-TIER_RANK[p["tier"]], -p["risk"])):
        if TIER_RANK[p["tier"]] < TIER_RANK["High"]:
            continue
        msgs = sorted([r for r in rows if r["sender"] == p["sender"] and r["target"] == p["target"]],
                      key=lambda r: -(r["T"] or 0.0))[:3]
        evidence.append({**p, "messages": [{"ts": r["ts"], "T": r["T"] or 0.0, "text": HIDDEN if mask_text else r["text"],
                                            "threat": (r.get("p_threat") or 0.0) >= cfg.threat_override_p} for r in msgs]})

    rep = {"session_id": session_id, "room": d["room"], "started_ts": s["started_ts"], "ended_ts": s["ended_ts"],
           "prev_session": prev["session_id"] if prev else None, "content_model": content_model or "as logged",
           "generated_ts": time.time(), "options": {"mask_text": mask_text, "include_truth": include_truth,
                                                    "codes_only": codes_only},
           "thresholds": {"hostile": thr, "threat_override_p": cfg.threat_override_p},
           "kpi": kpi, "categories": {k: sorted(v) for k, v in cats.items()}, "user_categories": dict(user_cats),
           "offenders": offenders, "targeted": targeted, "alerts": alert_rows, "pairs": pairs,
           "escalations": escalations, "evidence": evidence, "who": who, "rows": rows,
           "names": {} if codes_only else names}
    rep["charts"] = _charts(rep)
    rep["truth"] = _truth_appendix(load_truth(rdb), rep) if include_truth else None
    return rep


def _truth_appendix(truth: dict, rep: dict) -> dict:
    roles, labels = truth["roles"], truth["labels"]
    by_role = defaultdict(Counter)
    for u in {r["sender"] for r in rep["rows"]}:
        role = roles[u]["role"] if u in roles else "normal"
        for c in rep["user_categories"].get(u, ["normal"]):
            if c != "coordinated_target":
                by_role[role][c] += 1
    conf = Counter()
    for p in rep["pairs"]:
        eng = int(TIER_RANK[p["tier"]] >= TIER_RANK[REPEAT_MIN_TIER] and p["sessions"] >= REPEAT_MIN_SESSIONS)
        lab = int((labels.get((p["sender"], p["target"])) or {}).get("label") or 0)
        conf[(lab, eng)] += 1
    return {"role_by_category": {r: dict(c) for r, c in by_role.items()}, "confusion": {f"{k[0]}{k[1]}": v for k, v in conf.items()}}


# ---------------------------------------------------------------- charts
def _fig_png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8.5)


def _bar_values(ax, bars, fmt="{:.0f}"):
    for b in bars:
        h = b.get_height()
        if h:
            ax.text(b.get_x() + b.get_width() / 2, h, fmt.format(h), ha="center", va="bottom", fontsize=8)


def _charts(rep: dict) -> list[dict]:
    """Returns [{'title', 'alt', 'png' (base64)}]; the same PNGs are embedded in the HTML and the PDF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = []
    rows, thr, who = rep["rows"], rep["thresholds"]["hostile"], rep["who"]

    def add(fig, title, alt):
        out.append({"title": title, "alt": alt, "png": _fig_png(fig)})

    # 1. messages per minute, hostile vs not
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    _style(ax)
    start = rep["started_ts"]
    mins = max([int((r["ts"] - start) // 60) for r in rows] or [0]) + 1
    hos = [0] * mins
    non = [0] * mins
    for r in rows:
        m = int((r["ts"] - start) // 60)
        (hos if _is_hostile(r, thr) else non)[m] += 1
    x = list(range(1, mins + 1))
    b1 = ax.bar(x, non, color=OKABE["sky"], label="Not hostile")
    b2 = ax.bar(x, hos, bottom=non, color=OKABE["vermillion"], hatch="///", edgecolor="white", label=f"Hostile (T ≥ {thr})")
    for xi, n, h in zip(x, non, hos):
        if n + h:
            ax.text(xi, n + h, f"{h}/{n + h}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xlabel("Minute of session")
    ax.set_ylabel("Messages")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xticks(x if mins <= 30 else x[::max(1, mins // 15)])
    add(fig, "Messages per minute", "Stacked bars of hostile and non-hostile messages per minute; labels show hostile/total.")

    # 2. histogram of T
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    _style(ax)
    Ts = [r["T"] or 0.0 for r in rows]
    bins = [i / 20 for i in range(21)]
    n, _, patches = ax.hist(Ts, bins=bins, color=OKABE["blue"], edgecolor="white")
    for p, v in zip(patches, n):
        if p.get_x() >= thr:
            p.set_facecolor(OKABE["vermillion"])
            p.set_hatch("///")
        if v:
            ax.text(p.get_x() + p.get_width() / 2, v, f"{int(v)}", ha="center", va="bottom", fontsize=7.5)
    ax.axvline(thr, color=OKABE["ink"], ls="--", lw=1)
    ax.text(thr + 0.01, ax.get_ylim()[1] * 0.92, f"hostile threshold {thr}", fontsize=8)
    ax.set_xlabel("Content threat score T")
    ax.set_ylabel("Messages")
    add(fig, "Distribution of T", "Histogram of content threat score T; bars at or above the hostile threshold are hatched.")

    # 3. pair tiers at the end of the session
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    _style(ax)
    tc = Counter(p["tier"] for p in rep["pairs"])
    bars = ax.bar(TIERS, [tc.get(t, 0) for t in TIERS], color=[TIER_COLORS[t] for t in TIERS], edgecolor=OKABE["ink"], lw=0.5)
    _bar_values(ax, bars)
    ax.set_xlabel("Pair tier at end of session")
    ax.set_ylabel("Pairs active in session")
    add(fig, "Pair tiers at end of session", "Bar chart of how many pairs active in this session end at each tier.")

    # 4. category counts
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    _style(ax)
    keys = ["repeat_harasser", "explicit_threat", "burst_attacker", "troll", "coordinated_target", "normal"]
    labels = ["Repeat\nharasser", "Explicit\nthreat", "Burst\nattacker", "Troll", "Coordinated\n(victims)", "Normal"]
    vals = [len(rep["categories"][k]) for k in keys]
    bars = ax.bar(labels, vals, color=[OKABE["vermillion"], OKABE["purple"], OKABE["orange"], OKABE["green"],
                                       OKABE["blue"], OKABE["grey"]])
    _bar_values(ax, bars)
    ax.set_ylabel("Users")
    add(fig, "Behaviour categories", "Bar chart of users per behaviour category; a user can be in several.")

    # 5. top offenders
    top = rep["offenders"][:10]
    fig, ax = plt.subplots(figsize=(6.4, max(2.0, 0.34 * len(top) + 1)))
    _style(ax)
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    if top:
        names = [o["name"] for o in top][::-1]
        vals = [o["hostile"] for o in top][::-1]
        bars = ax.barh(names, vals, color=OKABE["vermillion"])
        for b, v in zip(bars, vals):
            ax.text(v, b.get_y() + b.get_height() / 2, f" {v}", va="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No hostile messages", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Hostile messages in this session")
    add(fig, "Top 10 offenders", "Horizontal bars of hostile messages per sender, top ten.")

    # 6. who targets whom
    hrows = [r for r in rows if _is_hostile(r, thr)]
    snd = [u for u, _ in Counter(r["sender"] for r in hrows).most_common(8)]
    tgt = [t for t, _ in Counter(r["target"] for r in hrows).most_common(8)]
    fig, ax = plt.subplots(figsize=(6.4, max(2.4, 0.42 * len(snd) + 1.4)))
    if snd and tgt:
        m = [[sum(1 for r in hrows if r["sender"] == u and r["target"] == t) for t in tgt] for u in snd]
        im = ax.imshow(m, cmap="Blues", aspect="auto", vmin=0)
        vmax = max(max(row) for row in m) or 1
        for i, row in enumerate(m):
            for j, v in enumerate(row):
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                        color="white" if v > vmax * 0.55 else OKABE["ink"])
        ax.set_xticks(range(len(tgt)))
        ax.set_xticklabels([who(t) for t in tgt], rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(snd)))
        ax.set_yticklabels([who(u) for u in snd], fontsize=8)
        ax.set_xlabel("Target")
        ax.set_ylabel("Sender")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Hostile messages")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "No hostile messages", ha="center", va="center", transform=ax.transAxes)
    add(fig, "Who targets whom", "Heatmap of hostile messages from top senders (rows) to top targets (columns); each cell shows the count.")
    return out


# ---------------------------------------------------------------- HTML
def _fmt_ts(ts, with_date=False):
    if not ts:
        return "–"
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S" if with_date else "%H:%M:%S")


def _duration(a, b):
    if not a or not b:
        return "ongoing"
    sec = int(b - a)
    return f"{sec // 60} min {sec % 60:02d} s"


CSS = """
:root{--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--accent:#2a78d6;--pos:#c43b3b;
  --t-low:#cde2fb;--t-med:#86b6ef;--t-high:#2a78d6;--t-crit:#0d366b;--t-low-ink:#0d366b;--t-med-ink:#0b0b0b;--t-high-ink:#fff;--t-crit-ink:#fff;--warn:#fff4d6;--warn-line:#e8c66a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--line:#2c2c2a;--accent:#3987e5;--pos:#ff7b72;
  --t-low:#184f95;--t-med:#256abf;--t-high:#5598e7;--t-crit:#cde2fb;--t-low-ink:#f0efec;--t-med-ink:#fff;--t-high-ink:#0b0b0b;--t-crit-ink:#0d366b;--warn:#3a3214;--warn-line:#8a7428}}
:root[data-theme="dark"]{--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--line:#2c2c2a;--accent:#3987e5;--pos:#ff7b72;
  --t-low:#184f95;--t-med:#256abf;--t-high:#5598e7;--t-crit:#cde2fb;--t-low-ink:#f0efec;--t-med-ink:#fff;--t-high-ink:#0b0b0b;--t-crit-ink:#0d366b;--warn:#3a3214;--warn-line:#8a7428}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:20px;margin:0 0 4px} h2{font-size:15px;margin:26px 0 10px} h3{font-size:13.5px;margin:16px 0 6px}
.sub{color:var(--ink2);font-size:12.5px}
.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:4px 16px;margin:10px 0 4px;font-size:13px}
.meta b{color:var(--ink2);font-weight:500}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-top:14px}
.kpi{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.kpi .v{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}.kpi .l{color:var(--ink2);font-size:12px}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:12px}
figure{margin:0;background:#fff;border:1px solid var(--line);border-radius:10px;padding:8px}
figure img{width:100%;height:auto;display:block} figcaption{font-size:12.5px;font-weight:600;color:#222;margin:2px 4px 6px}
.tbl{overflow-x:auto;background:var(--surface);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;table-layout:fixed;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;overflow:hidden;text-overflow:ellipsis}
th{color:var(--muted);font-weight:500;font-size:12px;white-space:nowrap}
.num{text-align:right}.c{text-align:center}.txt{white-space:normal;overflow-wrap:anywhere}
.tier{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600}
.Low{background:var(--t-low);color:var(--t-low-ink)}.Medium{background:var(--t-med);color:var(--t-med-ink)}
.High{background:var(--t-high);color:var(--t-high-ink)}.Critical{background:var(--t-crit);color:var(--t-crit-ink)}
.tag{display:inline-block;border:1px solid var(--line);border-radius:6px;padding:0 6px;margin:1px 3px 1px 0;font-size:11.5px;color:var(--ink2)}
.empty{color:var(--muted);padding:10px}
.appendix{margin-top:30px;border:2px dashed var(--warn-line);background:var(--warn);border-radius:10px;padding:4px 16px 14px}
footer{margin-top:30px;color:var(--ink2);font-size:12.5px;border-top:1px solid var(--line);padding-top:12px}
footer li{margin:2px 0}
@media print{body{background:#fff}.tbl{overflow:visible}}
"""


def _table(headers: list[tuple], rows: list[list[str]], empty="None in this session.") -> str:
    """headers: [(label, width, cls, title)], rows: lists of HTML-safe cell strings."""
    if not rows:
        return f'<div class="tbl"><div class="empty">{empty}</div></div>'
    total = sum(w for _, w, _, _ in headers)
    cols = "".join(f'<col style="width:{w}px">' for _, w, _, _ in headers)
    th = "".join(f'<th class="{c}" title="{html.escape(t)}">{html.escape(h)}</th>' for h, _, c, t in headers)
    body = "".join("<tr>" + "".join(f'<td class="{headers[i][2]}">{v}</td>' for i, v in enumerate(r)) + "</tr>" for r in rows)
    return (f'<div class="tbl"><table style="min-width:{total}px"><colgroup>{cols}</colgroup><thead><tr>{th}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


def _tier(t):
    return f'<span class="tier {t}">{t}</span>' if t else "–"


def render_html(rep: dict) -> str:
    e, who, k = html.escape, rep["who"], rep["kpi"]
    room = rep["room"]
    opts = rep["options"]
    tiles = [("messages", k["messages"], "Messages"), ("active_users", k["active_users"], "Active users"),
             ("hostile", f'{k["hostile"]} <span class="sub">({k["hostile_pct"]}%)</span>', "Hostile messages"),
             ("alerts", k["alerts"], "Alerts"), ("repeat_harassers", k["repeat_harassers"], "Repeat harassers"),
             ("trolls", k["trolls"], "Trolls"), ("burst_attackers", k["burst_attackers"], "Burst attackers"),
             ("explicit_threats", k["explicit_threats"], "Explicit-threat messages"),
             ("users_targeted", k["users_targeted"], "Users targeted")]
    kpis = "".join(f'<div class="kpi"><div class="v" id="kpi-{i}">{v}</div><div class="l">{l}</div></div>' for i, v, l in tiles)
    charts = "".join(f'<figure><figcaption>{e(c["title"])}</figcaption><img alt="{e(c["alt"])}" '
                     f'src="data:image/png;base64,{c["png"]}"></figure>' for c in rep["charts"])
    cat = lambda cs: "".join(f'<span class="tag">{e(CATEGORY_NAMES[c])}</span>' for c in cs) or '<span class="tag">Normal</span>'  # noqa: E731
    offenders = _table(
        [("User", 220, "", ""), ("Hostile msgs", 95, "num", "Hostile messages sent in this session"),
         ("Distinct targets", 105, "num", "Different people this user sent hostile messages to"),
         ("Highest tier", 100, "c", "Highest pair tier, among this user's pairs active in this session, at the end of the session"),
         ("Overall tier", 100, "c", "Overall = highest tier across all of this user's targets (engine user_max_tier)"),
         ("Categories", 240, "", ""),
         ("Change vs previous session", 170, "", "Highest tier at the end of the previous session -> now")],
        [[e(o["name"]), str(o["hostile"]), str(o["targets"]), _tier(o["highest_tier"]), _tier(o["overall_tier"]),
          cat(o["categories"]), e(o["change"])] for o in rep["offenders"]])
    targeted = _table(
        [("User", 240, "", ""), ("Hostile msgs received", 150, "num", ""), ("Distinct senders", 120, "num", ""),
         ("Coordinated alert", 130, "c", "Engine raised a coordinated-harassment alert for this person in this session")],
        [[e(t["name"]), str(t["hostile_received"]), str(t["senders"]), "yes" if t["coordinated"] else "no"]
         for t in rep["targeted"]])
    alerts = _table(
        [("Time", 90, "", ""), ("Kind", 170, "", ""), ("From", 85, "c", ""), ("To", 85, "c", ""), ("Pair / target", 330, "", "")],
        [[_fmt_ts(a["ts"]), e(a["kind"].replace("_", " ")), _tier(a["from"]) if a["from"] else "–", _tier(a["to"]) if a["to"] else "–",
          (f'{e(who(a["sender"]))} → {e(who(a["target"]))}' if a["kind"] != COORDINATED_ALERT
           else f'victim {e(who(a["target"]))} ({a["senders"]} hostile senders)')] for a in rep["alerts"]])
    esc_tab = _table(
        [("Sender", 210, "", ""), ("Target", 210, "", ""), ("Tier before", 95, "c", "Tier at the end of the previous session"),
         ("Tier now", 95, "c", ""), ("Hostile msgs (total)", 130, "num", ""), ("Sessions with hostility", 150, "num", "")],
        [[e(who(p["sender"])), e(who(p["target"])), _tier(p["prev_tier"] or "Low"), _tier(p["tier"]), str(p["hostile_total"]),
          str(p["sessions"])] for p in rep["escalations"]], empty="No pair's tier went up in this session.")
    ev_rows = []
    for p in rep["evidence"]:
        for i, m in enumerate(p["messages"]):
            ev_rows.append([e(who(p["sender"]) + " → " + who(p["target"])) if i == 0 else "", _tier(p["tier"]) if i == 0 else "",
                            f'{m["T"]:.2f}', e(m["text"]) + (' <span class="tag">explicit threat</span>' if m["threat"] else "")])
    evidence = _table([("Pair", 260, "", ""), ("Tier", 90, "c", ""), ("T", 60, "num", "Content threat score"), ("Message", 420, "txt", "")],
                      ev_rows, empty="No pair reached High or Critical in this session.")
    truth = ""
    if rep["truth"]:
        t = rep["truth"]
        roles = sorted(t["role_by_category"])
        rbc = _table([("Role (ground truth)", 180, "", "")] + [(CATEGORY_NAMES[c], 115, "num", "") for c in CATEGORIES],
                     [[e(r)] + [str(t["role_by_category"][r].get(c, 0)) for c in CATEGORIES] for r in roles])
        cf = t["confusion"]
        conf = _table([("", 220, "", ""), ("Engine: repeat-harasser pair", 200, "num", ""), ("Engine: not flagged", 160, "num", "")],
                      [["Label 1 (repeat harassment)", str(cf.get("11", 0)), str(cf.get("10", 0))],
                       ["Label 0", str(cf.get("01", 0)), str(cf.get("00", 0))]])
        truth = (f'<section class="appendix"><h2>Researcher appendix: engine vs ground truth</h2><p class="sub">This section '
                 f'uses role assignments and pair labels, which a streamer would not have. Everything above it is built from '
                 f'engine outputs only.</p><h3>Senders by assigned role and engine category (this session)</h3>{rbc}'
                 f'<h3>Pairs active in this session: label vs engine</h3>{conf}</section>')
    defs = "".join(f"<li><b>{e(CATEGORY_NAMES[c])}</b>: {e(DEFINITIONS[c].format(p=rep['thresholds']['threat_override_p']))}</li>"
                   for c in ["repeat_harasser", "explicit_threat", "burst_attacker", "troll", "coordinated_target", "normal"])
    flags = ", ".join(n for n, on in [("message text hidden", opts["mask_text"]), ("study codes only", opts["codes_only"]),
                                      ("ground-truth appendix included", opts["include_truth"])] if on) or "full view"
    prev = rep["prev_session"] or "none (first session in this room)"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Session Report {e(rep['session_id'])}</title><style>{CSS}</style></head><body><main>
<h1>Session Report: {e(room['name'])} · {e(rep['session_id'])}</h1>
<div class="sub">SafeChat Lab · Behavioral Analytics Engine (J26-CS-337, IT23212640) · {e(flags)}</div>
<div class="meta"><div><b>Room</b> {e(room['name'])} ({e(room['room_id'])})</div><div><b>Session</b> {e(rep['session_id'])}</div>
<div><b>Start</b> {_fmt_ts(rep['started_ts'], True)}</div><div><b>End</b> {_fmt_ts(rep['ended_ts'], True)}</div>
<div><b>Duration</b> {_duration(rep['started_ts'], rep['ended_ts'])}</div><div><b>Previous session</b> {e(prev)}</div>
<div><b>Content model</b> {e(rep['content_model'])}</div><div><b>Generated</b> {_fmt_ts(rep['generated_ts'], True)}</div></div>
<div class="kpis">{kpis}</div>
<h2>Charts</h2><div class="charts">{charts}</div>
<h2>Escalation since the previous session</h2>{esc_tab}
<h2>Top offenders</h2>{offenders}
<h2>Most-targeted users</h2>{targeted}
<h2>Alerts in this session</h2>{alerts}
<h2>Evidence: most hostile messages of each High / Critical pair</h2>{evidence}
{truth}
<footer><p><b>Privacy.</b> The engine's own store is pseudonymised (HMAC-SHA256) and holds no message text. This report is
generated from the SafeChat research log. Exported copies hide message text by default; use study codes only for anything
shared outside the research team.</p>
<p><b>T vs tier.</b> T is the content model's threat score for one message (0–1; hostile when T ≥ {rep['thresholds']['hostile']}).
The tier (Low / Medium / High / Critical) belongs to a sender → target pair and comes from the engine's risk model over
time: repeated hostility across sessions raises it; a single explicit threat holds it at High for 24 hours.</p>
<p><b>Categories</b> (computed only from engine outputs; a user can be in several):</p><ul>{defs}</ul></footer>
</main></body></html>"""


# ---------------------------------------------------------------- PDF (matplotlib only)
def render_pdf(rep: dict) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    who, k = rep["who"], rep["kpi"]
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        y = 0.95
        def line(text, size=10, bold=False, dy=0.025):  # noqa: E306
            nonlocal y
            fig.text(0.07, y, text, fontsize=size, weight="bold" if bold else "normal", va="top", wrap=True)
            y -= dy
        line(f"Session Report: {rep['room']['name']} · {rep['session_id']}", 15, True, 0.04)
        for t in [f"Room {rep['room']['name']} ({rep['room']['room_id']})  ·  previous session: {rep['prev_session'] or 'none'}",
                  f"Start {_fmt_ts(rep['started_ts'], True)}  ·  end {_fmt_ts(rep['ended_ts'], True)}  ·  duration "
                  f"{_duration(rep['started_ts'], rep['ended_ts'])}",
                  f"Content model {rep['content_model']}  ·  generated {_fmt_ts(rep['generated_ts'], True)}"]:
            line(t, 9.5)
        y -= 0.015
        line("Key numbers", 12, True, 0.03)
        for lab, v in [("Messages", k["messages"]), ("Active users", k["active_users"]),
                       ("Hostile messages", f"{k['hostile']} ({k['hostile_pct']}%)"), ("Alerts", k["alerts"]),
                       ("Repeat harassers", k["repeat_harassers"]), ("Trolls", k["trolls"]),
                       ("Burst attackers", k["burst_attackers"]), ("Explicit-threat messages", k["explicit_threats"]),
                       ("Users targeted", k["users_targeted"])]:
            line(f"{lab}: {v}", 10, False, 0.022)
        y -= 0.02
        line("Definitions", 12, True, 0.03)
        for c in ["repeat_harasser", "explicit_threat", "burst_attacker", "troll", "coordinated_target", "normal"]:
            line(f"{CATEGORY_NAMES[c]}: {DEFINITIONS[c].format(p=rep['thresholds']['threat_override_p'])}", 8.5, False, 0.034)
        line("Privacy: the engine store is pseudonymised and holds no message text. Built only from engine outputs.", 8.5)
        pdf.savefig(fig)
        plt.close(fig)
        import matplotlib.image as mpimg
        for i in range(0, len(rep["charts"]), 2):  # two charts per A4 page
            fig = plt.figure(figsize=(8.27, 11.69))
            for j, c in enumerate(rep["charts"][i:i + 2]):
                ax = fig.add_axes([0.06, 0.53 - j * 0.48, 0.88, 0.40])
                ax.imshow(mpimg.imread(io.BytesIO(base64.b64decode(c["png"])), format="png"))
                ax.axis("off")
                fig.text(0.06, 0.95 - j * 0.48, c["title"], fontsize=11, weight="bold")
            pdf.savefig(fig)
            plt.close(fig)
        tables = [
            ("Escalation since the previous session", ["Sender", "Target", "Before", "Now", "Hostile", "Sessions"],
             [[who(p["sender"]), who(p["target"]), p["prev_tier"] or "Low", p["tier"], p["hostile_total"], p["sessions"]]
              for p in rep["escalations"]]),
            ("Top offenders", ["User", "Hostile", "Targets", "Highest tier", "Overall tier", "Categories", "Change"],
             [[o["name"], o["hostile"], o["targets"], o["highest_tier"], o["overall_tier"], ", ".join(CATEGORY_NAMES[c] for c in o["categories"]) or "Normal",
               o["change"]] for o in rep["offenders"]]),
            ("Most-targeted users", ["User", "Hostile received", "Senders", "Coordinated alert"],
             [[t["name"], t["hostile_received"], t["senders"], "yes" if t["coordinated"] else "no"] for t in rep["targeted"]]),
            ("Alerts in this session", ["Time", "Kind", "From", "To", "Pair / target"],
             [[_fmt_ts(a["ts"]), a["kind"], a["from"] or "-", a["to"] or "-",
               f"{who(a['sender'])} -> {who(a['target'])}" if a["kind"] != COORDINATED_ALERT else f"victim {who(a['target'])}"]
              for a in rep["alerts"]]),
            ("Evidence (High / Critical pairs)", ["Pair", "Tier", "T", "Message"],
             [[f"{who(p['sender'])} -> {who(p['target'])}", p["tier"], f"{m['T']:.2f}", m["text"][:90]]
              for p in rep["evidence"] for m in p["messages"]]),
        ]
        if rep["truth"]:
            t = rep["truth"]
            tables.append(("Researcher appendix: role vs engine category", ["Role"] + [CATEGORY_NAMES[c] for c in CATEGORIES],
                           [[r] + [t["role_by_category"][r].get(c, 0) for c in CATEGORIES] for r in sorted(t["role_by_category"])]))
            cf = t["confusion"]
            tables.append(("Researcher appendix: label vs engine (pairs)", ["", "Engine flagged", "Not flagged"],
                           [["Label 1", cf.get("11", 0), cf.get("10", 0)], ["Label 0", cf.get("01", 0), cf.get("00", 0)]]))
        for title, head, rows in tables:
            for start in range(0, max(1, len(rows)), 28):
                chunk = rows[start:start + 28]
                fig, ax = plt.subplots(figsize=(8.27, 11.69))
                ax.axis("off")
                ax.set_title(title + (" (cont.)" if start else ""), loc="left", fontsize=12, weight="bold")
                if chunk:
                    tb = ax.table(cellText=[[str(v) for v in r] for r in chunk], colLabels=head, loc="upper left", cellLoc="left")
                    tb.auto_set_font_size(False)
                    tb.set_fontsize(7.5)
                    tb.auto_set_column_width(list(range(len(head))))
                    tb.scale(1, 1.35)
                else:
                    ax.text(0, 0.95, "None in this session.", fontsize=10, transform=ax.transAxes)
                pdf.savefig(fig)
                plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------- CSV zip
def render_csv_zip(rep: dict) -> bytes:
    codes_only = rep["options"]["codes_only"]
    who = lambda c: rep["names"].get(c, "")  # noqa: E731  (name column only; codes have their own column)
    users = sorted({r["sender"] for r in rep["rows"]} | {r["target"] for r in rep["rows"]})
    off = {o["code"]: o for o in rep["offenders"]}
    tgt = {t["code"]: t for t in rep["targeted"]}
    sent = Counter(r["sender"] for r in rep["rows"])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        def write(name, cols, rows):
            s = io.StringIO()
            w = csv.writer(s)
            w.writerow(cols)
            w.writerows(rows)
            z.writestr(name, s.getvalue())
        ucols = ["code"] + ([] if codes_only else ["name"]) + ["messages_sent", "hostile_sent", "distinct_targets",
                                                                "highest_tier", "previous_highest_tier", "hostile_received",
                                                                "distinct_hostile_senders", "coordinated_alert", "categories",
                                                                "overall_tier"]
        write("users.csv", ucols, [[u] + ([] if codes_only else [who(u)]) +
                                   [sent.get(u, 0), off.get(u, {}).get("hostile", 0), off.get(u, {}).get("targets", 0),
                                    off.get(u, {}).get("highest_tier", ""), off.get(u, {}).get("prev_tier") or "",
                                    tgt.get(u, {}).get("hostile_received", 0), tgt.get(u, {}).get("senders", 0),
                                    int(tgt.get(u, {}).get("coordinated", False)),
                                    ";".join(rep["user_categories"].get(u, [])),
                                    off.get(u, {}).get("overall_tier", "")] for u in users])
        write("pairs.csv", ["sender", "target", "messages_in_session", "hostile_in_session", "tier_end", "tier_previous_session",
                            "risk_end", "hostile_messages_total", "sessions_targeted", "max_T", "explicit_threat"],
              [[p["sender"], p["target"], p["messages"], p["hostile"], p["tier"], p["prev_tier"] or "", round(p["risk"], 4),
                p["hostile_total"], p["sessions"], round(p["max_T"], 4), int(p["explicit_threat"])] for p in rep["pairs"]])
        write("alerts.csv", ["iso_time", "kind", "from_tier", "to_tier", "sender", "target", "distinct_senders"],
              [[datetime.fromtimestamp(a["ts"]).isoformat(timespec="seconds"), a["kind"], a["from"] or "", a["to"] or "",
                a["sender"], a["target"], a["senders"] or ""] for a in rep["alerts"]])
        write("categories.csv", ["category", "code"] + ([] if codes_only else ["name"]),
              [[c, u] + ([] if codes_only else [who(u)]) for c in CATEGORIES + ["coordinated_target"]
               for u in rep["categories"][c]])
        z.writestr("README.txt", f"Session report data for {rep['session_id']} (built only from engine outputs).\n"
                                 "users.csv, pairs.csv, alerts.csv, categories.csv. No message text is included.\n")
    return buf.getvalue()
