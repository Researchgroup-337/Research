"""The Behavioral Analytics Engine - one call per chat message.

    engine = BehavioralEngine(db_path="data/bae.db")
    record = engine.process({...chat event...})

Pipeline per message (stage letters follow the proposal, Fig. 2):
  A  validate event, resolve target, pseudonymise both identities
  B1 content threat score T (DistilBERT or TF-IDF) + peer signals -> effective T
  B2 escalation velocity (inside update_pair)
  C  load / update / persist (offender, target) state - no message text stored
  D  Target Persistence Index
  E  composite risk probability
  F  Low / Medium / High / Critical tier (+ explicit-threat safety override)
  G  structured risk record + alerts on tier escalation / coordinated attack
"""
from __future__ import annotations

import time
from datetime import datetime

from .config import LABELS, EngineConfig
from .content_model import ContentScorer, load_scorer
from .privacy import Pseudonymizer
from .risk import (TIER_RANK, max_tier, pair_features, risk_probability, tier_for, tpi, update_pair)
from .schemas import ChatEvent
from .store import BehaviorStore
from .target_resolver import TargetResolver

URL_T = {"malicious": 0.75, "suspicious": 0.45, "safe": 0.0, "unknown": 0.0}


def _ts(v) -> float:
    if isinstance(v, datetime):
        return v.timestamp()
    if isinstance(v, str):
        return datetime.fromisoformat(v.replace("Z", "+00:00")).timestamp()
    return float(v)


class BehavioralEngine:
    def __init__(self, cfg: EngineConfig | None = None, scorer: ContentScorer | str | None = "auto",
                 store: BehaviorStore | None = None, db_path: str = ":memory:",
                 encryption_key: str | None = None, pseudonymizer: Pseudonymizer | None = None,
                 autocommit: bool = True):
        self.cfg = cfg or EngineConfig.load()
        self.scorer = load_scorer(scorer) if isinstance(scorer, str) else scorer
        self.store = store or BehaviorStore(db_path, encryption_key)
        self.pseudo = pseudonymizer or Pseudonymizer()
        self.resolver = TargetResolver(self.cfg.broadcast_target)
        self.autocommit = autocommit

    # ------------------------------------------------------------------ main entry
    def process(self, event: dict | ChatEvent) -> dict:
        t0 = time.perf_counter()
        ev = event if isinstance(event, ChatEvent) else ChatEvent(**event)
        ts = _ts(ev.timestamp)

        # Stage A - target resolution happens on raw names, then both are pseudonymised.
        target_raw, method, conf = self.resolver.resolve(ev.text, ev.sender_id, ev.session_id,
                                                         ev.reply_to, ev.mentions, ev.channel_owner)
        offender = self.pseudo(ev.sender_id, ev.platform)
        target = self.pseudo(target_raw, ev.platform)

        # Stage B1 - content score.
        if ev.content_scores is not None:
            scores = {"T": float(ev.content_scores.get("T", 0.0)), **{l: float(ev.content_scores.get(l, 0.0)) for l in LABELS}}
        elif self.scorer is not None:
            scores = self.scorer.score_one(ev.text)
        else:
            raise RuntimeError("no content scorer configured")
        T = scores["T"]
        flags = []
        eff_T = T
        ps = ev.peer_signals
        if ps:
            if ps.hate_score is not None:
                eff_T = max(eff_T, ps.hate_score)
                if ps.hate_score >= 0.5:
                    flags.append(f"hate_speech:{ps.hate_type or 'unspecified'}")
            if ps.se_score is not None:
                eff_T = max(eff_T, 0.9 * ps.se_score)
                if ps.se_score >= 0.5:
                    flags.append("social_engineering")
            if ps.url_threat:
                eff_T = max(eff_T, URL_T[ps.url_threat])
                if ps.url_threat in ("malicious", "suspicious"):
                    flags.append(f"url_{ps.url_threat}")
        return self._update(ev.message_id, offender, target, method, conf, ev.session_id, ts, scores, eff_T, flags, t0)

    def process_scored(self, message_id: str, sender: str, target: str, session_id: str, ts: float,
                       T: float, platform: str = "sim", method: str = "given") -> dict:
        """Fast path for experiments: identities and T already known (no text handling)."""
        t0 = time.perf_counter()
        return self._update(message_id, self.pseudo(sender, platform), self.pseudo(target, platform), method,
                            1.0, session_id, ts, {"T": T}, T, [], t0)

    # ------------------------------------------------------------------ core update
    def _update(self, message_id, offender, target, method, conf, session_id, ts, scores, eff_T, flags, t0) -> dict:
        cfg = self.cfg
        st = self.store.get_pair(offender, target)
        prev_tier = st.tier
        hostile = update_pair(st, eff_T, ts, session_id, cfg)
        feat = pair_features(st, ts, cfg)
        p = risk_probability(feat, cfg)
        tier = tier_for(p, cfg)
        # Safety override: an explicit, high-confidence threat holds the pair at >= High for
        # threat_hold_hours (sticky, so the tier does not flicker and re-alert every message).
        if scores.get("threat", 0.0) >= cfg.threat_override_p:
            st.last_threat_ts = ts
        if st.last_threat_ts is not None and ts - st.last_threat_ts <= cfg.threat_hold_hours * 3600:
            tier = max_tier(tier, "High")
            flags.append("explicit_threat_override")
        st.risk, st.tier = p, tier
        st.flags = sorted(set(st.flags) | set(flags))[-10:]

        alerts = []
        if TIER_RANK[tier] > TIER_RANK[prev_tier] and TIER_RANK[tier] >= TIER_RANK["Medium"]:
            detail = {"hostile_messages": st.r_hostile, "sessions": st.s_hostile, "T": round(eff_T, 3),
                      "method": method}
            aid = self.store.add_alert(ts, "tier_escalation", offender, target, prev_tier, tier, p, detail)
            alerts.append({"id": aid, "kind": "tier_escalation", "from": prev_tier, "to": tier})

        pressure = 0
        coordinated = False
        if hostile:
            self.store.add_target_hostile(target, offender, ts)
            pressure = self.store.distinct_hostile_senders(target, ts - cfg.pressure_window_days * 86400)
            if pressure == cfg.pressure_min_senders:  # fire once when the threshold is crossed
                coordinated = True
                aid = self.store.add_alert(ts, "coordinated_harassment", None, target, None, None, None,
                                           {"distinct_hostile_senders": pressure, "window_days": cfg.pressure_window_days})
                alerts.append({"id": aid, "kind": "coordinated_harassment", "senders": pressure})
            coordinated = pressure >= cfg.pressure_min_senders

        if self.autocommit:
            self.store.put_pair(st)
        user_pairs = self.store.pairs_for_offender(offender) if self.autocommit else [st]
        u_tier, u_risk, n_targets = "Low", 0.0, 0
        for ps in user_pairs:
            u_tier = max_tier(u_tier, ps.tier)
            u_risk = max(u_risk, ps.risk)
            n_targets += ps.r_hostile > 0

        return {
            "message_id": message_id, "offender": offender, "target": target,
            "target_method": method, "target_confidence": conf,
            "message": {"T": round(scores["T"], 4), "labels": {l: round(scores[l], 4) for l in LABELS if l in scores},
                        "hostile": hostile, "effective_T": round(eff_T, 4)},
            "pair": {"risk": round(p, 4), "tier": tier, "previous_tier": prev_tier,
                     "tpi": round(tpi(feat, cfg), 4), "escalation_v": round(feat["v"], 4),
                     "t_pair": round(feat["T_pair"], 4), "hostile_messages": st.r_hostile,
                     "sessions_targeted": st.s_hostile, "total_messages": st.r_total,
                     "last_hostile_ts": st.last_hostile_ts},
            "user_max_tier": u_tier, "user_max_risk": round(u_risk, 4), "targets_harassed": n_targets,
            "target_pressure": pressure, "coordinated_attack": coordinated,
            "alerts": alerts, "flags": flags,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
        }

    # ------------------------------------------------------------------ queries
    def snapshot(self, now: float) -> list[dict]:
        """Features + risk for every pair at time `now` (used for evaluation and dashboards)."""
        out = []
        for st in self.store.all_pairs():
            feat = pair_features(st, now, self.cfg)
            out.append({"offender": st.offender, "target": st.target, **feat,
                        "risk": risk_probability(feat, self.cfg), "r_hostile": st.r_hostile,
                        "s_hostile": st.s_hostile, "r_total": st.r_total, "t_max": st.t_max})
        return out

    def user_report(self, user_id: str, platform: str, now: float | None = None) -> dict:
        h = self.pseudo(user_id, platform)
        now = now or time.time()
        pairs = []
        for st in self.store.pairs_for_offender(h):
            feat = pair_features(st, now, self.cfg)
            p = risk_probability(feat, self.cfg)
            pairs.append({"target": st.target, "risk_now": round(p, 4), "tier_now": tier_for(p, self.cfg),
                          "hostile_messages": st.r_hostile, "sessions_targeted": st.s_hostile,
                          "tpi": round(tpi(feat, self.cfg), 4), "flags": st.flags})
        pairs.sort(key=lambda d: -d["risk_now"])
        top = pairs[0]["tier_now"] if pairs else "Low"
        return {"offender": h, "tier": top, "pairs": pairs}

    def delete_user(self, user_id: str, platform: str) -> int:
        return self.store.delete_user(self.pseudo(user_id, platform))

    def purge(self, now: float | None = None) -> int:
        return self.store.purge(now or time.time(), self.cfg.retention_days, self.cfg.pressure_window_days)
