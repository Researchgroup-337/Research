"""Helpers to replay a synthetic corpus through the engine for fitting and evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EngineConfig
from .engine import BehavioralEngine
from .risk import FEATURES, TIER_RANK, pair_features
from .store import BehaviorStore


class IdentityPseudonymizer:
    """Experiments only: keep raw IDs so results can be joined to ground truth."""
    def __call__(self, user_id, platform=""):
        return str(user_id)


def replay(df: pd.DataFrame, cfg: EngineConfig, snapshot_days: list[float] | None = None,
           base_ts: float | None = None, track_stream: bool = True) -> dict:
    """Replay messages in time order. Returns per-snapshot pair features and streaming detections."""
    eng = BehavioralEngine(cfg=cfg, scorer=None, store=BehaviorStore(":memory:"),
                           pseudonymizer=IdentityPseudonymizer(), autocommit=False)
    base_ts = base_ts if base_ts is not None else df.ts.min()
    snap_ts = sorted(base_ts + d * 86400 for d in (snapshot_days or []))
    end_ts = df.ts.max() + 1
    snaps, first_high, first_med, coordinated = {}, {}, {}, {}
    si = 0

    def take(ts):
        rows = []
        for (o, t), st in eng.store._cache.items():
            if st.last_ts is None or st.last_ts > ts:
                continue
            f = pair_features(st, ts, cfg)
            rows.append({"sender": o, "target": t, **f, "r_hostile": st.r_hostile, "s_hostile": st.s_hostile,
                         "r_total": st.r_total, "t_max": st.t_max, "t_mean": st.t_ewma})
        return pd.DataFrame(rows)

    for r in df[["message_id", "sender", "target", "session_id", "ts", "T", "p_threat"]].itertuples(index=False):
        while si < len(snap_ts) and r.ts > snap_ts[si]:
            snaps[snap_ts[si]] = take(snap_ts[si])
            si += 1
        rec = eng._update(r.message_id, r.sender, r.target, "given", 1.0, r.session_id, r.ts,
                          {"T": r.T, "threat": r.p_threat}, r.T, [], 0.0)
        if track_stream:
            key = (r.sender, r.target)
            rank = TIER_RANK[rec["pair"]["tier"]]
            if rank >= TIER_RANK["High"] and key not in first_high:
                first_high[key] = (r.ts, rec["pair"]["sessions_targeted"], rec["pair"]["hostile_messages"])
            if rank >= TIER_RANK["Medium"] and key not in first_med:
                first_med[key] = (r.ts, rec["pair"]["sessions_targeted"], rec["pair"]["hostile_messages"])
            if rec["coordinated_attack"] and r.target not in coordinated:
                coordinated[r.target] = r.ts
    # final snapshot: end of observation window
    snaps[end_ts] = take(end_ts)
    for ts in snap_ts[si:]:
        snaps.setdefault(ts, take(ts))
    return {"snapshots": snaps, "first_high": first_high, "first_medium": first_med,
            "coordinated": coordinated, "end_ts": end_ts, "engine": eng}


def label_snapshot(snap: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    lab = pairs[["sender", "target", "label", "persona"]]
    out = snap.merge(lab, on=["sender", "target"], how="left")
    out["label"] = out["label"].fillna(0).astype(int)
    return out


def X(df: pd.DataFrame, feats=FEATURES) -> np.ndarray:
    return df[list(feats)].values.astype(float)
