"""Stages B2, D, E, F - escalation, Target Persistence Index, composite risk, tiers.

Per (offender, target) pair the engine keeps exponentially DECAYED counters:

    r_eff  <- r_eff * exp(-lambda*dt) + 1          on every hostile message
    s_eff  <- s_eff * exp(-lambda*dt) + 1          on the first hostile message of a new session

Decaying the counts themselves (not only adding a separate recency term) means an old,
resolved incident fades back to Low instead of holding a user at an elevated tier forever.

    TPI  = w1*log(1 + r_eff) + w2*s_eff + w3*exp(-lambda * dt_since_last_hostile)
    v    = max(0, slope of T over the last k messages to the target)      (escalation)
    Risk = sigmoid(b + alpha*T_pair + beta*TPI + gamma*v)

alpha, beta*w1, beta*w2, beta*w3, gamma and b are fitted by logistic regression on the
synthetic multi-session corpus (scripts/fit_risk_model.py), so beta and w are reported
results rather than guesses.
"""
from __future__ import annotations

import math

import numpy as np

from .config import EngineConfig
from .store import PairState

TIERS = ["Low", "Medium", "High", "Critical"]
TIER_RANK = {t: i for i, t in enumerate(TIERS)}
FEATURES = ["T_pair", "log_r", "s", "recency", "v"]


def _decay(cfg: EngineConfig, dt: float) -> float:
    if cfg.half_life_days <= 0 or not math.isfinite(cfg.half_life_days):
        return 1.0
    return math.exp(-cfg.decay_lambda * max(dt, 0.0))


def escalation_slope(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    y = np.asarray(values, dtype=float)
    x = np.arange(len(y), dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(max(0.0, slope))


def update_pair(st: PairState, T: float, ts: float, session_id: str, cfg: EngineConfig) -> bool:
    """Fold one message into the pair state. Returns True if the message was hostile."""
    if st.last_ts is not None:
        f = _decay(cfg, ts - st.last_ts)
        st.r_eff *= f
        st.s_eff *= f
        st.t_ewma *= f
    else:
        st.first_ts = ts
    st.last_ts = ts
    st.r_total += 1
    st.t_ewma = cfg.ewma_alpha * T + (1 - cfg.ewma_alpha) * st.t_ewma
    st.t_max = max(st.t_max, T)
    st.recent_t = (st.recent_t + [round(T, 4)])[-cfg.escalation_window:]
    hostile = T >= cfg.hostile_threshold
    if hostile:
        st.r_hostile += 1
        st.r_eff += 1.0
        st.last_hostile_ts = ts
        if st.last_session != session_id:
            st.s_hostile += 1
            st.s_eff += 1.0
            st.last_session = session_id
    return hostile


def pair_features(st: PairState, now: float, cfg: EngineConfig) -> dict:
    f = _decay(cfg, now - st.last_ts) if st.last_ts is not None else 0.0
    recency = _decay(cfg, now - st.last_hostile_ts) if st.last_hostile_ts is not None else 0.0
    return {
        "T_pair": st.t_ewma * f,
        "log_r": math.log1p(st.r_eff * f),
        "s": st.s_eff * f,
        "recency": recency,
        "v": escalation_slope(st.recent_t) * f,
    }


def tpi(feat: dict, cfg: EngineConfig) -> float:
    """Target Persistence Index in the fitted weights: beta*TPI = c_logr*log_r + c_s*s + c_rec*recency.
    Reported as the un-scaled weighted sum (so it is comparable across pairs)."""
    return cfg.coef_logr * feat["log_r"] + cfg.coef_s * feat["s"] + cfg.coef_recency * feat["recency"]


def risk_probability(feat: dict, cfg: EngineConfig) -> float:
    z = (cfg.intercept + cfg.coef_T * feat["T_pair"] + tpi(feat, cfg) + cfg.coef_v * feat["v"])
    return 1.0 / (1.0 + math.exp(-max(min(z, 50), -50)))


def tier_for(p: float, cfg: EngineConfig) -> str:
    if p >= cfg.tier_critical:
        return "Critical"
    if p >= cfg.tier_high:
        return "High"
    if p >= cfg.tier_medium:
        return "Medium"
    return "Low"


def max_tier(a: str, b: str) -> str:
    return a if TIER_RANK[a] >= TIER_RANK[b] else b
