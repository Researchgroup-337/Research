"""Step 5 - Full evaluation of the Behavioral Analytics Engine.

  E1  Pair-level detection on the held-out TEST world vs. baselines
        (a) stateless message classifier  - flag a pair if ANY message has T >= 0.5
        (b) frequency rule ("3 strikes")  - flag if >= 3 hostile messages to the target
        (c) proposed engine               - flag if tier >= High
        (d) upper bound                   - gradient boosting on the same features
  E2  Ablations (refit without one term each)
  E3  Per-persona behaviour (which kinds of users each system flags)
  E4  Streaming: time-to-detect for harassers, false-alarm rate for everyone else
  E5  Stress tests: heavy target-resolution noise; account rotation (+ victim pressure)
  E6  Adversarial evasion of the content model (leet, spacing, zero-width, homoglyph...)
  E7  Latency (content model + engine, plain and encrypted SQLite)
  E8  Real-data sanity check on the Toxic Comment Moderation dataset (real user IDs)
Writes reports/evaluation_results.json, reports/EVALUATION_REPORT.md and figures.
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from bae.config import DATA_PROCESSED, LABELS, MODELS, REPORTS, EngineConfig  # noqa: E402
from bae.content_model import load_scorer, threat_signal  # noqa: E402
from bae.engine import BehavioralEngine  # noqa: E402
from bae.experiment import X, label_snapshot, replay  # noqa: E402
from bae.risk import FEATURES, TIERS, tier_for  # noqa: E402
from bae.text_norm import perturb  # noqa: E402
from fit_risk_model import BASE_TS, build_xy, fit_lr, load  # noqa: E402

FIG = REPORTS / "figures"
C = {"engine": "#2a78d6", "stateless": "#eb6834", "frequency": "#1baf7a", "upper": "#6250d6",
     "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
     "surface": "#fcfcfb"}
TIER_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def metrics(y, score, flag) -> dict:
    return {"roc_auc": round(roc_auc_score(y, score), 4), "pr_auc": round(average_precision_score(y, score), 4),
            "precision": round(precision_score(y, flag, zero_division=0), 4),
            "recall": round(recall_score(y, flag, zero_division=0), 4),
            "f1": round(f1_score(y, flag, zero_division=0), 4), "flagged": int(np.sum(flag))}


def lr_prob(cfg: EngineConfig, df: pd.DataFrame) -> np.ndarray:
    z = (cfg.intercept + cfg.coef_T * df.T_pair + cfg.coef_logr * df.log_r + cfg.coef_s * df.s
         + cfg.coef_recency * df.recency + cfg.coef_v * df.v)
    return 1 / (1 + np.exp(-z.values))


def end_snapshot(name: str, cfg: EngineConfig, stream=True):
    df, pairs = load(name)
    rep = replay(df, cfg, [], base_ts=BASE_TS, track_stream=stream)
    snap = label_snapshot(rep["snapshots"][rep["end_ts"]], pairs)
    return df, pairs, snap, rep


def system_scores(snap: pd.DataFrame, cfg: EngineConfig) -> dict:
    p = lr_prob(cfg, snap)
    return {
        "stateless": (snap.t_max.values, snap.t_max.values >= 0.5),
        "frequency": (snap.r_hostile.values, snap.r_hostile.values >= 3),
        "engine": (p, p >= cfg.tier_high),
    }


# ---------------------------------------------------------------------------- plotting
def _style(ax):
    import matplotlib.pyplot as plt  # noqa
    ax.set_facecolor(C["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["axis"])
    ax.tick_params(colors=C["ink2"], labelsize=9)
    ax.grid(True, color=C["grid"], linewidth=0.6)
    ax.set_axisbelow(True)


def plot_roc(snap, cfg, path):
    """Precision-recall curves (more informative than ROC at a 2% positive rate) + operating points."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    fig, ax = plt.subplots(figsize=(6.2, 4.6), facecolor=C["surface"])
    _style(ax)
    names = {"engine": "Proposed engine (TPI + risk)", "stateless": "Stateless classifier (max T)",
             "frequency": "Frequency count (hostile msgs)"}
    for k, (score, flag) in system_scores(snap, cfg).items():
        prec, rec, _ = precision_recall_curve(snap.label, score)
        ax.plot(rec, prec, color=C[k], lw=2, label=f"{names[k]}  PR-AUC {average_precision_score(snap.label, score):.3f}")
        op_p, op_r = precision_score(snap.label, flag, zero_division=0), recall_score(snap.label, flag)
        ax.plot([op_r], [op_p], marker="o", ms=8, color=C[k], mec=C["surface"], mew=2)
    ax.set_xlabel("Recall (harasser pairs found)", color=C["ink2"])
    ax.set_ylabel("Precision (flags that are real)", color=C["ink2"])
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_title("Repeat-harassment pairs, held-out test world (dots = operating points)", color=C["ink"], fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", labelcolor=C["ink"])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_hbar(values: dict, title: str, xlabel: str, path, highlight=None, xlim=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(values)
    fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(names) + 1.2), facecolor=C["surface"])
    _style(ax)
    ax.grid(axis="y", visible=False)
    cols = [C["engine"] if (highlight and n in highlight) else "#9ec5f4" for n in names]
    ax.barh(names, [values[n] for n in names], color=cols, height=0.6)
    for i, n in enumerate(names):
        ax.text(values[n] + 0.005, i, f"{values[n]:.3f}", va="center", fontsize=8.5, color=C["ink"])
    ax.invert_yaxis()
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, color=C["ink2"])
    ax.set_title(title, color=C["ink"], fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_evasion(out, model, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds = ["none", "leet", "spaced", "zero_width", "homoglyph", "stretch", "mixed"]
    fig, ax = plt.subplots(figsize=(7.4, 3.8), facecolor=C["surface"])
    _style(ax)
    ax.grid(axis="x", visible=False)
    x = np.arange(len(kinds))
    w = 0.38
    a = [out[k]["recall_without_normalisation"] for k in kinds]
    b = [out[k]["recall_with_normalisation"] for k in kinds]
    ax.bar(x - w / 2, a, w - 0.03, color="#9ec5f4", label="Without normalisation")
    ax.bar(x + w / 2, b, w - 0.03, color=C["engine"], label="With adversarial normalisation")
    for i in range(len(kinds)):
        ax.text(x[i] + w / 2, b[i] + 0.01, f"{b[i]:.2f}", ha="center", fontsize=8, color=C["ink"])
        ax.text(x[i] - w / 2, a[i] + 0.01, f"{a[i]:.2f}", ha="center", fontsize=8, color=C["ink2"])
    ax.set_xticks(x)
    ax.set_xticklabels(["no attack", "leetspeak", "spaced", "zero-width", "homoglyph", "stretch", "mixed"], fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall on abusive messages", color=C["ink2"])
    ax.set_title(f"Evasion attacks on the content model ({model})", color=C["ink"], fontsize=11, loc="left", pad=26)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), labelcolor=C["ink"])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_persona(tab: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.4, 4.4), facecolor=C["surface"])
    _style(ax)
    ax.grid(axis="x", visible=False)
    x = np.arange(len(tab))
    w = 0.26
    for i, k in enumerate(["stateless", "frequency", "engine"]):
        ax.bar(x + (i - 1) * w, tab[k].values, width=w - 0.02, color=C[k],
               label={"stateless": "Stateless", "frequency": "Frequency", "engine": "Proposed engine"}[k])
    ax.set_xticks(x)
    ax.set_xticklabels(tab.index, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("Share of pairs flagged", color=C["ink2"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Who gets flagged? Share of pairs flagged at the end of the window", color=C["ink"], fontsize=11, loc="left", pad=26)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), labelcolor=C["ink"])
    ax.axvline(5.5, color=C["axis"], lw=1, ls="--")
    ax.text(5.6, 1.0, "true harassers (victim pair) →", fontsize=8, color=C["ink2"], va="bottom")
    ax.text(5.4, 1.0, "← should NOT be flagged", fontsize=8, color=C["ink2"], va="bottom", ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_timeline(df, pairs, cfg, path):
    """Risk over time for one escalator / persistent pair vs a burst attacker."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pos = pairs[(pairs.label == 1) & (pairs.persona == "escalator")].sort_values("n", ascending=False)
    if pos.empty:
        pos = pairs[pairs.label == 1].sort_values("n", ascending=False)
    burst = pairs[(pairs.persona == "burst") & (pairs.label == 0)].sort_values("n", ascending=False)
    picks = [(pos.iloc[0], "Repeat harasser (" + pos.iloc[0].persona + ")", C["engine"]),
             (burst.iloc[0], "Burst attacker (one session)", C["stateless"])]
    fig, ax = plt.subplots(figsize=(7.4, 3.8), facecolor=C["surface"])
    _style(ax)
    for row, lab, col in picks:
        sub = df[(df.sender == row.sender) & (df.target == row.target)]
        eng = BehavioralEngine(cfg=cfg, scorer=None, autocommit=False,
                               pseudonymizer=__import__("bae.experiment", fromlist=["x"]).IdentityPseudonymizer())
        xs, ys = [], []
        for r in sub.itertuples():
            rec = eng._update(r.message_id, r.sender, r.target, "given", 1, r.session_id, r.ts,
                              {"T": r.T, "threat": r.p_threat}, r.T, [], 0)
            xs.append((r.ts - BASE_TS) / 86400)
            ys.append(rec["pair"]["risk"])
        ax.plot(xs, ys, color=col, lw=2, marker="o", ms=4, label=lab)
    for thr, name in [(cfg.tier_medium, "Medium"), (cfg.tier_high, "High"), (cfg.tier_critical, "Critical")]:
        ax.axhline(thr, color=C["muted"], lw=0.8, ls=":")
        ax.text(64, thr + 0.01, name, fontsize=8, color=C["ink2"], va="bottom", ha="right")
    ax.set_xlim(0, 64)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Day of observation window", color=C["ink2"])
    ax.set_ylabel("Pair risk", color=C["ink2"])
    ax.set_title("Persistence raises risk; a one-session burst decays", color=C["ink"], fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=C["ink"])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------- experiments
def e1_e3(cfg, res):
    tr_df, tr_pairs = load("train")
    train = build_xy(tr_df, tr_pairs, cfg, base_ts=BASE_TS)
    df, pairs, snap, rep = end_snapshot("test", cfg)
    out = {}
    for k, (score, flag) in system_scores(snap, cfg).items():
        out[k] = metrics(snap.label, score, flag)
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    gb.fit(X(train), train.label)
    pg = gb.predict_proba(X(snap))[:, 1]
    out["upper_bound_gbm"] = metrics(snap.label, pg, pg >= 0.5)
    tiers = [tier_for(p, cfg) for p in lr_prob(cfg, snap)]
    snap["tier"] = tiers
    res["E1_pair_detection"] = {"n_pairs": len(snap), "positives": int(snap.label.sum()), "systems": out,
                                "tier_by_label": pd.crosstab(snap.label, snap.tier).reindex(columns=TIERS, fill_value=0).to_dict()}
    plot_roc(snap, cfg, FIG / "pr_curve_test_world.png")

    # E2 ablations: refit LR on train with features removed / no decay
    abl = {}
    full = fit_lr(train)
    abl["Full model"] = average_precision_score(snap.label, full.predict_proba(X(snap))[:, 1])
    for name, drop in [("without session recurrence s", ["s"]), ("without escalation v", ["v"]),
                       ("without recency term", ["recency"]), ("without volume log(1+r)", ["log_r"]),
                       ("without content T (TPI only)", ["T_pair"]),
                       ("content T only (no TPI)", ["log_r", "s", "recency", "v"])]:
        feats = [f for f in FEATURES if f not in drop]
        m = fit_lr(train, feats)
        abl[name] = average_precision_score(snap.label, m.predict_proba(X(snap, feats))[:, 1])
    cfg_nd = EngineConfig(**{**cfg.__dict__, "half_life_days": 0.0})
    tr_nd = build_xy(tr_df, tr_pairs, cfg_nd, base_ts=BASE_TS)
    _, _, snap_nd, _ = end_snapshot("test", cfg_nd, stream=False)
    m = fit_lr(tr_nd)
    abl["without decay (lambda = 0)"] = average_precision_score(snap_nd.label, m.predict_proba(X(snap_nd))[:, 1])
    abl = {k: round(float(v), 4) for k, v in abl.items()}
    res["E2_ablation_pr_auc"] = abl
    plot_hbar(abl, "Ablation: PR-AUC on held-out test world", "PR-AUC (higher is better)",
              FIG / "ablation_pr_auc.png", highlight={"Full model"}, xlim=(0, 1.08))

    # E3 per persona (pairs with >= 1 hostile message)
    pos_personas = {"persistent", "persistent_subtle", "escalator"}
    h = snap[(snap.r_hostile > 0) & ((snap.label == 1) | ~snap.persona.isin(pos_personas))].copy()
    sc = system_scores(h, cfg)
    for k, (_, flag) in sc.items():
        h[k] = flag.astype(float)
    tab = h.groupby("persona")[["stateless", "frequency", "engine"]].mean()
    tab["n_pairs"] = h.groupby("persona").size()
    order = ["normal", "banter", "one_off", "burst", "roaming_troll", "reformed", "persistent", "persistent_subtle", "escalator"]
    tab = tab.reindex([o for o in order if o in tab.index])
    # positives only for harasser personas (their victim pair)
    pos_tab = snap[snap.label == 1].copy()
    for k, (_, flag) in system_scores(pos_tab, cfg).items():
        pos_tab[k] = flag.astype(float)
    res["E3_flag_rate_by_persona_hostile_pairs"] = tab.round(3).to_dict(orient="index")
    res["E3_recall_by_harasser_type"] = pos_tab.groupby("persona")[["stateless", "frequency", "engine"]].mean().round(3).to_dict(orient="index")
    plot_persona(tab[["stateless", "frequency", "engine"]], FIG / "flag_rate_by_persona.png")

    # E4 streaming detection
    s = streaming(df, pairs, rep, cfg)
    # same engine with the explicit-threat override disabled (threshold above any probability)
    cfg_no = EngineConfig(**{**cfg.__dict__, "threat_override_p": 1.01})
    rep_no = replay(df, cfg_no, [], base_ts=BASE_TS)
    s["engine_no_override"] = streaming(df, pairs, rep_no, cfg_no)["engine"]
    s["override_only_false_alarms"] = override_only(df, pairs, rep, rep_no, cfg)
    res["E4_streaming"] = s
    plot_timeline(df, pairs, cfg, FIG / "risk_timeline_example.png")
    return snap


def streaming(df, pairs, rep, cfg):
    """When does each system first raise an alarm for a pair?"""
    pos = set(map(tuple, pairs[pairs.label == 1][["sender", "target"]].values))
    allpairs = set(map(tuple, pairs[["sender", "target"]].values))
    neg = allpairs - pos
    g = df.sort_values("ts")
    first_stateless, first_freq, hostile_count = {}, {}, {}
    sess_seen: dict = {}
    sess_at: dict = {}
    for r in g[["sender", "target", "T", "ts", "session_id"]].itertuples(index=False):
        key = (r.sender, r.target)
        if r.T >= 0.5:
            hostile_count[key] = hostile_count.get(key, 0) + 1
            if key not in sess_seen or sess_seen[key][-1] != r.session_id:
                sess_seen.setdefault(key, []).append(r.session_id)
            first_stateless.setdefault(key, (r.ts, len(sess_seen[key])))
            if hostile_count[key] >= 3:
                first_freq.setdefault(key, (r.ts, len(sess_seen[key])))
    eng = {k: (v[0], v[1]) for k, v in rep["first_high"].items()}

    def summarise(first):
        det = [first[k][1] for k in pos if k in first]
        fa = sum(1 for k in neg if k in first)
        return {"harasser_pairs_detected": round(len(det) / len(pos), 4),
                "median_sessions_to_detect": float(np.median(det)) if det else None,
                "false_alarm_pairs": fa, "false_alarm_rate": round(fa / len(neg), 5),
                "alarms_total": sum(1 for k in allpairs if k in first)}
    hostile_sessions = {k: (v[0], v[1]) for k, v in rep["first_medium"].items()}
    return {"positives": len(pos), "negatives": len(neg), "stateless": summarise(first_stateless),
            "frequency": summarise(first_freq), "engine": summarise(eng),
            "engine_medium_or_above": summarise(hostile_sessions)}


def override_only(df, pairs, rep, rep_no, cfg):
    """Non-harasser pairs that reach High ONLY because of the explicit-threat override, split by
    whether the triggering text carries a Jigsaw threat label or is a content-model false positive."""
    pos = set(map(tuple, pairs[pairs.label == 1][["sender", "target"]].values))
    neg = set(map(tuple, pairs[["sender", "target"]].values)) - pos
    only = [k for k in neg if k in rep["first_high"] and k not in rep_no["first_high"]]
    trig = df[df.p_threat >= cfg.threat_override_p]
    real = set(map(tuple, trig[trig.bucket == "threat"][["sender", "target"]].values))
    n_real = sum(1 for k in only if k in real)
    return {"false_alarm_pairs": len(only), "false_alarm_rate_from_override": round(len(only) / max(1, len(neg)), 5),
            "triggered_by_labelled_threat": n_real, "triggered_by_non_threat_text": len(only) - n_real,
            "threat_override_p": cfg.threat_override_p}


def e5(cfg, res):
    out = {}
    for name in ["test_noisy", "test_rotation"]:
        df, pairs, snap, rep = end_snapshot(name, cfg)
        out[name] = {k: metrics(snap.label, sc, fl) for k, (sc, fl) in system_scores(snap, cfg).items()}
        if name == "test_rotation":
            # victim-centric view: are harassment victims caught by the coordinated-pressure alert?
            victims = set(pairs[pairs.label == 1].target)
            targets = set(df.target)
            alerted = set(rep["coordinated"])
            out[name]["victim_pressure_alert"] = {
                "victims": len(victims), "victims_alerted": round(len(victims & alerted) / max(1, len(victims)), 4),
                "non_victim_targets_alerted": round(len((targets - victims) & alerted) / max(1, len(targets - victims)), 4)}
    res["E5_stress_tests"] = out


def e6(scorer, res, n=1500):
    te = pd.read_parquet(DATA_PROCESSED / "jigsaw_test.parquet")
    rng = np.random.default_rng(0)
    tox = te[te[LABELS].sum(1) > 0]
    thr = tox[tox.threat == 1]
    sample_tox = pd.concat([thr, tox.drop(thr.index).sample(n - len(thr), random_state=0)])
    sample_tox = sample_tox.comment_text.map(lambda t: " ".join(str(t).split()[:40])).tolist()
    out = {}
    for kind in ["none", "leet", "spaced", "zero_width", "homoglyph", "stretch", "mixed"]:
        r = random.Random(1)
        texts = sample_tox if kind == "none" else [perturb(t, kind, r, rate=0.6) for t in sample_tox]
        rec_norm = float((threat_signal(scorer.predict_proba(texts, normalise=True)) >= 0.5).mean())
        rec_raw = float((threat_signal(scorer.predict_proba([t.lower() for t in texts], normalise=False)) >= 0.5).mean())
        out[kind] = {"recall_with_normalisation": round(rec_norm, 4), "recall_without_normalisation": round(rec_raw, 4)}
    res["E6_evasion_recall_on_abusive_messages"] = {"model": scorer.name, "n": len(sample_tox), **out}
    plot_evasion(out, scorer.name, FIG / "evasion_recall.png")


def e7(scorer, cfg, res):
    msgs = ["@victim I know where you live", "gg nice stream", "you are pathetic and nobody likes you",
            "lol that play was insane", "@victim I'll find you after the stream"]
    out = {}
    stores = [("plain_sqlite", None)]
    if importlib.util.find_spec("sqlcipher3") is not None:  # no sqlcipher3 wheel on Windows
        stores.append(("sqlcipher_encrypted", "test-key-123"))
    for label, key in stores:
        with tempfile.TemporaryDirectory() as d:
            eng = BehavioralEngine(cfg=cfg, scorer=scorer, db_path=str(Path(d) / "bae.db"), encryption_key=key)
            t_all = []
            for i in range(600):
                ev = {"message_id": f"m{i}", "platform": "youtube", "session_id": f"s{i // 100}",
                      "channel_owner": "streamer", "sender_id": f"user{i % 37}", "text": msgs[i % len(msgs)],
                      "timestamp": 1_780_000_000 + i * 5}
                t = time.perf_counter()
                eng.process(ev)
                t_all.append((time.perf_counter() - t) * 1000)
            eng.store.close()
        t_all = np.array(t_all[50:])
        out[label] = {"p50_ms": round(float(np.percentile(t_all, 50)), 2), "p95_ms": round(float(np.percentile(t_all, 95)), 2),
                      "max_ms": round(float(t_all.max()), 2)}
    # engine-only overhead (no content model)
    eng = BehavioralEngine(cfg=cfg, scorer=None)
    t = time.perf_counter()
    for i in range(5000):
        eng.process_scored(f"m{i}", f"u{i % 200}", f"t{i % 50}", f"s{i // 500}", 1_780_000_000 + i, (i % 10) / 10)
    out["engine_only_ms_per_message"] = round((time.perf_counter() - t) / 5000 * 1000, 3)
    out["content_model"] = scorer.name
    out["target_ms"] = 250
    res["E7_latency"] = out


def e8(scorer, cfg, res):
    tcm = pd.read_parquet(DATA_PROCESSED / "tcm_eval.parquet").sort_values("date_posted")
    eng = BehavioralEngine(cfg=cfg, scorer=scorer, pseudonymizer=__import__("bae.experiment", fromlist=["x"]).IdentityPseudonymizer())
    for r in tcm.itertuples():
        eng.process({"message_id": str(r.comment_id), "platform": r.platform, "session_id": f"{r.platform}_{r.date_posted:%Y-%m}",
                     "channel_owner": f"{r.platform}_page", "sender_id": r.user_id, "text": r.comment_text,
                     "timestamp": r.date_posted.timestamp()})
    users = tcm.groupby("user_id").agg(n=("is_toxic", "size"), toxic=("is_toxic", "sum")).reset_index()
    tiers = []
    end = tcm.date_posted.max().timestamp()
    for u in users.user_id:
        rep = eng.user_report(u, "x", now=end) if False else None  # placeholder (platform differs per row)
        tiers.append(max([p.tier for p in eng.store.pairs_for_offender(u)], key=TIERS.index, default="Low"))
    users["tier"] = tiers
    from scipy.stats import spearmanr
    rho = spearmanr(users.toxic, users.tier.map(TIERS.index)).statistic
    res["E8_real_data_sanity"] = {
        "dataset": "Toxic Comment Moderation (1,050 comments, 308 users)",
        "users_by_tier": users.tier.value_counts().reindex(TIERS, fill_value=0).to_dict(),
        "mean_toxic_comments_by_tier": users.groupby("tier").toxic.mean().reindex(TIERS).round(2).to_dict(),
        "spearman_toxic_count_vs_tier": round(float(rho), 4),
        "note": "No victim/target field in this dataset, so all comments resolve to the page owner; "
                "this only checks that tiers order real users sensibly, not the TPI itself.",
    }


def write_markdown(res: dict, fit: dict):
    L = []
    a = L.append
    a("# Evaluation Report: Behavioral Analytics Engine\n")
    a(f"Component of J26-CS-337, by Jayarathna W.D.O.B. (IT23212640). Generated automatically by `scripts/evaluate.py`.\n")
    a(f"**Content model used in these experiments:** `{res['content_model']}`. "
      "Re-run `python scripts/evaluate.py` to regenerate every number.\n")
    a("## 0. Content model (Stage B1): threat signal T\n")
    a("| Model | Jigsaw mean ROC-AUC | Jigsaw macro-F1 (6 labels) | Abusive vs clean F1 (Jigsaw test) | Threat F1 | Davidson tweets F1 (cross-domain) | Moderation set F1 |")
    a("|---|---|---|---|---|---|---|")
    for f in sorted(REPORTS.glob("content_model_*.json")):
        cm = json.loads(f.read_text(encoding="utf-8"))
        a(f"| {cm['model']} | {cm['jigsaw_test_mean_roc_auc']:.3f} | {cm['jigsaw_test_macro_f1']:.3f} | "
          f"{cm['jigsaw_test_abusive_binary']['f1']:.3f} | {cm['jigsaw_test_per_label']['threat']['f1']:.3f} | "
          f"{cm['davidson_cross_domain']['f1']:.3f} | {cm['tcm_sanity']['f1']:.3f} |")
    a("\nPer-label tuned thresholds come from the validation split. The Jigsaw test labels are known to be noisy. "
      "Davidson is short tweet text, closer to live chat.\n")
    a("## 1. Fitted risk model (proposal Appendix B)\n")
    co = fit["coefficients"]
    a(f"- Selected decay half-life: **{fit['selected_half_life_days']} days** (grid search on the validation world)")
    a(f"- Risk = σ({co['intercept']:.2f} + **{co['alpha_T']:.2f}**·T + **{co['beta_w1_logr']:.2f}**·log(1+r) + "
      f"**{co['beta_w2_s']:.2f}**·s + **{co['beta_w3_recency']:.2f}**·recency + **{co['gamma_v']:.2f}**·v)")
    a(f"- β (total TPI weight) = {fit['beta_total']:.2f}; normalised TPI weights: w₁ = {fit['tpi_weights_normalised']['w1_logr']:.2f}, "
      f"w₂ = {fit['tpi_weights_normalised']['w2_s']:.2f}, w₃ = {fit['tpi_weights_normalised']['w3_recency']:.2f}")
    a(f"- Standardised feature importance (coefficient × feature std): {fit['standardised_importance']}")
    a(f"- Tier boundaries: Medium ≥ {fit['tiers']['medium']:.3f}, High ≥ {fit['tiers']['high']:.3f}, Critical ≥ {fit['tiers']['critical']:.3f}\n")

    e1 = res["E1_pair_detection"]
    a("## 2. E1: Pair-level detection on the held-out test world\n")
    a(f"{e1['n_pairs']:,} (offender, target) pairs, of which {e1['positives']} are true repeat-harassment pairs. "
      "Message texts come from a pool the content model never saw.\n")
    a("| System | ROC-AUC | PR-AUC | Precision | Recall | F1 | Pairs flagged |")
    a("|---|---|---|---|---|---|---|")
    names = {"stateless": "(a) Stateless classifier (any T ≥ 0.5)", "frequency": "(b) Frequency rule (≥ 3 hostile)",
             "engine": "**(c) Proposed engine (tier ≥ High)**", "upper_bound_gbm": "(d) Upper bound: gradient boosting"}
    for k, m in e1["systems"].items():
        a(f"| {names[k]} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['flagged']:,} |")
    a("\n![Precision-recall](figures/pr_curve_test_world.png)\n")
    a("Tier distribution by ground-truth label (0 = not a repeat harasser, 1 = repeat harasser):\n")
    a("| Label | " + " | ".join(TIERS) + " |")
    a("|---|" + "---|" * len(TIERS))
    for lab in (0, 1):
        a(f"| {lab} | " + " | ".join(str(e1['tier_by_label'][t].get(lab, 0)) for t in TIERS) + " |")

    a("\n## 3. E2: Ablation study (PR-AUC, test world)\n")
    a("| Variant | PR-AUC |")
    a("|---|---|")
    for k, v in res["E2_ablation_pr_auc"].items():
        a(f"| {k} | {v:.3f} |")
    a("\n![Ablation](figures/ablation_pr_auc.png)\n")

    a("## 4. E3: Who gets flagged\n")
    a("Share of pairs with at least one hostile message that each system flags. For harasser personas only the true victim pair is counted.\n")
    a("| Persona | Pairs | Stateless | Frequency | Engine |")
    a("|---|---|---|---|---|")
    for p, v in res["E3_flag_rate_by_persona_hostile_pairs"].items():
        a(f"| {p} | {int(v['n_pairs'])} | {v['stateless']:.2f} | {v['frequency']:.2f} | {v['engine']:.2f} |")
    a("\nRecall on true harasser pairs, by harasser type:\n")
    a("| Harasser type | Stateless | Frequency | Engine |")
    a("|---|---|---|---|")
    for p, v in res["E3_recall_by_harasser_type"].items():
        a(f"| {p} | {v['stateless']:.2f} | {v['frequency']:.2f} | {v['engine']:.2f} |")
    a("\n![Personas](figures/flag_rate_by_persona.png)\n")

    s = res["E4_streaming"]
    a("## 5. E4: Real-time behaviour (streaming replay)\n")
    a(f"{s['positives']} harasser pairs and {s['negatives']:,} other pairs, replayed message by message.\n")
    a("| System | Harasser pairs detected | Median sessions to detect | False-alarm pairs | False-alarm rate |")
    a("|---|---|---|---|---|")
    for k in ("stateless", "frequency", "engine_medium_or_above", "engine", "engine_no_override"):
        if k not in s:
            continue
        v = s[k]
        a(f"| {k} | {v['harasser_pairs_detected']:.1%} | {v['median_sessions_to_detect']} | {v['false_alarm_pairs']:,} | {v['false_alarm_rate']:.2%} |")
    if "override_only_false_alarms" in s:
        o = s["override_only_false_alarms"]
        a(f"\n`engine_no_override` is the same engine with the explicit-threat override disabled. "
          f"False-alarm pairs caused **only** by the override (P(threat) ≥ {o['threat_override_p']}) in the normal engine: "
          f"**{o['false_alarm_pairs']:,}** ({o['false_alarm_rate_from_override']:.2%} of non-harasser pairs); "
          f"{o['triggered_by_labelled_threat']:,} of them contain a message with a Jigsaw *threat* label, "
          f"{o['triggered_by_non_threat_text']:,} were triggered by text without that label.")
    a("\n![Timeline](figures/risk_timeline_example.png)\n")

    a("## 6. E5: Stress tests\n")
    for name, v in res["E5_stress_tests"].items():
        a(f"**{name}**\n")
        a("| System | ROC-AUC | PR-AUC | Precision | Recall | F1 |")
        a("|---|---|---|---|---|---|")
        for k in ("stateless", "frequency", "engine"):
            m = v[k]
            a(f"| {k} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
        if "victim_pressure_alert" in v:
            vp = v["victim_pressure_alert"]
            a(f"\nVictim-centric coordinated-pressure alert: **{vp['victims_alerted']:.1%}** of harassment victims alerted, "
              f"versus {vp['non_victim_targets_alerted']:.1%} of other targets.")
        a("")

    e6 = res["E6_evasion_recall_on_abusive_messages"]
    a(f"## 7. E6: Adversarial evasion of the content model ({e6['model']}, {e6['n']:,} abusive messages)\n")
    a("| Attack | Recall with normalisation | Recall without |")
    a("|---|---|---|")
    for k, v in e6.items():
        if isinstance(v, dict):
            a(f"| {k} | {v['recall_with_normalisation']:.3f} | {v['recall_without_normalisation']:.3f} |")
    a("\n![Evasion](figures/evasion_recall.png)\n")

    e7 = res["E7_latency"]
    a("## 8. E7: Latency (CPU, per message, end to end incl. content model)\n")
    a("| Store | p50 ms | p95 ms | max ms |")
    a("|---|---|---|---|")
    for k in ("plain_sqlite", "sqlcipher_encrypted"):
        if k not in e7:
            a(f"| {k} | not run (sqlcipher3 not installed) | | |")
            continue
        a(f"| {k} | {e7[k]['p50_ms']} | {e7[k]['p95_ms']} | {e7[k]['max_ms']} |")
    a(f"\nEngine overhead alone: {e7['engine_only_ms_per_message']} ms/message. Target: ≤ {e7['target_ms']} ms.\n")

    e8 = res["E8_real_data_sanity"]
    a("## 9. E8: Real-data sanity check\n")
    a(f"{e8['dataset']}. Users by tier: {e8['users_by_tier']}. Mean toxic comments per user by tier: "
      f"{e8['mean_toxic_comments_by_tier']}. Spearman ρ (toxic count vs tier) = {e8['spearman_toxic_count_vs_tier']}.\n")
    a(f"_{e8['note']}_\n")
    a("## 10. Threats to validity\n")
    a("- **Synthetic behaviour.** The TPI is validated on a simulated multi-session corpus because no public corpus records "
      "cross-session offender–target interactions. Message texts are real, but behaviour patterns are designed. "
      "Hard negatives (banter, bursts, roaming trolls, reformed users) and hard positives (subtle harassers) were added, "
      "along with target-resolution noise, to reduce circularity. Results should be confirmed on real platform logs.")
    a("- **Generator vs model independence.** The generator does not use the TPI formula. The fitted coefficients, "
      "not hand-set weights, decide what matters.")
    a("- **Account rotation** defeats any per-account history. The victim-centric pressure alert is a partial mitigation; "
      "linking accounts (device or behavioural fingerprinting) is future work.")
    a("- **Domain shift.** Jigsaw is Wikipedia talk-page text; live chat is shorter and more informal (see the Davidson cross-domain result).")
    (REPORTS / "EVALUATION_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    if "--report-only" in sys.argv:
        write_markdown(json.loads((REPORTS / "evaluation_results.json").read_text(encoding="utf-8")),
                       json.loads((REPORTS / "risk_model_fit.json").read_text(encoding="utf-8")))
        return
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = EngineConfig.load()
    scorer = load_scorer("auto")
    res = {"content_model": scorer.name}
    t0 = time.time()
    e1_e3(cfg, res); print("E1-E4 done", round(time.time() - t0))
    e5(cfg, res); print("E5 done", round(time.time() - t0))
    e6(scorer, res); print("E6 done", round(time.time() - t0))
    e7(scorer, cfg, res); print("E7 done", round(time.time() - t0))
    e8(scorer, cfg, res); print("E8 done", round(time.time() - t0))
    (REPORTS / "evaluation_results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    fit = json.loads((REPORTS / "risk_model_fit.json").read_text(encoding="utf-8"))
    write_markdown(res, fit)
    print(json.dumps({k: v for k, v in res.items() if k.startswith(("E1", "E2", "E4", "E5"))}, indent=1, default=float))


if __name__ == "__main__":
    main()
