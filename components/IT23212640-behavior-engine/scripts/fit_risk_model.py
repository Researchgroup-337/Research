"""Step 4 - Fit the composite risk model and calibrate tier boundaries (proposal Appendix B).

  1. Replay the TRAIN world through the engine; take pair snapshots on days 30, 45 and the end.
  2. For each decay half-life in a grid, fit logistic regression on
         [T_pair, log(1+r_eff), s_eff, recency, v]  ->  repeat-harassment label
     and pick the half-life with the best PR-AUC on the VALIDATION world.
  3. The fitted coefficients ARE alpha, beta*w1, beta*w2, beta*w3, gamma.
  4. Tier boundaries chosen on the validation world:
         High     = threshold that maximises F1
         Critical = lowest threshold with precision >= 0.95 (and >= High)
         Medium   = highest threshold with recall >= 0.95 (and <= High)
Writes models/risk_model.json and reports/risk_model_fit.json.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bae.config import DATA_PROCESSED, MODELS, REPORTS, EngineConfig  # noqa: E402
from bae.experiment import X, label_snapshot, replay  # noqa: E402
from bae.risk import FEATURES  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_synthetic import BASE_TS  # noqa: E402

SNAP_DAYS = [30, 45]
HALF_LIVES = [3.0, 7.0, 14.0, 30.0, float("inf")]


def load(name):
    return pd.read_parquet(DATA_PROCESSED / f"synthetic_{name}.parquet"), pd.read_parquet(DATA_PROCESSED / f"synthetic_{name}_pairs.parquet")


def build_xy(df, pairs, cfg, snap_days=SNAP_DAYS, base_ts=None):
    rep = replay(df, cfg, snap_days, base_ts=base_ts, track_stream=False)
    frames = [label_snapshot(s, pairs) for s in rep["snapshots"].values() if len(s)]
    return pd.concat(frames, ignore_index=True)


class MonotoneLR:
    """Logistic regression with NON-NEGATIVE feature coefficients (L2, C=1).

    Security rationale: more hostility, more repetition or more escalation must never LOWER
    a user's risk. An unconstrained model can learn small negative weights from correlated
    features, which an adversary could exploit (e.g. escalating to reduce the score)."""

    def __init__(self, C: float = 1.0):
        self.C = C

    def fit(self, Xm, y):
        from scipy.optimize import minimize
        n, d = Xm.shape

        def loss(w):
            z = Xm @ w[:d] + w[d]
            p = 1 / (1 + np.exp(-z))
            nll = np.sum(np.logaddexp(0, z) - y * z)
            g = Xm.T @ (p - y) + w[:d] / self.C
            return nll + 0.5 * np.dot(w[:d], w[:d]) / self.C, np.append(g, np.sum(p - y))

        bounds = [(0, None)] * d + [(None, None)]
        r = minimize(loss, np.zeros(d + 1), jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 5000})
        self.coef_ = r.x[:d][None, :]
        self.intercept_ = np.array([r.x[d]])
        return self

    def predict_proba(self, Xm):
        p = 1 / (1 + np.exp(-(Xm @ self.coef_[0] + self.intercept_[0])))
        return np.column_stack([1 - p, p])


def fit_lr(train: pd.DataFrame, feats=FEATURES):
    return MonotoneLR().fit(X(train, feats), train.label.values.astype(float))


def tier_thresholds(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    high = float(thr[np.argmax(f1)])
    ok = np.where(prec >= 0.95)[0]
    crit = float(thr[ok[0]]) if len(ok) else min(0.95, high + 0.2)
    crit = max(crit, high + 0.05)
    okr = np.where(rec >= 0.95)[0]
    med = float(thr[okr[-1]]) if len(okr) else high / 2
    med = min(med, high - 0.05)
    return max(med, 0.05), high, min(crit, 0.99)


def main():
    tr_df, tr_pairs = load("train")
    va_df, va_pairs = load("val")
    grid = []
    best = None
    for hl in HALF_LIVES:
        cfg = EngineConfig(half_life_days=hl if math.isfinite(hl) else 0.0)
        tr = build_xy(tr_df, tr_pairs, cfg, base_ts=BASE_TS)
        va = build_xy(va_df, va_pairs, cfg, snap_days=[], base_ts=BASE_TS)  # end-of-window snapshot
        lr = fit_lr(tr)
        p = lr.predict_proba(X(va))[:, 1]
        row = {"half_life_days": hl, "val_pr_auc": average_precision_score(va.label, p),
               "val_roc_auc": roc_auc_score(va.label, p), "n_train_rows": len(tr), "train_pos": int(tr.label.sum())}
        grid.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
        if best is None or row["val_pr_auc"] > best[0]["val_pr_auc"]:
            best = (row, cfg, lr, va, p)

    row, cfg, lr, va, p = best
    med, high, crit = tier_thresholds(va.label.values, p)
    coef = dict(zip(FEATURES, lr.coef_[0]))
    cfg.intercept = float(lr.intercept_[0])
    cfg.coef_T, cfg.coef_logr, cfg.coef_s = coef["T_pair"], coef["log_r"], coef["s"]
    cfg.coef_recency, cfg.coef_v = coef["recency"], coef["v"]
    cfg.tier_medium, cfg.tier_high, cfg.tier_critical = med, high, crit
    MODELS.mkdir(exist_ok=True)
    cfg.to_json(MODELS / "risk_model.json")

    beta = cfg.coef_logr + cfg.coef_s + cfg.coef_recency
    summary = {
        "selected_half_life_days": cfg.half_life_days,
        "grid": grid,
        "coefficients": {"intercept": cfg.intercept, "alpha_T": cfg.coef_T, "beta_w1_logr": cfg.coef_logr,
                         "beta_w2_s": cfg.coef_s, "beta_w3_recency": cfg.coef_recency, "gamma_v": cfg.coef_v},
        "beta_total": beta,
        "tpi_weights_normalised": {"w1_logr": cfg.coef_logr / beta, "w2_s": cfg.coef_s / beta, "w3_recency": cfg.coef_recency / beta},
        "feature_scale_note": "coefficients are on raw feature scales; see standardised_importance",
        "standardised_importance": dict(zip(FEATURES, (lr.coef_[0] * X(va).std(0)).round(4).tolist())),
        "tiers": {"medium": med, "high": high, "critical": crit},
        "val_pr_auc": row["val_pr_auc"], "val_roc_auc": row["val_roc_auc"],
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "risk_model_fit.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))
    print("saved", MODELS / "risk_model.json", asdict(cfg)["half_life_days"])


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
