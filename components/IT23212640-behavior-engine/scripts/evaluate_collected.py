"""Evaluate the engine on a dataset collected with SafeChat Lab (real/hybrid sessions).

    python scripts/evaluate_collected.py --export data/safechat/safechat_dataset_XXXX.zip [--rescore]

--rescore   re-score every message with the current content model (e.g. after installing
            DistilBERT) instead of using the T logged at collection time.
Writes reports/collected/COLLECTED_REPORT.md and collected_results.json.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from bae.config import REPORTS, EngineConfig  # noqa: E402
from bae.experiment import label_snapshot, replay  # noqa: E402
from bae.risk import TIERS, tier_for  # noqa: E402
from evaluate import lr_prob, metrics, streaming, system_scores  # noqa: E402

OUT = REPORTS / "collected"


def load_export(path: Path):
    z = zipfile.ZipFile(path)
    rd = lambda n: pd.read_csv(io.BytesIO(z.read(n)))  # noqa: E731
    return rd("messages.csv"), rd("pairs.csv"), rd("participants.csv"), rd("sessions.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--rescore", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    msgs, pairs, parts, sess = load_export(Path(a.export))
    if "engine_ts" in msgs and msgs.engine_ts.notna().all():
        msgs["ts"] = msgs["engine_ts"]  # replay on the lab clock the engine used live
    msgs = msgs.sort_values("ts").reset_index(drop=True)
    msgs["target"] = msgs["target"].astype(str)
    if a.rescore:
        from bae.content_model import load_scorer, threat_signal
        sc = load_scorer("auto")
        P = sc.predict_proba(msgs.text.fillna("").tolist())
        msgs["T"], msgs["p_threat"] = threat_signal(P), P[:, 3]
        model = sc.name
    else:
        model = "as logged"
    msgs["T"] = msgs["T"].fillna(0.0)
    msgs["p_threat"] = msgs["p_threat"].fillna(0.0)
    pairs["target"] = pairs["target"].astype(str)
    pairs["persona"] = pairs["role"].fillna("normal")

    cfg = EngineConfig.load()
    df = msgs[["message_id", "sender", "target", "session_id", "ts", "T", "p_threat"]].copy()
    rep = replay(df, cfg, [], base_ts=df.ts.min(), track_stream=True)
    snap = label_snapshot(rep["snapshots"][rep["end_ts"]], pairs)
    snap["tier"] = [tier_for(p, cfg) for p in lr_prob(cfg, snap)]

    res = {"content_model": model, "dataset": {
        "messages": len(msgs), "sessions": int(msgs.session_id.nunique()),
        "participants": int(msgs.sender.nunique()),
        "humans": int((~parts.is_bot.astype(bool)).sum()), "bots": int(parts.is_bot.astype(bool).sum()),
        "pairs": len(snap), "positive_pairs": int(snap.label.sum()),
        "hostile_message_rate": round(float((msgs["T"] >= 0.5).mean()), 4),
        "target_method": msgs.target_method.value_counts().to_dict(),
        "mean_sessions_per_positive_pair": round(float(pairs[pairs.label == 1].sessions.mean()), 2) if (pairs.label == 1).any() else None,
    }}
    both = snap.label.nunique() == 2
    if both:
        res["pair_detection"] = {k: metrics(snap.label, sc_, fl) for k, (sc_, fl) in system_scores(snap, cfg).items()}
        res["streaming"] = streaming(df, pairs, rep, cfg)
    res["tier_by_label"] = pd.crosstab(snap.label, snap.tier).reindex(columns=TIERS, fill_value=0).to_dict()
    res["tier_by_role"] = pd.crosstab(snap.persona.fillna("normal"), snap.tier).reindex(columns=TIERS, fill_value=0).to_dict(orient="index")
    (OUT / "collected_results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")

    L = ["# Engine evaluation on SafeChat Lab collected data\n",
         f"Export: `{Path(a.export).name}` · content model: {model}\n", "## Dataset\n"]
    for k, v in res["dataset"].items():
        L.append(f"- **{k.replace('_', ' ')}**: {v}")
    if both:
        L += ["\n## Pair-level detection\n", "| System | ROC-AUC | PR-AUC | Precision | Recall | F1 | Flagged |", "|---|---|---|---|---|---|---|"]
        for k, m in res["pair_detection"].items():
            L.append(f"| {k} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['flagged']} |")
        s = res["streaming"]
        L += ["\n## Streaming detection\n", "| System | Harasser pairs detected | Median sessions to detect | False-alarm pairs |", "|---|---|---|---|"]
        for k in ("stateless", "frequency", "engine_medium_or_above", "engine"):
            v = s[k]
            L.append(f"| {k} | {v['harasser_pairs_detected']:.1%} | {v['median_sessions_to_detect']} | {v['false_alarm_pairs']} |")
    else:
        L.append("\n_Only one class present in the labels - pair-level metrics need at least one harassment pair and one normal pair._")
    L += ["\n## Tier by role (pairs)\n", "| Role | " + " | ".join(TIERS) + " |", "|---|" + "---|" * 4]
    for r, v in res["tier_by_role"].items():
        L.append(f"| {r} | " + " | ".join(str(v.get(t, 0)) for t in TIERS) + " |")
    L.append("\n_Small samples: report counts and confidence intervals, and treat this as validation of the synthetic results rather than a replacement._")
    (OUT / "COLLECTED_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
