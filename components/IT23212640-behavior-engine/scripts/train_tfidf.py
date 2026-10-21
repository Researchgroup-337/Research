"""Step 2a - Train the classical baseline content model (TF-IDF + logistic regression).

Word 1-2 grams + character 2-5 grams (char n-grams add robustness to misspellings).
One logistic regression per Jigsaw label; per-label threshold tuned on validation.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bae.config import DATA_PROCESSED, LABELS, MODELS, REPORTS  # noqa: E402
from bae.content_model import TfidfScorer, evaluate_scorer, tune_thresholds  # noqa: E402
from bae.text_norm import normalize  # noqa: E402


def main():
    t0 = time.time()
    tr = pd.read_parquet(DATA_PROCESSED / "jigsaw_train.parquet")
    va = pd.read_parquet(DATA_PROCESSED / "jigsaw_val.parquet")
    Xtr_txt = [normalize(t) for t in tr.comment_text]
    Xva_txt = [normalize(t) for t in va.comment_text]

    vec = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=5, max_features=200_000, sublinear_tf=True)),
    ])
    Xtr = vec.fit_transform(Xtr_txt)
    Xva = vec.transform(Xva_txt)
    print(f"vectorised {Xtr.shape} in {time.time()-t0:.0f}s")

    clfs = []
    for l in LABELS:
        c = LogisticRegression(C=4.0, max_iter=2000, solver="liblinear")
        c.fit(Xtr, tr[l].values)
        clfs.append(c)
        print(f"  trained {l}")
    Pva = np.column_stack([c.predict_proba(Xva)[:, 1] for c in clfs])
    thr = tune_thresholds(va[LABELS].values, Pva)
    print("thresholds", dict(zip(LABELS, thr.round(3))))

    MODELS.mkdir(exist_ok=True)
    joblib.dump({"vectorizer": vec, "classifiers": clfs, "thresholds": thr.tolist()}, MODELS / "tfidf_lr.joblib", compress=3)

    res = evaluate_scorer(TfidfScorer(), DATA_PROCESSED)
    res["train_seconds"] = round(time.time() - t0, 1)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "content_model_tfidf.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "jigsaw_test_per_label"}, indent=2))
    for l, v in res["jigsaw_test_per_label"].items():
        print(f"  {l:14s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} AUC={v['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
