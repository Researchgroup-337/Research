"""Stage B1 - message-level content threat scoring.

Two interchangeable scorers share one interface:
  * DistilBertScorer - the main model (fine-tuned DistilBERT, multi-label, 6 Jigsaw labels).
    Loads an ONNX export when present (fast CPU inference), otherwise the HF checkpoint.
  * TfidfScorer      - TF-IDF + logistic regression. Used as the classical baseline and as a
    fallback so the engine runs on machines without the DistilBERT weights.

Every scorer returns calibrated per-label probabilities in which 0.5 is the tuned decision
threshold for that label, plus the single threat signal T used by the behavioural engine.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import LABEL_WEIGHTS, LABELS, MODELS
from .text_norm import normalize


def recalibrate(p: np.ndarray, thr: np.ndarray) -> np.ndarray:
    """Monotone piecewise-linear map that sends each label's tuned threshold to 0.5."""
    thr = np.clip(thr, 1e-4, 1 - 1e-4)
    low = 0.5 * p / thr
    high = 0.5 + 0.5 * (p - thr) / (1 - thr)
    return np.where(p < thr, low, high).clip(0, 1)


def threat_signal(probs: np.ndarray) -> np.ndarray:
    """T = max_k(w_k * p_k) over the six labels (probs shape [n, 6])."""
    w = np.array([LABEL_WEIGHTS[l] for l in LABELS])
    return (probs * w).max(axis=1)


class ContentScorer:
    name = "base"
    thresholds: np.ndarray = np.full(len(LABELS), 0.5)

    def raw_probs(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def predict_proba(self, texts: Iterable[str], normalise: bool = True) -> np.ndarray:
        texts = [normalize(t) if normalise else t for t in texts]
        return recalibrate(self.raw_probs(texts), self.thresholds)

    def score(self, texts: Iterable[str], normalise: bool = True) -> list[dict]:
        probs = self.predict_proba(list(texts), normalise)
        T = threat_signal(probs)
        return [{"T": float(T[i]), **{l: float(probs[i, j]) for j, l in enumerate(LABELS)}} for i in range(len(T))]

    def score_one(self, text: str) -> dict:
        return self.score([text])[0]


class TfidfScorer(ContentScorer):
    name = "tfidf_lr"

    def __init__(self, path: Path = MODELS / "tfidf_lr.joblib"):
        import joblib
        bundle = joblib.load(path)
        self.vec = bundle["vectorizer"]
        self.clfs = bundle["classifiers"]
        self.thresholds = np.array(bundle["thresholds"])

    def raw_probs(self, texts):
        X = self.vec.transform(texts)
        return np.column_stack([c.predict_proba(X)[:, 1] for c in self.clfs])


class DistilBertScorer(ContentScorer):
    name = "distilbert"

    def __init__(self, path: Path = MODELS / "distilbert", max_len: int = 128, prefer_onnx: bool = True):
        from transformers import AutoTokenizer
        self.path = Path(path)
        self.max_len = max_len
        self.tok = AutoTokenizer.from_pretrained(self.path)
        meta = json.loads((self.path / "thresholds.json").read_text(encoding="utf-8"))
        self.thresholds = np.array([meta["thresholds"][l] for l in LABELS])
        onnx_file = self.path / "model.quant.onnx"
        if not onnx_file.exists():
            onnx_file = self.path / "model.onnx"
        self.session = None
        if prefer_onnx and onnx_file.exists():
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                self.session = ort.InferenceSession(str(onnx_file), opts, providers=["CPUExecutionProvider"])
                self.name = "distilbert_onnx"
            except Exception:  # pragma: no cover - fall back to torch
                self.session = None
        if self.session is None:
            import torch
            from transformers import AutoModelForSequenceClassification
            self.torch = torch
            self.model = AutoModelForSequenceClassification.from_pretrained(self.path).eval()

    def raw_probs(self, texts, batch_size: int = 64):
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            if self.session is not None:
                enc = self.tok(chunk, truncation=True, max_length=self.max_len, padding=True, return_tensors="np")
                feeds = {k: enc[k].astype(np.int64) for k in ("input_ids", "attention_mask")}
                logits = self.session.run(None, feeds)[0]
            else:
                enc = self.tok(chunk, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
                with self.torch.no_grad():
                    logits = self.model(**enc).logits.numpy()
            out.append(1 / (1 + np.exp(-logits)))
        return np.vstack(out)


class ExternalSignalScorer(ContentScorer):
    """Used in experiments where T is pre-computed and stored with the corpus."""
    name = "precomputed"


def load_scorer(kind: str = "auto") -> ContentScorer:
    """auto -> DistilBERT if models/distilbert/thresholds.json exists, else TF-IDF."""
    if kind in ("auto", "distilbert") and (MODELS / "distilbert" / "thresholds.json").exists():
        return DistilBertScorer()
    if kind == "distilbert":
        raise FileNotFoundError("models/distilbert not found - run the Colab notebook first")
    return TfidfScorer()


# ------------------------------------------------------------- shared evaluation
def tune_thresholds(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Pick, per label, the threshold that maximises F1 on validation data."""
    from sklearn.metrics import precision_recall_curve
    thr = []
    for j in range(y.shape[1]):
        prec, rec, t = precision_recall_curve(y[:, j], p[:, j])
        f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
        thr.append(float(t[np.nanargmax(f1[:-1])]) if len(t) else 0.5)
    return np.array(thr)


def evaluate_scorer(scorer: ContentScorer, data_dir: Path) -> dict:
    import pandas as pd
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

    res = {"model": scorer.name}
    te = pd.read_parquet(data_dir / "jigsaw_test.parquet")
    P = scorer.predict_proba(te.comment_text.tolist())
    Y = te[LABELS].values
    per = {}
    for j, l in enumerate(LABELS):
        yhat = P[:, j] >= 0.5
        per[l] = {
            "precision": round(precision_score(Y[:, j], yhat, zero_division=0), 4),
            "recall": round(recall_score(Y[:, j], yhat, zero_division=0), 4),
            "f1": round(f1_score(Y[:, j], yhat, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(Y[:, j], P[:, j]), 4),
            "pr_auc": round(average_precision_score(Y[:, j], P[:, j]), 4),
            "support": int(Y[:, j].sum()),
        }
    res["jigsaw_test_per_label"] = per
    res["jigsaw_test_mean_roc_auc"] = round(float(np.mean([v["roc_auc"] for v in per.values()])), 4)
    res["jigsaw_test_macro_f1"] = round(float(np.mean([v["f1"] for v in per.values()])), 4)
    any_y = (Y.sum(1) > 0).astype(int)
    T = threat_signal(P)
    res["jigsaw_test_abusive_binary"] = {
        "f1": round(f1_score(any_y, T >= 0.5), 4),
        "precision": round(precision_score(any_y, T >= 0.5), 4),
        "recall": round(recall_score(any_y, T >= 0.5), 4),
        "roc_auc": round(roc_auc_score(any_y, T), 4),
    }

    dv = pd.read_parquet(data_dir / "davidson_eval.parquet")
    Td = threat_signal(scorer.predict_proba(dv.text.tolist()))
    res["davidson_cross_domain"] = {
        "f1": round(f1_score(dv.is_abusive, Td >= 0.5), 4),
        "precision": round(precision_score(dv.is_abusive, Td >= 0.5), 4),
        "recall": round(recall_score(dv.is_abusive, Td >= 0.5), 4),
        "roc_auc": round(roc_auc_score(dv.is_abusive, Td), 4),
        "n": len(dv),
    }
    tcm = pd.read_parquet(data_dir / "tcm_eval.parquet")
    Pt = scorer.predict_proba(tcm.comment_text.tolist())
    Tt = threat_signal(Pt)
    res["tcm_sanity"] = {
        "f1": round(f1_score(tcm.is_toxic, Tt >= 0.5), 4),
        "precision": round(precision_score(tcm.is_toxic, Tt >= 0.5), 4),
        "recall": round(recall_score(tcm.is_toxic, Tt >= 0.5), 4),
        "roc_auc": round(roc_auc_score(tcm.is_toxic, Tt), 4),
        "n": len(tcm),
    }
    return res
