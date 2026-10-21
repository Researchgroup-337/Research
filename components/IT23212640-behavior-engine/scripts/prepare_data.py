"""Step 1 - Prepare all datasets.

Outputs (data/processed/):
  jigsaw_train.parquet / jigsaw_val.parquet   -> fine-tune the content model
  jigsaw_test.parquet                          -> official held-out Kaggle test (labelled rows only)
  davidson_eval.parquet                        -> cross-domain test (short tweets)
  tcm_eval.parquet                             -> Toxic Comment Moderation sanity set (has user IDs)
  pool_gen_train.parquet / pool_gen_test.parquet -> message pools for the synthetic
      multi-session generator. These texts are NEVER seen by the content model, so the
      threat score T the engine receives in experiments is realistic, not memorised.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bae.config import DATA_PROCESSED, DATA_RAW, LABELS  # noqa: E402

SEED = 42


def severity_bucket(row) -> str:
    if row["threat"] == 1:
        return "threat"
    if row["severe_toxic"] == 1 or row["identity_hate"] == 1:
        return "severe"
    if row["toxic"] == 1 or row["insult"] == 1 or row["obscene"] == 1:
        return "mild"
    return "clean"


def shorten(text: str, max_words: int = 40) -> str:
    """Live chat is short. Keep the first max_words words of long talk-page comments."""
    w = str(text).split()
    return " ".join(w[:max_words])


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    report = []

    # ---------------- Jigsaw
    tr = pd.read_csv(DATA_RAW / "train.csv")
    tr = tr.drop_duplicates("comment_text")
    tr["bucket"] = tr.apply(severity_bucket, axis=1)
    strat = tr["bucket"]
    model_part, gen_part = train_test_split(tr, test_size=0.10, stratify=strat, random_state=SEED)
    jtrain, jval = train_test_split(model_part, test_size=0.1111, stratify=model_part["bucket"], random_state=SEED)

    te = pd.read_csv(DATA_RAW / "test.csv").merge(pd.read_csv(DATA_RAW / "test_labels.csv"), on="id")
    te = te[te["toxic"] != -1].copy()  # -1 = not scored by Kaggle, unusable
    te["bucket"] = te.apply(severity_bucket, axis=1)

    for name, df in [("jigsaw_train", jtrain), ("jigsaw_val", jval), ("jigsaw_test", te)]:
        df[["id", "comment_text", *LABELS, "bucket"]].reset_index(drop=True).to_parquet(DATA_PROCESSED / f"{name}.parquet")
        report.append(f"{name}: {len(df):,} rows | threat={int(df.threat.sum())} toxic={int(df.toxic.sum())}")

    # ---------------- Davidson (tweets): half cross-domain eval, half generator pool
    dv = pd.read_csv(DATA_RAW / "davidson.csv")
    dv["text"] = dv["tweet"]
    dv["is_abusive"] = (dv["class"] != 2).astype(int)
    dv["bucket"] = dv["class"].map({0: "severe", 1: "mild", 2: "clean"})
    dv_eval, dv_pool = train_test_split(dv, test_size=0.5, stratify=dv["class"], random_state=SEED)
    dv_eval[["text", "class", "is_abusive"]].reset_index(drop=True).to_parquet(DATA_PROCESSED / "davidson_eval.parquet")
    report.append(f"davidson_eval: {len(dv_eval):,} rows | abusive={int(dv_eval.is_abusive.sum())}")

    # ---------------- Toxic Comment Moderation (xlsx)
    tcm = pd.read_excel(DATA_RAW / "toxic_comment_moderation.xlsx")
    tcm["platform"] = tcm["platform"].str.strip().str.title().replace({"Youtube": "YouTube"})
    tcm["is_toxic"] = tcm["is_toxic"].astype(int)
    tcm.to_parquet(DATA_PROCESSED / "tcm_eval.parquet")
    report.append(f"tcm_eval: {len(tcm):,} rows, {tcm.user_id.nunique()} users | toxic={int(tcm.is_toxic.sum())}"
                  f" | unique texts={tcm.comment_text.nunique()}")

    # ---------------- Generator pools (unseen by content model)
    pool = pd.concat([
        gen_part.rename(columns={"comment_text": "text"})[["text", "bucket"]].assign(source="jigsaw_holdout"),
        te.rename(columns={"comment_text": "text"})[["text", "bucket"]].assign(source="jigsaw_test"),
        dv_pool[["text", "bucket"]].assign(source="davidson"),
    ], ignore_index=True)
    pool["text"] = pool["text"].map(shorten)
    pool = pool[pool["text"].str.split().str.len() >= 3].drop_duplicates("text")
    p_train, p_test = train_test_split(pool, test_size=0.5, stratify=pool["bucket"], random_state=SEED)
    for name, df in [("pool_gen_train", p_train), ("pool_gen_test", p_test)]:
        df.reset_index(drop=True).to_parquet(DATA_PROCESSED / f"{name}.parquet")
        report.append(f"{name}: {len(df):,} | " + ", ".join(f"{k}={v}" for k, v in df.bucket.value_counts().items()))

    # ---------------- Note on the Combined CSV
    comb = pd.read_csv(DATA_RAW / "combined_hate_speech.csv")
    overlap = comb["text"].isin(dv["tweet"]).sum()
    report.append(f"combined_hate_speech: {len(comb):,} rows, {overlap:,} overlap with Davidson -> EXCLUDED (leakage risk)")

    txt = "\n".join(report)
    (DATA_PROCESSED / "data_report.txt").write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
