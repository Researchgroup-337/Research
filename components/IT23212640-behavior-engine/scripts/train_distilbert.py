"""Step 2b - Fine-tune DistilBERT (multi-label, 6 Jigsaw labels). Run on a GPU (Colab T4).

    python scripts/train_distilbert.py                    # full run (~1-1.5 h on a T4)
    python scripts/train_distilbert.py --smoke            # 2-minute CPU check, random tiny model

What it does
  1. Loads data/processed/jigsaw_{train,val}.parquet (created by prepare_data.py).
  2. Normalises text (same normaliser used at inference -> evasion robustness).
  3. Oversamples threat examples and uses a positive-class weight in BCE loss, because
     threat is only 0.3% of comments.
  4. Fine-tunes distilbert-base-uncased, tunes per-label thresholds on validation.
  5. Saves models/distilbert/ (HF format + thresholds.json), exports ONNX and an int8
     dynamically quantised ONNX for fast CPU inference, and writes the evaluation report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bae.config import DATA_PROCESSED, LABELS, MODELS, REPORTS  # noqa: E402
from bae.content_model import DistilBertScorer, evaluate_scorer, tune_thresholds  # noqa: E402
from bae.text_norm import normalize  # noqa: E402


class TextDS(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.enc = tok(texts, truncation=True, max_length=max_len)
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--threat-oversample", type=int, default=4, help="repeat threat rows N times")
    ap.add_argument("--out", default=str(MODELS / "distilbert"))
    ap.add_argument("--smoke", action="store_true", help="tiny random model + 2k rows, CPU, no download")
    ap.add_argument("--skip-onnx", action="store_true")
    args = ap.parse_args()

    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              DistilBertConfig, DistilBertForSequenceClassification, Trainer, TrainingArguments)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tr = pd.read_parquet(DATA_PROCESSED / "jigsaw_train.parquet")
    va = pd.read_parquet(DATA_PROCESSED / "jigsaw_val.parquet")

    if args.smoke:
        tr = pd.concat([tr[tr.threat == 1].head(100), tr.sample(1900, random_state=0)])
        va = va.sample(500, random_state=0)
        tok = _smoke_tokenizer(tr.comment_text.tolist(), out)
        cfg = DistilBertConfig(vocab_size=tok.vocab_size, dim=64, hidden_dim=128, n_layers=2, n_heads=2,
                               num_labels=len(LABELS), problem_type="multi_label_classification")
        model = DistilBertForSequenceClassification(cfg)
        args.epochs, args.batch = 1, 32
    else:
        tok = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=len(LABELS), problem_type="multi_label_classification",
            id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})

    # Oversample the rare threat class.
    if args.threat_oversample > 1:
        tr = pd.concat([tr] + [tr[tr.threat == 1]] * (args.threat_oversample - 1)).sample(frac=1, random_state=42)

    Ytr = tr[LABELS].values
    pos = Ytr.mean(0)
    pos_weight = torch.tensor(np.clip((1 - pos) / np.clip(pos, 1e-6, None), 1, 30) ** 0.5, dtype=torch.float32)
    print("pos_weight", dict(zip(LABELS, pos_weight.numpy().round(2))))

    ds_tr = TextDS([normalize(t) for t in tr.comment_text], Ytr, tok, args.max_len)
    ds_va = TextDS([normalize(t) for t in va.comment_text], va[LABELS].values, tok, args.max_len)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                outputs.logits, labels, pos_weight=pos_weight.to(outputs.logits.device))
            return (loss, outputs) if return_outputs else loss

    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"), num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch * 2,
        warmup_steps=int(0.06 * args.epochs * len(ds_tr) / args.batch), weight_decay=0.01, fp16=torch.cuda.is_available(),
        eval_strategy="epoch", save_strategy="no", logging_steps=200, report_to=[], seed=42,
    )
    trainer = WeightedTrainer(model=model, args=targs, train_dataset=ds_tr, eval_dataset=ds_va,
                              data_collator=DataCollatorWithPadding(tok))
    t0 = time.time()
    trainer.train()
    train_sec = time.time() - t0

    logits = trainer.predict(ds_va).predictions
    Pva = 1 / (1 + np.exp(-logits))
    thr = tune_thresholds(va[LABELS].values, Pva)
    trainer.save_model(str(out))
    tok.save_pretrained(str(out))
    (out / "thresholds.json").write_text(json.dumps({"thresholds": dict(zip(LABELS, thr.tolist())),
                                                     "max_len": args.max_len, "base": args.model_name,
                                                     "smoke": args.smoke}, indent=2), encoding="utf-8")
    print("thresholds", dict(zip(LABELS, thr.round(3))))

    if not args.skip_onnx:
        export_onnx(out, tok, args.max_len)

    if args.smoke:
        print("SMOKE TEST OK - pipeline runs end to end (metrics meaningless for a random tiny model).")
        return

    scorer = DistilBertScorer(out)
    res = evaluate_scorer(scorer, DATA_PROCESSED)
    res["train_seconds"] = round(train_sec, 1)
    res["latency_ms_per_message"] = measure_latency(scorer)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "content_model_distilbert.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


def export_onnx(out: Path, tok, max_len: int):
    """Export to ONNX and apply dynamic int8 quantisation (about 4x smaller, 2-3x faster on CPU)."""
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(out).eval().cpu()
    dummy = tok(["export example"], return_tensors="pt", padding="max_length", max_length=16)
    onnx_path = out / "model.onnx"
    torch.onnx.export(
        model, (dummy["input_ids"], dummy["attention_mask"]), str(onnx_path),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "logits": {0: "b"}},
        opset_version=17, dynamo=False,
    )
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(onnx_path), str(out / "model.quant.onnx"), weight_type=QuantType.QInt8)
        print("ONNX + int8 export done")
    except Exception as e:  # pragma: no cover
        print("quantisation skipped:", e)


def measure_latency(scorer, n: int = 200) -> dict:
    msgs = ["I know where you live and I will find you", "gg nice stream", "you are pathetic, quit"] * (n // 3)
    times = []
    for m in msgs:
        t = time.perf_counter()
        scorer.score_one(m)
        times.append((time.perf_counter() - t) * 1000)
    return {"p50": round(float(np.percentile(times, 50)), 2), "p95": round(float(np.percentile(times, 95)), 2)}


def _smoke_tokenizer(texts, out: Path):
    """Build a small WordPiece tokenizer locally so the smoke test needs no download."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    t = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    t.normalizer = normalizers.BertNormalizer(lowercase=True)
    t.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    t.train_from_iterator([normalize(x) for x in texts], trainers.WordPieceTrainer(
        vocab_size=3000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    from tokenizers.processors import TemplateProcessing
    t.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    return PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                   sep_token="[SEP]", mask_token="[MASK]", model_input_names=["input_ids", "attention_mask"])


if __name__ == "__main__":
    main()
