"""Check that everything is installed and in the right place, with plain-language fixes.

    python scripts/check_setup.py            # can the engine, API and chat app run?
    python scripts/check_setup.py --rebuild  # also: is the raw data there to rebuild all results?
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
ok_all = True


def report(ok: bool, what: str, fix: str = "", critical: bool = True):
    global ok_all
    print(("  [ OK ] " if ok else ("  [FAIL] " if critical else "  [WARN] ")) + what)
    if not ok and fix:
        print("         -> " + fix)
    if not ok and critical:
        ok_all = False


def main():
    rebuild = "--rebuild" in sys.argv
    print(f"\nProject folder: {ROOT}\n")

    print("1. Python")
    v = sys.version_info
    report(v >= (3, 10), f"Python {v.major}.{v.minor}", "Install Python 3.11 from python.org (tick 'Add Python to PATH')")

    print("2. Libraries")
    for mod, pipname in [("numpy", "numpy"), ("pandas", "pandas"), ("pyarrow", "pyarrow"), ("sklearn", "scikit-learn"),
                         ("scipy", "scipy"), ("joblib", "joblib"), ("openpyxl", "openpyxl"), ("pydantic", "pydantic"),
                         ("fastapi", "fastapi"), ("uvicorn", "uvicorn"), ("httpx", "httpx"), ("websockets", "websockets"),
                         ("matplotlib", "matplotlib"), ("transformers", "transformers"), ("onnxruntime", "onnxruntime")]:
        try:
            importlib.import_module(mod)
            report(True, mod)
        except Exception as e:
            report(False, f"{mod} missing ({e.__class__.__name__})", f"pip install {pipname}")
    try:
        importlib.import_module("sqlcipher3")
        report(True, "sqlcipher3 (database encryption)")
    except Exception:
        report(False, "sqlcipher3 not installed - database encryption off (normal on Windows)", critical=False)

    print("3. Models")
    report((ROOT / "models" / "tfidf_lr.joblib").exists(), "TF-IDF model  models/tfidf_lr.joblib",
           "Re-extract behavior_engine.zip - this file ships with it")
    report((ROOT / "models" / "risk_model.json").exists(), "Risk model   models/risk_model.json",
           "Re-extract behavior_engine.zip, or run scripts/fit_risk_model.py")
    dist = ROOT / "models" / "distilbert"
    has_d = (dist / "thresholds.json").exists()
    report(has_d, "DistilBERT   models/distilbert/thresholds.json",
           "Unzip distilbert_model.zip INSIDE the behavior_engine folder so that models/distilbert/ appears "
           "(not models/models/distilbert or behavior_engine/behavior_engine/...)", critical=False)
    if has_d:
        report((dist / "model.quant.onnx").exists() or (dist / "model.onnx").exists(), "DistilBERT ONNX file (fast CPU inference)",
               "Not fatal - the engine falls back to PyTorch (slower). Re-run the last Colab cells to get model.onnx", critical=False)
        report((dist / "tokenizer.json").exists() or (dist / "vocab.txt").exists(), "DistilBERT tokenizer files",
               "The download is incomplete - re-download distilbert_model.zip from Colab")

    print("4. Live test")
    try:
        from bae.content_model import load_scorer
        t = time.time()
        sc = load_scorer("auto")
        load_s = time.time() - t
        r1, r2 = sc.score_one("I know where you live and I will hurt you"), sc.score_one("great stream, thanks!")
        report(r1["T"] > r2["T"], f"content model '{sc.name}' loads ({load_s:.1f}s): threat T={r1['T']:.2f}, friendly T={r2['T']:.2f}",
               "The model loaded but scores look wrong - send this output to Claude")
        t = time.time()
        for _ in range(20):
            sc.score_one("you are pathetic, nobody likes you")
        report(True, f"latency {((time.time() - t) / 20) * 1000:.1f} ms per message (target <= 250 ms)")
    except Exception as e:
        report(False, f"content model failed: {e}", "Send this error text to Claude")
    try:
        from bae.engine import BehavioralEngine
        eng = BehavioralEngine(scorer=None)
        rec = None
        for s in range(6):
            for m in range(3):
                rec = eng.process_scored(f"m{s}{m}", "bully", "victim", f"s{s}", 1.78e9 + s * 2 * 86400 + m, 0.8)
        report(rec["pair"]["tier"] in ("High", "Critical"), f"behaviour engine: 6-session harasser reaches {rec['pair']['tier']}")
    except Exception as e:
        report(False, f"engine failed: {e}", "Send this error text to Claude")

    if rebuild:
        print("5. Data for rebuilding results (data/raw/)")
        raw = ROOT / "data" / "raw"
        for f, hint in [("train.csv", "from archive.zip (Jigsaw)"), ("test.csv", "from archive.zip (Jigsaw)"),
                        ("test_labels.csv", "from archive.zip (Jigsaw)"),
                        ("davidson.csv", "rename 'Hate Speech and Offensive Language Dataset.csv'"),
                        ("toxic_comment_moderation.xlsx", "rename 'Toxic Comment Moderation Dataset.xlsx'"),
                        ("combined_hate_speech.csv", "rename 'Combined Hate Speech Dataset.csv'")]:
            report((raw / f).exists(), f"data/raw/{f}", f"Copy it into data\\raw\\ - {hint}")

    print("\nRESULT: " + ("everything needed is in place." if ok_all else "fix the [FAIL] lines above, then run this check again."))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
