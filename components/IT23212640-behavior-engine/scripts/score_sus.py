"""Score the System Usability Scale (SUS) and task results from the moderator walkthrough.

    python scripts/score_sus.py docs/thesis/USER_STUDY_KIT/sus_results_template.csv

CSV columns: participant, q1..q10 (1-5), task1_success..task4_success (1/0),
task1_time_s..task4_time_s (seconds), comments.
SUS (Brooke, 1996): odd items contribute (answer - 1), even items (5 - answer); sum x 2.5 -> 0..100.
Prints per-participant SUS, mean, SD, 95% CI (t distribution) and per-task success rate / median time.
Rows with any missing SUS answer are skipped with a warning.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

ITEMS = [f"q{i}" for i in range(1, 11)]
TASKS = [f"task{i}" for i in range(1, 5)]


def sus_score(answers: list[int]) -> float:
    if len(answers) != 10:
        raise ValueError("SUS needs exactly 10 answers")
    if any(a not in (1, 2, 3, 4, 5) for a in answers):
        raise ValueError(f"SUS answers must be integers 1-5, got {answers}")
    total = sum((a - 1) if i % 2 == 0 else (5 - a) for i, a in enumerate(answers))  # i=0 is item 1 (odd)
    return total * 2.5


def t_crit_95(df: int) -> float:
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, df))
    except ImportError:  # normal approximation if scipy is missing
        return 1.96


def summarise(rows: list[dict]) -> dict:
    scores, skipped = {}, []
    for r in rows:
        try:
            scores[r["participant"]] = sus_score([int(float(r[q])) for q in ITEMS])
        except (KeyError, ValueError):
            skipped.append(r.get("participant", "?"))
    vals = list(scores.values())
    n = len(vals)
    mean = sum(vals) / n if n else float("nan")
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else float("nan")
    half = t_crit_95(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    tasks = {}
    for t in TASKS:
        succ = [int(float(r[f"{t}_success"])) for r in rows if str(r.get(f"{t}_success", "")).strip() != ""]
        times = sorted(float(r[f"{t}_time_s"]) for r in rows if str(r.get(f"{t}_time_s", "")).strip() != "")
        med = (times[len(times) // 2] if len(times) % 2 else (times[len(times) // 2 - 1] + times[len(times) // 2]) / 2) if times else None
        tasks[t] = {"n": len(succ), "success_rate": (sum(succ) / len(succ)) if succ else None, "median_time_s": med}
    return {"per_participant": scores, "skipped": skipped, "n": n, "mean": mean, "sd": sd,
            "ci95": (mean - half, mean + half), "tasks": tasks}


def main(path: str) -> dict:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    s = summarise(rows)
    print(f"SUS results from {Path(path).name}\n")
    print("participant   SUS")
    for p, v in s["per_participant"].items():
        print(f"{p:<12} {v:6.1f}")
    if s["skipped"]:
        print(f"\nskipped (missing or invalid answers): {', '.join(s['skipped'])}")
    print(f"\nn = {s['n']}   mean = {s['mean']:.1f}   SD = {s['sd']:.1f}   "
          f"95% CI = [{s['ci95'][0]:.1f}, {s['ci95'][1]:.1f}]   target >= 70: "
          f"{'met' if s['n'] and s['mean'] >= 70 else 'not met'}")
    print("\ntask    n   success   median time (s)")
    for t, v in s["tasks"].items():
        sr = "-" if v["success_rate"] is None else f"{v['success_rate']:.0%}"
        mt = "-" if v["median_time_s"] is None else f"{v['median_time_s']:.0f}"
        print(f"{t:<7} {v['n']:<3} {sr:>7}   {mt:>6}")
    return s


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python scripts/score_sus.py <results.csv>")
    main(sys.argv[1])
