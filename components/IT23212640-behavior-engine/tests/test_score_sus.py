"""SUS scoring (scripts/score_sus.py): formula, summary statistics, CSV handling."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from score_sus import main, sus_score, summarise  # noqa: E402

HEADER = ("participant,q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,task1_success,task2_success,task3_success,task4_success,"
          "task1_time_s,task2_time_s,task3_time_s,task4_time_s,comments\n")


def test_sus_formula_extremes_and_neutral():
    assert sus_score([5, 1] * 5) == 100.0   # best possible
    assert sus_score([1, 5] * 5) == 0.0     # worst possible
    assert sus_score([3] * 10) == 50.0      # all neutral
    # odd items: answer-1, even items: 5-answer -> (3+3+3+3+3)+(4+4+4+4+4)=35 -> 87.5
    assert sus_score([4, 1] * 5) == 87.5


def test_sus_rejects_bad_input():
    with pytest.raises(ValueError):
        sus_score([3] * 9)
    with pytest.raises(ValueError):
        sus_score([0, 3, 3, 3, 3, 3, 3, 3, 3, 3])


def test_summary_mean_sd_ci_and_tasks():
    rows = [{"participant": "M01", **{f"q{i}": a for i, a in enumerate([5, 1] * 5, 1)},
             "task1_success": "1", "task1_time_s": "30"},
            {"participant": "M02", **{f"q{i}": 3 for i in range(1, 11)},
             "task1_success": "0", "task1_time_s": "90"}]
    s = summarise(rows)
    assert s["per_participant"] == {"M01": 100.0, "M02": 50.0}
    assert s["mean"] == 75.0
    assert s["sd"] == pytest.approx(35.3553, abs=1e-3)
    lo, hi = s["ci95"]
    assert lo < 75.0 < hi
    assert s["tasks"]["task1"] == {"n": 2, "success_rate": 0.5, "median_time_s": 60.0}
    assert s["tasks"]["task2"]["success_rate"] is None


def test_csv_with_missing_answer_is_skipped(tmp_path, capsys):
    p = tmp_path / "sus.csv"
    p.write_text(HEADER + "M01,4,2,4,2,4,2,4,2,4,2,1,1,1,0,40,60,80,50,ok\n"
                          "M02,4,2,,2,4,2,4,2,4,2,1,1,0,1,45,70,95,55,\n", encoding="utf-8")
    s = main(str(p))
    assert s["n"] == 1 and s["skipped"] == ["M02"]
    assert s["per_participant"]["M01"] == 75.0
    assert "skipped" in capsys.readouterr().out


def test_template_csv_parses():
    tpl = ROOT / "docs" / "thesis" / "USER_STUDY_KIT" / "sus_results_template.csv"
    if not tpl.exists():
        pytest.skip("template not present")
    s = main(str(tpl))
    assert s["n"] >= 0
