"""Export SafeChat Lab session reports (built only from engine outputs) from the research database.

    python scripts/session_report.py --session room1_s003
    python scripts/session_report.py --session room1_s003 --format pdf --mask-text --codes-only
    python scripts/session_report.py --all --format html --out reports/sessions/
    python scripts/session_report.py --session room1_s003 --include-truth      # adds the researcher appendix

Reads data/safechat/research.db (or SAFECHAT_DB / --db). Safe to run while the chat server is running.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bae.chatapp import session_report  # noqa: E402
from bae.chatapp.research_db import ResearchDB  # noqa: E402
from bae.config import EngineConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--session", help="session id, e.g. room1_s003")
    g.add_argument("--all", action="store_true", help="one report per session")
    ap.add_argument("--format", choices=["html", "pdf", "csv"], default="html")
    ap.add_argument("--mask-text", action="store_true", help="replace message text with [hidden]")
    ap.add_argument("--include-truth", action="store_true", help="add the researcher appendix (roles and labels)")
    ap.add_argument("--codes-only", action="store_true", help="show study codes only (no display names)")
    ap.add_argument("--out", default=str(ROOT / "reports" / "sessions"))
    ap.add_argument("--db", default=os.environ.get("SAFECHAT_DB", str(ROOT / "data" / "safechat" / "research.db")))
    ap.add_argument("--content-model", help="name shown in the header (default: the model the engine loads)")
    a = ap.parse_args()

    if not Path(a.db).exists():
        sys.exit(f"research database not found: {a.db}")
    rdb = ResearchDB(a.db)
    cfg = EngineConfig.load()
    model = a.content_model
    if not model:
        from bae.content_model import load_scorer
        model = load_scorer("auto").name
    ids = [s["session_id"] for s in rdb.sessions()] if a.all else [a.session]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for sid in ids:
        try:
            rep = session_report.build_report(rdb, sid, cfg, content_model=model, mask_text=a.mask_text,
                                              include_truth=a.include_truth, codes_only=a.codes_only)
        except KeyError:
            sys.exit(f"unknown session {sid}. Sessions: " + ", ".join(s["session_id"] for s in rdb.sessions()))
        if a.format == "html":
            path = out / f"{sid}.html"
            path.write_text(session_report.render_html(rep), encoding="utf-8")
        elif a.format == "pdf":
            path = out / f"{sid}.pdf"
            path.write_bytes(session_report.render_pdf(rep))
        else:
            path = out / f"{sid}_csv.zip"
            path.write_bytes(session_report.render_csv_zip(rep))
        k = rep["kpi"]
        print(f"{sid}: {k['messages']} msgs, {k['hostile']} hostile, repeat harassers {k['repeat_harassers']}, "
              f"trolls {k['trolls']}, bursts {k['burst_attackers']}, threats {k['explicit_threats']} -> {path}")


if __name__ == "__main__":
    main()
