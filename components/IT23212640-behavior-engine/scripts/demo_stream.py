"""Demo: replay part of the synthetic TEST world (real message text, @mentions) through the
FULL engine path - text normalisation, content model, target resolver, pseudonymisation,
encrypted store - and print every tier escalation as it happens.

    python scripts/demo_stream.py                  # in-process
    python scripts/demo_stream.py --api http://127.0.0.1:8000 --key dev-key-change-me   # through the REST API
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bae.config import DATA_PROCESSED  # noqa: E402


def events(n_days: int):
    df = pd.read_parquet(DATA_PROCESSED / "synthetic_test.parquet")
    df = df[df.ts <= df.ts.min() + n_days * 86400]
    for r in df.itertuples():
        yield {"message_id": r.message_id, "platform": "youtube", "session_id": r.session_id,
               "channel_owner": r.channel, "sender_id": r.sender, "text": r.text, "timestamp": float(r.ts)}, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--api")
    ap.add_argument("--key", default="dev-key-change-me")
    a = ap.parse_args()
    t0 = time.time()
    n = esc = 0
    if a.api:
        import httpx
        client = httpx.Client(base_url=a.api, headers={"X-API-Key": a.key}, timeout=30)

        def send(ev):
            while True:  # respect the server's rate limit (HTTP 429) with a short back-off
                r = client.post("/v1/score", json=ev)
                if r.status_code != 429:
                    return r.raise_for_status().json()
                time.sleep(0.05)
    else:
        from bae.engine import BehavioralEngine
        db = Path(tempfile.mkdtemp()) / "demo.db"
        eng = BehavioralEngine(db_path=str(db), encryption_key="demo-db-key")
        print(f"content model: {eng.scorer.name} | encrypted store: {eng.store.encrypted} | db: {db}")
        send = eng.process
    lat = []
    for ev, truth in events(a.days):
        rec = send(ev)
        n += 1
        lat.append(rec["latency_ms"])
        for al in rec["alerts"]:
            esc += 1
            if al["kind"] == "tier_escalation" and al["to"] in ("High", "Critical"):
                day = (ev["timestamp"] - 1782844200) / 86400
                print(f"day {day:5.1f}  {truth.sender:>12s} -> {truth.target:<14s} {al['from']:>6s} -> {al['to']:<8s} "
                      f"risk={rec['pair']['risk']:.2f} hostile={rec['pair']['hostile_messages']:>2d} "
                      f"sessions={rec['pair']['sessions_targeted']}  [truth: {truth.persona}]")
            elif al["kind"] == "coordinated_harassment":
                print(f"           COORDINATED HARASSMENT on {truth.target} ({al['senders']} hostile senders in 7 days)")
    lat = pd.Series(lat)
    print(f"\n{n:,} messages in {time.time()-t0:.1f}s | alerts={esc} | engine latency p50={lat.median():.2f} ms p95={lat.quantile(.95):.2f} ms")


if __name__ == "__main__":
    main()
