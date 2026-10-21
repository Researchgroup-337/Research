"""Type messages and see how DistilBERT and the behaviour engine react. Type 'quit' to stop."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bae.engine import BehavioralEngine

eng = BehavioralEngine()                      # uses DistilBERT automatically
print("Content model:", eng.scorer.name)
print("Format:  sender > target > message      e.g.  kasun > nimal > I will find you")
print("Type 'next' to start a new session (2 days later). 'quit' to stop.\n")
session, day = 1, 0
while True:
    line = input(f"[session {session}] ").strip()
    if line == "quit": break
    if line == "next":
        session += 1; day += 2; print(f"--- session {session} (day {day}) ---"); continue
    try:
        sender, target, text = [p.strip() for p in line.split(">", 2)]
    except ValueError:
        print("Use: sender > target > message"); continue
    r = eng.process({"message_id": str(time.time()), "session_id": f"s{session}", "sender_id": sender,
                     "text": f"@{target} {text}", "timestamp": 1_780_000_000 + day * 86400 + time.time() % 3600,
                     "channel_owner": "streamer"})
    m, p = r["message"], r["pair"]
    labels = ", ".join(f"{k}={v:.2f}" for k, v in m["labels"].items() if v >= 0.3) or "none"
    print(f"  T={m['T']:.2f} hostile={m['hostile']}  labels: {labels}")
    print(f"  {sender}->{target}: tier={p['tier']}  risk={p['risk']:.2f}  hostile msgs={p['hostile_messages']}  "
          f"sessions={p['sessions_targeted']}  flags={r['flags']}  ({r['latency_ms']:.0f} ms)\n")