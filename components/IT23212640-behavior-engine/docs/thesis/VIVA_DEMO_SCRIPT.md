# Viva Demo Script (5–7 minutes)

Jayarathna W.D.O.B. (IT23212640): Behavioral Analytics Engine, J26-CS-337

## Before the viva (do this 30 minutes early)

```
# Terminal 1: dashboard API (fresh database)
del data\bae.db*
python -m uvicorn bae.api:app --app-dir src --host 127.0.0.1 --port 8000

# Terminal 2: SafeChat Lab
$env:BAE_RESEARCHER_KEY="lab-key-change-me"
python -m uvicorn bae.chatapp.server:app --app-dir src --host 0.0.0.0 --port 8100

# Terminal 3: pre-load bot sessions so a harasser has history (about 8 min)
python scripts\run_chat_bots.py --sessions 6 --minutes 1
```

Open these browser tabs:
1. `http://127.0.0.1:8000/dashboard` (key `dev-key-change-me`)
2. `http://127.0.0.1:8100/researcher` (key `lab-key-change-me`)
3. `reports/EVALUATION_REPORT.md`
4. `reports/figures/risk_timeline_example.png`

Backup if anything fails live: screenshots in `reports/figures/` (`dashboard_light.png`, `safechat_researcher_feed.png`, `safechat_researcher_pairs.png`).

---

## 0:00–0:30 One-sentence pitch

> "This is a privacy-preserving behaviour analytics engine for live chat. Instead of judging each message on its own, it tracks whether the *same person keeps targeting the same victim across sessions*, and turns that into a real-time Low-to-Critical risk tier for moderators. Everything runs locally, and no message text is stored."

## 0:30–1:30 Why this is cybersecurity

- **UEBA (User and Entity Behaviour Analytics):** the same idea SOCs use to detect insider threats. Baseline behaviour, decay old evidence, alert on persistent anomalies. Here the entities are offender → target pairs.
- **Threat-actor persistence:** a harasser is a persistent threat actor. The Target Persistence Index measures decayed volume, session recurrence and recency, much as threat intelligence tracks campaigns over time.
- **Privacy by design:** HMAC-SHA256 pseudonyms, no message text stored, SQLCipher encryption at rest, retention purge, right to erasure (PDPA No. 9 of 2022). All of these are tested.
- **Adversarial evasion:** obfuscation attacks (leetspeak, homoglyphs, spacing) cut recall to about 0.58; normalisation restores about 0.85. Account rotation is handled by a victim-centric alert.
- **STRIDE:** API key auth, input validation, rate limiting against floods, audit trail of alerts, local-only binding. The STRIDE table is in the README.

## 1:30–3:00 Live demo 1: moderator dashboard

1. In **Terminal 4**: `python scripts\demo_stream.py --days 3 --api http://127.0.0.1:8000`
2. Point to the terminal: tier escalations and **COORDINATED HARASSMENT** lines appear in real time, labelled with ground truth.
3. Switch to the **dashboard** tab and press Connect:
   - *Highest-risk pairs*: a Critical pair with many hostile messages across several sessions. "That is persistence."
   - Find a pair with the flag **`explicit_threat_override`**: "This pair has only one hostile message, but it was a credible threat. The safety override holds it at High for 24 hours, so a moderator sees it immediately, without waiting for repetition."
   - *Alert feed*: show a coordinated-harassment alert. "Three different accounts hit the same victim within 7 days. This is our answer to sock-puppet account rotation."
4. When the demo finishes, read out the last line: **latency p50 / p95 in milliseconds** (target 250 ms).

## 3:00–4:30 Live demo 2: SafeChat Lab

1. Researcher console → **Live feed**: messages with T, resolved target and tier as they happen.
2. **Pairs & labels**: show the role-play harasser bot's pair climbing to **Critical** over the sessions, while the **burst** bot (one bad session, then quiet) has **decayed** back to Low. Then open `risk_timeline_example.png`: "This is the same behaviour on the synthetic test world: persistence raises risk, a burst decays."
3. Show the **role cards**: the ground truth comes from assigned roles, and harassment only ever targets bots or my own persona.
4. **Export**: click *Download dataset*, then open the zip and show `participants.csv` (codes only, no display names). Point out the *without message text* option for sharing.

## 4:30–5:30 The three numbers to say

1. **F1 0.859** (precision 0.910, recall 0.812) on repeat-harassment pairs in the held-out test world, against **0.070** for a content-only classifier.
2. **PR-AUC 0.379 → 0.947**: content alone versus content plus persistence. Behaviour, not text, carries the signal.
3. **About 10 ms per message** end to end on a laptop CPU (p95 9.89 ms), against a 250 ms target, with DistilBERT running locally.

## 5:30–7:00 Questions (see below)

---

## Likely examiner questions and short, honest answers

**1. "Your behaviour data is synthetic. Why should we trust it?"**
No public corpus records who targeted whom across sessions with ground truth, so I had to build one. To limit circularity, the message texts are real, the generator does not use my TPI formula, the coefficients are fitted rather than hand-set, and I included hard negatives (banter, bursts, reformed users, roaming trolls) and a hard positive (subtle harassers), plus target-resolution noise. Tests use a separate text pool. Synthetic results show the method works on the behaviours I modelled. SafeChat Lab exists to validate it with real people; the bot pilot is done and the human study is waiting for ethics approval.

**2. "Why is γ (escalation) zero?"**
The fitted risk model is constrained to non-negative weights for security reasons: more escalation must never lower risk. On this data, escalation velocity added no information beyond decayed volume and recency, so the fit set γ to 0, and the ablation confirms that removing v changes nothing (PR-AUC 0.947 either way). I report it as a finding. It may reflect how my generator models escalation, so I list it as future work. v is still shown to moderators as evidence.

**3. "What if the attacker switches accounts?"**
Then per-account persistence fails completely: pair-level recall is 0 in my rotation test, and I report that. The victim-centric coordinated-harassment alert catches 100% of victims in that test, at the cost of alerting on 12% of other targets. Proper account linking is future work, and it has privacy implications of its own.

**4. "Why not use a cloud moderation API such as Perspective?"**
It would send raw chat text to a third party, which contradicts the privacy-preserving goal. Perspective API is also being shut down at the end of 2026. Local DistilBERT keeps text on the device, runs in about 7 ms on CPU, and matches the TAF (decision D1).

**5. "Your content-model F1 on Jigsaw is 0.71, below your 0.82 target."**
Correct. I report that target as partly met. On Davidson tweets, which are closer to live chat, F1 is 0.898. Jigsaw labels are noisy and threats are rare (211 of 63,978), while ranking quality is high (ROC-AUC 0.984). The engine-level targets are met because persistence, not single-message accuracy, drives detection.

**6. "Why did the false-alarm rate rise from 3.0% to 7.5% with DistilBERT?"**
Almost all of it comes from the explicit-threat override: 607 of 696 false-alarm pairs on the test world. Of those, 452 contain a message that human annotators labelled as a threat, so they are false only with respect to *repeat* harassment, not as safety alerts. Raising the threshold reduces alerts but drops safety coverage (65.9% → 38.8% at 0.99 on the validation world). I tuned this on validation data only and recommend showing threat alerts separately.

**7. "How do you know the engine isn't just counting messages?"**
The frequency rule (three strikes) has PR-AUC 0.907 but precision 0.117. It flags 54% of banter pairs, 59% of burst attackers and 63% of reformed users, while the engine flags at most 1% of each. Decay and session structure make the difference: removing decay drops PR-AUC to 0.914.

**8. "What about subtle harassers?"**
They are the weakest case: recall 0.60, against 0.93 for overt harassers. Their messages mostly fall below the hostility threshold, so evidence builds slowly. A graded contribution from mild messages is my proposed fix.

**9. "Is it really privacy-preserving? Show me."**
User IDs are HMAC-SHA256 pseudonyms with a secret key. The store holds no message text, only scores, counters and timestamps. There is SQLCipher encryption at rest (where available), a retention purge and an erasure endpoint, and each has a unit test (for example `test_no_message_text_stored` and `test_right_to_erasure`). The API binds to localhost and requires a key.

**10. "Could an attacker game the risk model?"**
The monotone fit ensures more hostility, repetition or escalation can never lower risk. Text obfuscation is countered by normalisation (E6). Flooding is rate-limited. Spreading a threat across accounts triggers the victim alert. Remaining gaps are subtle, low-level harassment and accounts rotated slowly enough to avoid the 7-day victim window. I say so in the limitations section.
