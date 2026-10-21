# SafeChat Lab: Cross-Session Data Collection Protocol

This document responds to the proposal-panel comment:

> *"Success depends on creating a test messaging application for session tracking behavior. Dataset availability is crucial for successful model development (especially for cross session detection)."*

## 1. What was built

**SafeChat Lab** (`src/bae/chatapp/`) is a working test live-chat application. Every message passes through the Behavioral Analytics Engine in real time, and everything is logged as a labelled cross-session dataset.

| Need raised by the panel | How SafeChat Lab meets it |
|---|---|
| A messaging app to test session tracking | Rooms work like live streams: the researcher starts and ends **sessions**, and participants keep the **same study code across sessions**. Messages travel over WebSockets, with @mentions and replies. |
| Session-level behaviour must be observable | Each message is logged with session ID, sender, resolved target, how the target was resolved, content score T and the engine's risk and tier at that moment. |
| A cross-session dataset (none exists publicly) | Role assignments give pair-level ground truth (e.g. P007 harasses persona "Nimal" across sessions). The researcher can also label pairs manually. **Export** produces CSVs in the same schema as the synthetic corpus. |
| Enough volume | **Bots** (`scripts/run_chat_bots.py`) join the same rooms alongside humans, giving a hybrid dataset: realistic human behaviour plus controlled volume. |
| Evaluation on the collected data | `scripts/evaluate_collected.py --export <zip>` reports the same metrics as the synthetic study: stateless vs frequency vs engine, streaming detection, and tiers by role. |

The three data sources then reinforce each other:

1. **Jigsaw / Davidson (real text, no sessions)** train and test the content model.
2. **The synthetic multi-session corpus (large, controlled)** fits the TPI and risk model and runs the ablations.
3. **The SafeChat Lab dataset (real humans, real sessions, smaller)** validates that the engine behaves the same with real people. This is the answer to the panel.

## 2. Study design

| Item | Plan |
|---|---|
| Participants | 12–20 volunteer SLIIT students (18+), recruited openly; no course credit tied to taking part |
| Bots | 16–19 scripted bots (normal, banter, harasser, subtle harasser, escalator, troll, burst) for volume and controlled cases |
| Sessions | **Option A (recommended, realistic):** 6–8 sessions of 15 min over 2–3 weeks, set `SAFECHAT_DAYS_PER_SESSION=0` so the engine uses real time. **Option B (one lab day):** 8 sessions of 10 min in one afternoon, with `SAFECHAT_DAYS_PER_SESSION=2` so each session is treated as a stream 2 days after the previous one. |
| Roles | About 25% of humans are role-play harassers (harasser / subtle_harasser / escalator); the rest are normal, bystander or banter. One burst and one troll role give hard negatives. Roles are given on a private **role card**. |
| Targets of harassment | **Bots or the researcher's persona only**, never another human participant (see section 4). |
| Writing hostile messages | Role-players may use the **message bank** buttons (real dataset messages) instead of writing abuse themselves. Bank use is logged (`from_bank`). |
| Blinding | Participants never see risk scores or tiers; only the researcher console shows them. |

## 3. Ground truth and labelling

1. **Role-derived labels (automatic):** a pair (sender, assigned target) is labelled 1 when the sender's role is harasser, subtle_harasser or escalator. Every other pair is 0.
2. **Manual review:** after collection, two annotators (you and a teammate) independently review each pair with more than 3 messages in the console (*Pairs & labels* tab) and set 1/0. Report **Cohen's κ** between annotators, then resolve disagreements by discussion. Manual labels override role labels in the export.
3. Record any role that was not followed (for example, a "harasser" who stayed polite). Manual review handles this.

## 4. Ethics and safety (include this in the ethics application)

- Get **SLIIT ethics clearance** before running sessions with people.
- **Informed consent** is shown and must be ticked before joining (built into the join page). Participants can withdraw at any time, and the researcher can delete their data.
- **No human is targeted.** Harassment roles target **bots** or the **researcher persona**, so no participant receives hostile messages aimed at them. Participants may still *see* offensive language, which the consent text states.
- **No real personal information:** participants use study tokens and invented display names. Display names are never exported; exports contain only codes (P001, B003).
- **Debrief** after the final session: explain the roles, show the role cards, and share support contacts (SLIIT student counselling).
- **Distress protocol:** any participant can end a session for everyone by asking the researcher, who presses *End session*.
- **Data handling:** the research database stays on the researcher's laptop; exports have no names. Use **Download without message text** for anything shared outside the team. Delete raw data after the project, per the SLIIT retention policy.

## 5. Running a session (checklist)

**Before the first session:**
1. `pip install -r requirements.txt`
2. Start the app on a laptop connected to the same Wi-Fi as participants:
   ```
   set BAE_RESEARCHER_KEY=choose-a-secret
   set SAFECHAT_DAYS_PER_SESSION=0          (or 2 for a one-day lab)
   uvicorn bae.chatapp.server:app --app-dir src --host 0.0.0.0 --port 8100
   ```
   (On macOS/Linux use `export` instead of `set`.)
3. Open `http://localhost:8100/researcher` and enter the key.
4. **Add participants**, one row per person, using an invented persona name. Hand each person their **token** on a slip of paper. Add yourself as a persona too, for example "Moderator".
5. Run the bots once to create them and the room: `python scripts/run_chat_bots.py --sessions 1 --minutes 1 --key choose-a-secret`.
6. **Assign roles**: in the participants table, choose a role and target, then click *Save role*. Harasser targets should be bots.

**Each session:**
1. Click *Start session* on the room, or let the bot script start it.
2. Participants open `http://<laptop-ip>:8100`, enter their token, consent, and choose *Lab Stream*.
3. For background traffic, run the bots for the same length:
   `python scripts/run_chat_bots.py --sessions 1 --minutes 15 --key choose-a-secret`
4. Watch the *Live feed*. Click *End session* when time is up. (Tick *Auto-save report when a session ends* first if you want the report saved automatically.)
5. **After each session:**
   - Open *Pairs & labels* and **review any label-check warnings** (yellow panel). Choose *Use role label* or *Keep manual label* (or *Mark reviewed*) for each one. Do not leave conflicts unreviewed: they change the ground truth.
   - **Export the session report** from the *Sessions* table (*PDF* or *CSV*; message text is hidden), or run `python scripts/session_report.py --session <id> --mask-text --codes-only` for anything shared outside the team. The report uses engine outputs only. Add `--include-truth` only for the researcher's own comparison.

**After the last session:**
1. Go to *Pairs & labels* and review and label the pairs (two annotators).
2. *Export* → *Download dataset*.
3. `python scripts/evaluate_collected.py --export safechat_dataset_XXXX.zip`. If DistilBERT is installed, add `--rescore`.
4. Use `reports/collected/COLLECTED_REPORT.md` in the thesis, next to the synthetic results.

## 6. How to report it

- Describe the dataset: participants (humans/bots), sessions, messages, pairs, positive pairs, and how targets were resolved (reply / mention / broadcast share).
- Compare the systems (stateless, frequency, engine) on the collected pairs. With a small number of positive pairs, report counts and confidence intervals rather than only percentages.
- State clearly that role-play is not the same as natural harassment. The collected data **validates** the synthetic findings; it does not replace a large natural corpus.

## 7. Pilot run (bots only, already done)

A bots-only pilot checked the full pipeline: live chat, then the engine, then the research log, export and `evaluate_collected.py`. It ran 18 bots for 10 sessions with the lab clock at 2 days per session, producing 1,024 messages and 63 pairs, of which 4 are harassment pairs. Results are in `reports/collected/` and the export is `data/safechat/pilot_bots_export.zip`.

| System | Precision | Recall | F1 | False-alarm pairs |
|---|---|---|---|---|
| Stateless classifier | 0.09 | 1.00 | 0.16 | 42 |
| Frequency rule | 0.19 | 1.00 | 0.32 | 17 |
| **Engine** | **0.57** | **1.00** | **0.73** | **7** |

All 4 harassers, including the subtle one, reached **Critical**. The burst attacker decayed to Medium and the troll stayed Low. The remaining false alarms are mainly the **banter** pair, where two friends trade rough language every session. This is a known hard case: the content model cannot tell friendly profanity from abuse. Reviewing banter is a good use of the manual-labelling step and a good discussion point for the thesis. With humans, expect noisier behaviour than bots. That is exactly why this data is worth collecting.
