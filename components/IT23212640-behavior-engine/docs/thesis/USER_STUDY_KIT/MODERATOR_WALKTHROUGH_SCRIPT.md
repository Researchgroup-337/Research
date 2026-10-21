# Moderator Walkthrough Script: Behavioral Analytics Engine Dashboard

Usability evaluation for J26-CS-337 (component IT23212640). Target: System Usability Scale (SUS) mean ≥ 70.
Run only after SLIIT ethics clearance has been granted.

## 0. Before the participant arrives (researcher only)

1. Start the API with a clean database so every participant sees the same data:
   ```
   del data\bae.db*            (Windows; on macOS/Linux: rm data/bae.db*)
   python -m uvicorn bae.api:app --app-dir src --host 127.0.0.1 --port 8000
   ```
2. In a second terminal, load the demo traffic. It replays the same messages from the synthetic test world every time (about 8 minutes for 14 days):
   ```
   python scripts\demo_stream.py --days 14 --api http://127.0.0.1:8000
   ```
3. Before the first session, note the correct answers for Tasks 1–4 on the answer sheet (section 5) by using the dashboard yourself.
   The replay is deterministic, and pseudonyms come from the fixed key in `models/.secret_key`, so the answers are the same for every participant as long as step 1 and step 2 are repeated exactly.
4. Open `http://127.0.0.1:8000/dashboard` in a browser, but do **not** enter the key yet.
5. Have ready: printed information sheet and consent form, SUS questionnaire (`SUS_QUESTIONNAIRE.html`, printed), stopwatch, this script, and a blank row in `sus_results_template.csv` for the participant code (M01, M02, ...).

## 1. Welcome and consent (about 3 min)

> "Thank you for taking part. Today you will try a moderator dashboard that helps detect users who repeatedly harass the same person in live chat. We are testing the **dashboard**, not you. There are no wrong answers, and you can stop at any time without giving a reason.
> The dashboard shows offensive language from public research datasets. All user names are pseudonyms and no real people are involved.
> I will ask you to do four short tasks and to think aloud while you work. Afterwards you will fill in a 10-question form. The whole session takes about 20 minutes."

- Confirm that the consent form is signed. Assign a participant code (M01...). Do **not** record the participant's name on any results sheet.

## 2. Short orientation (about 2 min, read exactly)

> "Each row in the table is a pair: one user (the *offender*) and the person they are writing to (the *target*). The *tier* runs from Low to Critical and shows how serious and persistent the behaviour towards that target is. *Hostile msgs* and *Sessions* show how often and across how many separate live streams the hostility happened. The *alert feed* lists events such as a tier going up, or several different users attacking the same person. Please enter the key `dev-key-change-me` and press Connect."

Do not explain anything further. If the participant asks for help during a task, reply: *"What would you try?"* Only give a hint after 60 seconds of no progress, and record that the hint was given.

## 3. Tasks (start the stopwatch when you finish reading each task; stop it when the participant states an answer)

| # | Read to the participant | Success criterion | Time limit |
|---|---|---|---|
| **T1** | "Find the user you think is the **most dangerous** right now, and tell me why." | Names an offender from the top row of *Highest-risk pairs* (highest tier and risk), and justifies it using tier, hostile messages or sessions. | 3 min |
| **T2** | "For that user, how many **hostile messages** have they sent to their target, and across how many **sessions**?" | Reads both numbers correctly from the same row. | 2 min |
| **T3** | "Find a **coordinated-harassment** alert in the alert feed. Explain in your own words what it means." | Finds the alert and explains that several different users (three or more) sent hostile messages to the same target within a few days. | 3 min |
| **T4** | "Find the **most recent** case where a pair went from **Medium to High**." | Identifies the newest *tier escalation* alert from Medium to High (correct offender and target). | 3 min |

Record for each task: success (1 = correct without a hint, 0 = wrong, gave up, or needed a hint), time in seconds, and any comments made while thinking aloud.

Optional probe after T1 (do not time it): if a pair shows the flag `explicit_threat_override`, ask *"What do you think this flag means?"* and note the answer. This checks whether the safety override is understandable.

## 4. SUS questionnaire and debrief (about 5 min)

1. Hand over the printed SUS questionnaire. Say: *"Please answer quickly. Your first impression is what we want. Tick one box per line."*
2. Check that all 10 items are answered. Do not influence the answers.
3. Debrief:
   > "Thank you. The data you saw was simulated, built from public research datasets, and no real people were harassed. If anything you saw upset you, SLIIT student counselling is available [insert contact]. You may withdraw your responses until [date] by quoting your participant code."
4. Enter the answers into `sus_results_template.csv` (q1–q10 as 1–5, task success as 1/0, task times in seconds).

## 5. Answer sheet (researcher fills in before the first session)

| Task | Correct answer for this seeded dataset |
|---|---|
| T1 | offender: __________  target: __________  tier: ______ |
| T2 | hostile messages: ______  sessions: ______ |
| T3 | alert target: __________ |
| T4 | offender: __________  target: __________  time: ______ |

## 6. After all sessions

```
python scripts/score_sus.py docs/thesis/USER_STUDY_KIT/sus_results_template.csv
```
This prints the SUS score for each participant, the mean, SD and 95% confidence interval, and the success rate and median time for each task.
Report the mean SUS against the target of 70 together with the confidence interval. With fewer than 10 participants, state that the interval is wide.
