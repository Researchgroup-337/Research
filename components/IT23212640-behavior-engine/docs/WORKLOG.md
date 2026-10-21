# Work Log: Behavioral Analytics Engine (IT23212640)

Running log of changes made while finishing the component. Newest entries are at the bottom of each day.

## 2026-10-18: DistilBERT switch-over (Windows)

- Confirmed `models/distilbert/` (thresholds.json, config, tokenizer.json, model.onnx, model.quant.onnx) and all six `data/raw/` files were already in place.
- `check_setup.py --rebuild`: every check `[ OK ]`; content model `distilbert_onnx`.
- Rebuilt every result with DistilBERT: `prepare_data.py`, `generate_synthetic.py`, `fit_risk_model.py`, `evaluate.py`.
- **Bug fix (`scripts/evaluate.py`, E7):** the latency experiment always ran a second pass with an SQLCipher-encrypted store. On Windows `sqlcipher3` cannot be installed, so `BehaviorStore` correctly refused the key and the whole evaluation crashed after about 23 minutes. E7 now skips the encrypted pass when `sqlcipher3` is not importable, and the report prints "not run (sqlcipher3 not installed)". No research logic changed.

## 2026-10-19

### Task 1: small fixes
- **1a `src/bae/engine.py`:** `st.flags = sorted(set(st.flags) | set(flags))[-10:]` was already in the file (line 114) when work started, so no edit was needed. The dashboard and API now show `explicit_threat_override` on pairs held at High by a single credible threat. Verified through the API: a first message "@nimal I know where you live, watch your back" returns tier High, risk 0.0053, flag `explicit_threat_override`.
- **1b `scripts/evaluate.py`:** the report header now says "Re-run `python scripts/evaluate.py` to regenerate every number." (the content-model name is still filled in from the run).

### Task 2: explicit-threat override analysis
- **2a `scripts/evaluate.py` E4:** added
  - an `engine_no_override` row: the same engine replayed on the test world with `threat_override_p = 1.01`, so the override never fires;
  - `override_only_false_alarms`: non-harasser pairs that reach High in the normal engine but never without the override, split by whether the triggering pair contains a message with a Jigsaw *threat* label (`bucket == "threat"`) or not.

  Because the override only changes the tier (never the stored counters or risk), the no-override run's High set is a subset of the normal run's, so the difference is exactly the set of alarms caused by the override.
- **2b `scripts/tune_override.py` (new):** replays **only the validation world** (`synthetic_val` + pairs) with the fitted risk model, sweeping `threat_override_p` in {0.90, 0.93, 0.95, 0.97, 0.99, never}. It also reports a *safety* column: the share of pairs that received a Jigsaw-labelled threat message and were at ≥ High within the 24 h hold. Output: `reports/override_tuning.json`, `reports/override_tuning.md`. The test worlds are not read.
- **Finding:** at 0.90, 500 of 618 validation false alarms are override-only, but 386 of those 500 pairs actually received a Jigsaw-labelled threat message. They are "false" only with respect to the repeat-harassment label, not as safety alerts. Raising the threshold to 0.99 cuts false alarms to 3.98% but drops safety coverage from 65.9% to 38.8%. See the table in `reports/override_tuning.md`.
- **2c:** recommendation sent to the author. `config.py` is **unchanged** until approved.
- **2d E8 check (no bug):** E8 does use `distilbert_onnx`. DistilBERT marks 156 of the 1,050 Toxic Comment Moderation comments hostile versus 122 for TF-IDF, with T correlation 0.93. The tier outcome is nevertheless identical (Low 305, High 3; Spearman ρ = 0.1593) because:
  1. The comments span 2021-03-10 to 2024-12-24 across 5 platforms. With a 30-day half-life the decayed counters never build up: at most 4 hostile messages per pair, and maximum pair risk is 0.0055 for both models, far below the Medium boundary of 0.461.
  2. The only non-Low users (U009, U025, U049) are held at High **solely by the explicit-threat override**. Both models give those threat messages P(threat) ≥ 0.9. With the override disabled, both models give 308/308 users Low.

  E8 therefore tests the override rather than persistence on this dataset. It should be presented as a sanity check only, as its note already says.

### Full evaluation re-run (2026-10-19)
- `python scripts/evaluate.py` re-run with the new E4 rows (about 22 min). All earlier numbers are unchanged. New test-world results:
  - `engine_no_override`: harassers detected 91.5%, median 6 sessions, false-alarm rate 0.96% (89 pairs).
  - Normal engine: 607 of its 696 false-alarm pairs are override-only; 452 of those contain a Jigsaw-labelled threat message and 155 do not.
- SafeChat pilot re-scored with DistilBERT: `python scripts/evaluate_collected.py --export data/safechat/pilot_bots_export.zip --rescore`. The previous ("as logged") report is kept in `reports/collected_pilot_as_logged/`. The engine result is the same (P 0.571, R 1.0, F1 0.727). Streaming false-alarm pairs go from 7 to 8 (stateless 43, frequency 18), and the burst bot's pairs now end at Low/Low instead of Low/Medium.
- `docs/DATA_COLLECTION_PROTOCOL.md` section 7 and the README "Headline results" still show the earlier (TF-IDF / as-logged) numbers. They were not edited; update them if wanted.
- `DECISIONS.md` still contains TF-IDF-era numbers: D6 (ablation 0.94 → 0.90; now 0.947 → 0.914), D8 (validation PR-AUC 0.959; now 0.968), D12 (7.8% of other targets; now 12.0%) and D14 (recall 0.56–0.64 → about 0.75 with TF-IDF; DistilBERT 0.58–0.72 → 0.77–0.86). Not changed without approval.

### Task 4: documents
- `scripts/score_sus.py` (new) with `tests/test_score_sus.py` (5 tests: formula extremes, input validation, mean/SD/CI, missing-answer handling, template parse).
- Verified that the DistilBERT tokenizer already strips zero-width characters: "i w\u200bi\u200bl\u200bl ..." gives the same token IDs as "i will ...". This explains why E6 zero-width recall is identical with and without normalisation.
- `docs/thesis/`: see the files listed in the final summary.

### Task 3: GitHub backup
- Pushed branch `it23212640-behavior-engine` to https://github.com/oshith555/Reasearch (92 files; `model.quant.onnx` and `tfidf_lr.joblib` via Git LFS). The repo previously held only the initial README commit; nothing existing was changed.
- Verified from a fresh clone of the branch: `git lfs pull` restored `model.quant.onnx` at its full size (67,361,410 bytes). With `data/raw` copied in, `check_setup.py` gave all `[ OK ]` (`distilbert_onnx`, 5.1 ms/message) and `pytest` gave 37 passed, 1 skipped.

### Hands-on checks (try_model, dashboard, SafeChat)
- `scripts/try_model.py` (author's script): friendly messages score T 0.00; insults 0.71–0.76; evasion ("y0u 4re p4thetic") gives the same T as clean text (0.76); "i will k i l l you" gives threat 0.99 and the override fires.
  - **Not as the guide expected:** "I know where you live" alone scores threat 0.21 (0.01 with an @mention), so the override does **not** fire. "I know where you live, watch your back" scores 0.94.
  - **Not as the guide expected:** with one hostile message per session, the tier stays Low for 7 sessions (Medium at session 8). With 3 per session it reaches Medium at session 3, High at 4 and Critical at 5. With 5 per session: Medium at 2, Critical at 3. This follows from the fitted model (decayed volume dominates).
- Dashboard: 14-day demo through the API on a fresh DB (`data/bae_demo.db`): 22,487 messages, 371 alerts, 71 coordinated-harassment lines, latency p50 17.57 ms / p95 25.85 ms. Final tiers: Critical 15, High 136, Medium 5, Low 6,097.
- SafeChat Lab: scripted participant "Tester" (P019) sent 3 insults per session to @Nimal. Tier after sessions 1–4: Low (0.05), Low (0.23), Medium (0.49), High (0.69). Targets were resolved by @mention.

### SafeChat Lab improvements (Parts A–C)
- **A. Researcher console tables (`static/researcher.html`):**
  - every table uses `table-layout:fixed` with a `<colgroup>`; numeric headers and cells are right-aligned, badges centred and text left-aligned
  - sticky headers inside a scrolling card; horizontal scroll on narrow screens; messages wrap in their own column
  - Pairs & labels has the specified columns and tooltips and a centred label with a "role"/"manual" tag
  - search plus click-to-sort on Pairs (risk descending by default) and Live feed (newest first); the 3-second refresh re-renders only the table bodies, so sort, filter and scroll survive
  - Participants is now full width, and Rooms : Sessions share the row 5 : 7 so all columns fit at 1300 px
  - screenshots in `reports/figures/safechat_console/` (light/dark × 1300/390, every tab, plus the sample report)
- **B. Label integrity:**
  - `ResearchDB.assign` refuses harasser / subtle_harasser / escalator / burst / banter without a target (HTTP 422), and the console checks this before saving
  - new `label_reviews` table and `ResearchDB.label_conflicts()`; `GET /api/admin/label-check` and `POST /api/admin/label-check/resolve` (`use_role` removes the manual label; `keep_manual` records the review)
  - an extra conflict kind, `role_missing_target`, catches positive roles saved without a target before this fix
  - yellow panel in Pairs & labels; `pairs.csv` gains `label_review`, plus a new `label_reviews.csv`
  - On the real lab database the check found 4 conflicts: at 09:15 on 2026-10-19, B010, B011, B012 and B013 were re-saved with target B001. Harasser B013's real pair B013 → B007 therefore silently lost its label 1. **Left for the researcher to review; nothing changed.**
- **C. Session reports:**
  - new `src/bae/chatapp/session_report.py`: categories with named thresholds, KPIs, 6 matplotlib charts embedded as base64 (Okabe-Ito colours, hatching for hostile, values on bars and in heatmap cells), tables, masked evidence, footer, and the optional researcher appendix
  - output as HTML, PDF (matplotlib `PdfPages`, no new dependency) or a CSV zip
  - `GET /api/admin/sessions/{id}/report`; `POST .../rooms/{rid}/end?save_report=true` writes `data/safechat/reports/<id>.html`
  - `scripts/session_report.py` (with `--all`); Report column and auto-save checkbox in the console
  - The message log now also stores the engine's `hostile`, `pair_hostile`, `pair_sessions` and `flags` (new nullable columns, added automatically to existing databases). Older rows are completed with the engine's own rule from `risk.update_pair`.
- Tests: `tests/test_session_report.py` (9 tests: known scenario categories and KPIs, escalation vs previous session, mask_text/codes_only/PDF/CSV, no ground-truth reads without include_truth, appendix, endpoint auth and formats, auto-save, Save role validation, label-check conflicts and resolution and export).
- DECISIONS.md: added D19 and D20 (existing rows unchanged). README SafeChat section and protocol section 5 updated.

### Merge of two colliding SafeChat change sets
- **What happened:** after change set 1 (Parts A–C above, commit `da12e03`) a second set of changes ("Overall status", from another assistant) was copied by hand into `server.py` and `research_db.py`. Those copies were based on the files from *before* set 1, so they silently removed set 1 from both files: label-check, Save-role target validation, session-report endpoints, auto-save, the engine-output log columns and the review columns in the export. The set 2 version of `researcher.html` was never copied, so that file still had set 1 only.
- **Merge:** started from `da12e03` (set 1) and re-applied set 2 on top.
  - `research_db.py`: `user_tier TEXT` and `user_risk REAL` were added to the existing migration list (`ENGINE_COLUMNS`) and to `log()`. These were the only overlapping lines; both column lists were combined.
  - `server.py`: set 2 re-applied unchanged (`user_tier`/`user_risk` logged from `user_max_tier`/`user_max_risk`; the feed selects them; `/pairs` adds `sender_overall_tier`, `sender_overall_risk`, `sender_targets_high`; `messages.csv` gains `user_tier`, `user_risk`).
  - `researcher.html`: set 2 UI rebuilt from the description on the set 1 layout. Participants "Now" became "Overall status". Pairs "Tier now" became "Pair tier", and a new "Sender overall" badge shows "N targets" when more than 1 pair is High+. Live feed "Tier" became "Pair tier", and a new "Overall" column shows "–" for old rows. All are centred, sortable, with tooltips; the Pairs column widths were trimmed to fit at 1300 px.
  - `session_report.py`: each offender also gets an "Overall tier" (logged `user_tier` at their last message up to the end of the session; older rows fall back to the highest pair tier across all their pairs) in the HTML, PDF and `users.csv`.
  - No engine formula, threshold or existing API field changed. The project's set 2 files were backed up before the merge.
- **Verification:**
  - 49 passed, 1 skipped (4 new tests: `user_tier` logged by the server, `/pairs` overall fields, migration of an old database, overall tier in the report). `check_setup.py` OK.
  - Live check on a **copy** of the real SafeChat databases (port 8101; the real DB and the 8100 server untouched): the migration added the columns; 3 sessions with bots in follow mode (93, 92, 106 messages) all logged `user_tier`; auto-save wrote 3 reports.
  - Screenshots: `reports/figures/safechat_console/merged_*.png`.

### Known issues noticed (not changed)
- `scripts/demo_stream.py` without `--api` passes `encryption_key="demo-db-key"`, so the in-process demo fails on Windows (no sqlcipher3). The Windows batch file uses `--api`, so it is unaffected.
- SafeChat Lab has no "delete participant" function, although the protocol and consent text promise deletion on request. It is needed before the human study (flagged in the ethics draft).
- The on-screen consent text says role-players target "other study personas". The protocol says bots or the researcher persona only. Suggest rewording to avoid implying that participants are targeted.
- On this PC, `git.exe`, bash and PowerShell cannot create files under `D:\SLIIT` (Windows "Controlled folder access" or similar); `python.exe` can. Git work for the backup was therefore done outside `D:\SLIIT`.
