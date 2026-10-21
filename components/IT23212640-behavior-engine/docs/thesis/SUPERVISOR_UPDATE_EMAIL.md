**To:** Prof. Harinda Fernando; Ms. Thriyashi Silva
**Subject:** J26-CS-337 – IT23212640 progress update: Behavioral Analytics Engine results, design changes for approval, ethics guidance

Dear Prof. Fernando and Ms. Silva,

I hope you are both well. I am writing with a progress update on my component, the *Behavior-Based Threat Intelligence Engine for Real-Time Threatening Behavior and Repeated Harassment Detection*, and to ask for your approval of some changes from the approved TAF/proposal and for your guidance on the ethics application.

**Progress**

The engine is complete and runs fully locally. It uses my fine-tuned DistilBERT content model (quantised ONNX, CPU only) followed by the cross-session Target Persistence Index, a fitted risk model, Low–Critical tiers and alerts. A REST API and moderator dashboard are working, and so is the SafeChat Lab test chat application that the proposal panel requested. All results have been regenerated with DistilBERT, and the automated test suite passes.

**Key results** (held-out synthetic test world; real message texts the models never saw)

- Pair-level detection of repeat harassment: precision 0.910, recall 0.812, F1 0.859, ROC-AUC 0.998, PR-AUC 0.947. This meets the proposal targets for precision, recall, F1 and AUC.
- Content-only moderation flags 5,898 pairs to find 213 harassers (precision 0.036). The engine flags 190.
- Ablation: content alone gives PR-AUC 0.379, while adding persistence raises it to 0.947. Persistence carries the signal.
- Latency: 8.06 ms per message (p50) and 9.89 ms (p95) end to end on a laptop CPU, against the 250 ms target.
- Content model: DistilBERT F1 0.898 on Davidson tweets (cross-domain) against 0.842 for the TF-IDF baseline. On the noisy Jigsaw test set, F1 is 0.711, below the 0.82 target. I will report this target as partly met.
- Known weak spots, reported honestly: subtle harassers (recall 0.60) and account rotation (pair-level detection 0). For rotation, the victim-centric alert catches 100% of victims but also 12% of other targets. The fitted escalation weight γ is 0.

**Changes from the TAF/proposal that need your approval** (full reasons in `DECISIONS.md`)

| # | Change |
|---|---|
| D1 | Fine-tuned local DistilBERT instead of a paid third-party moderation API, to keep chat text on the device (privacy) |
| D4 | Combined Hate Speech dataset excluded, because it contains the Davidson tweets (test-data leakage) |
| D6 | Volume and session counters decay exponentially, so reformed users stop being flagged |
| D7 | Only hostile messages (T ≥ 0.5) count towards persistence |
| D8 | Risk model fitted as a monotone (non-negative) logistic regression, so more hostility can never lower risk |
| D11 | Explicit-threat safety override: a single credible threat holds a pair at High for 24 h |
| D12 | Victim-centric coordinated-harassment alert as a counter to account rotation |
| D13 | Privacy controls: HMAC pseudonyms, no message text stored, encryption at rest, retention purge and erasure |
| D17 | SafeChat Lab test chat application and cross-session data-collection protocol |
| D18 | *(only if adopted)* new value for the explicit-threat override threshold. I have analysed this on the validation data. Most of the extra alerts with DistilBERT are on messages that human annotators labelled as threats. My current recommendation is to keep the existing threshold of 0.9 and show threat alerts separately from persistence alerts. |

**Ethics guidance**

The next step is a small study with 12–20 adult SLIIT student volunteers chatting in SafeChat Lab over several sessions, plus a short SUS usability session on the moderator dashboard. Harassment is role-played and directed only at bots or my own persona, never at another participant. A message bank of public-dataset messages means participants do not need to write abuse themselves, and an offensive-language warning, withdrawal rights, a debrief and a distress protocol are built in. I have prepared a draft ethics application with an information sheet and consent form (`docs/thesis/ETHICS_APPLICATION_DRAFT.md`). Could you please advise:

1. which SLIIT ethics form and committee I should submit to, and the expected review time;
2. whether a study of this kind can be reviewed as minimal risk; and
3. whether you would like any changes to the draft before submission.

I would be grateful for a short meeting at your convenience to go through the results and the changes above. The full evaluation report is in `reports/EVALUATION_REPORT.md`, and the draft results chapter is in `docs/thesis/RESULTS_AND_DISCUSSION.md`.

Thank you very much for your time and guidance.

Kind regards,
Jayarathna W.D.O.B.
IT23212640, J26-CS-337
[SLIIT e-mail] | [phone]
