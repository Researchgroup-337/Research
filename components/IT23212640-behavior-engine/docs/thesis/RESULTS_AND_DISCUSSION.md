# Chapter: Results and Discussion

*Behavior-Based Threat Intelligence Engine for Real-Time Threatening Behavior and Repeated Harassment Detection.* Jayarathna W.D.O.B. (IT23212640), J26-CS-337.

All figures quoted in this chapter are produced by the scripts in the project repository. The sources are `reports/EVALUATION_REPORT.md`, `reports/evaluation_results.json`, `reports/risk_model_fit.json`, `reports/content_model_*.json`, `reports/override_tuning.json` and `reports/collected/`. The engine configuration is `models/risk_model.json`. All experiments use the fine-tuned DistilBERT content model in int8-quantised ONNX form (`distilbert_onnx`) unless stated otherwise.

---

## 1. Experimental setup

### 1.1 Datasets

| Dataset | Size used | Role |
|---|---|---|
| Jigsaw Toxic Comment Classification [10] | 127,657 training, 15,956 validation and 63,978 labelled test comments (211 *threat*, 6,090 *toxic* in test) | Training, threshold tuning and testing of the content model. Held-out text feeds the generator pools |
| Davidson et al. [7] | 12,391 tweets for cross-domain evaluation (10,310 abusive); the other half is used as generator text | Cross-domain test on short, informal text closer to live chat |
| Toxic Comment Moderation | 1,050 comments from 308 users (385 toxic) | Real-data sanity check of user-level tiers (E8) |
| Combined Hate Speech | 48,049 rows | **Excluded**: 24,783 rows overlap with Davidson, which would leak test data (decision D4) |
| Generator pools | 45,774 (train) and 45,775 (test) messages in four severity buckets: clean, mild, severe, threat | Real message texts placed into the synthetic worlds; the content model never saw them during training |

### 1.2 Synthetic multi-session worlds

No public corpus records which user targeted which victim across several sessions with ground truth. Prior work notes that most cyberbullying datasets are labelled per message, not per relationship over time [1], [6]. Behaviour-aware studies on Twitter and Instagram depend on user, network or temporal features that require such histories [3]–[5].

A generator therefore simulates live-stream communities. Each world has 6 channels, about 1,500 viewers and 60 days, with 25% *mention dropout*: aimed messages that lack an @mention, so the target resolver must fall back to the channel owner. Nine personas include hard negatives (banter between friends, one-session bursts, roaming trolls, reformed users) and a hard positive (the subtle harasser, whose messages are mostly mild). The generator does not use the Target Persistence Index (TPI) formula (decision D16). Message texts are real; only the behaviour is simulated.

| World | Messages | Sessions | Senders | Pairs | Harassment pairs | Text pool |
|---|---|---|---|---|---|---|
| train | 102,836 | 177 | 1,500 | 9,357 | 201 | train pool |
| validation | 106,549 | 183 | 1,500 | 8,968 | 206 | train pool |
| test | 104,927 | 179 | 1,500 | 9,481 | 213 | **test pool** |
| test_noisy (50% mention dropout) | 100,694 | 172 | 1,499 | 8,399 | 185 | test pool |
| test_rotation (new account every session) | 104,013 | 182 | 5,070 | 16,859 | 3,235 | test pool |

The risk model is fitted on **train**. The decay half-life, tier boundaries and override threshold are chosen on **validation**. All headline results are reported on the held-out **test** world, whose message texts come from a pool that neither the content model nor the fitted risk model saw.

### 1.3 Baselines and metrics

| System | Flags a pair when |
|---|---|
| (a) Stateless classifier | any message in the pair has T ≥ 0.5 (content-only moderation) |
| (b) Frequency rule | ≥ 3 hostile messages to the same target ("three strikes") |
| (c) **Proposed engine** | pair tier ≥ High |
| (d) Upper bound | gradient boosting on the same features (non-linear, not deployable as a transparent model) |

Metrics are pair-level ROC-AUC, PR-AUC (more informative than ROC-AUC at a positive rate of about 2%), precision, recall and F1. For streaming, the metrics are detection rate, median number of sessions until detection, and false-alarm rate. Latency is measured per message on a laptop CPU.

---

## 2. Content model results (Stage B1)

**Table 1: Content models on the Jigsaw test set and cross-domain data** (`reports/content_model_*.json`)

| Model | Jigsaw mean ROC-AUC | Jigsaw macro-F1 (6 labels) | Abusive vs clean F1 (Jigsaw) | Threat F1 | Davidson F1 (cross-domain) | Moderation set F1 |
|---|---|---|---|---|---|---|
| **DistilBERT (ONNX int8)** | **0.984** | **0.612** | **0.711** | 0.510 | **0.898** | **0.561** |
| TF-IDF + logistic regression | 0.980 | 0.565 | 0.694 | 0.513 | 0.842 | 0.481 |

DistilBERT [8] improves on the classical baseline on every aggregate measure. The largest gain is cross-domain: Davidson F1 rises from 0.842 to 0.898 (precision 0.970, recall 0.835). On the abusive-vs-clean task on Jigsaw it reaches precision 0.605 and recall 0.862. Threat F1 is the same for both models (0.510 vs 0.513), but the balance differs: DistilBERT has higher threat recall (0.815 vs 0.621) and lower precision (0.371 vs 0.437). For a safety signal this trade-off is preferable, and it has consequences for the explicit-threat override (Section 7).

**Why Jigsaw F1 stays below 0.82.** The Jigsaw test labels are known to be noisy and inconsistent, a limitation also noted in surveys of cyberbullying datasets [6]. Positives are rare: 211 threats among 63,978 comments. The per-label thresholds were tuned on validation data, not on the test set. Ranking quality is high (mean ROC-AUC 0.984; threat ROC-AUC 0.990), so the shortfall comes mostly from precision at the chosen thresholds on noisy rare labels, not from poor separation.

**Why Davidson is the more relevant result.** Live-stream chat is short and informal, like tweets, while Jigsaw is Wikipedia talk-page text. A model trained only on Jigsaw that reaches F1 0.898 on Davidson tweets shows that the representation transfers to the target domain. Against the proposal target (content-model F1 ≥ 0.82), the target is therefore met on cross-domain chat-like text but not on the Jigsaw test set (Section 13).

**Latency.** DistilBERT ONNX int8 runs at 6.78 ms (p50) and 10.25 ms (p95) per message for the content model alone.

---

## 3. Fitted risk model

The composite risk is a monotone (non-negative) logistic regression fitted on pair snapshots of the train world (decision D8). The decay half-life was chosen by grid search on the validation world (decision D9).

**Table 2: Half-life grid search (validation world)** (`reports/risk_model_fit.json`)

| Half-life (days) | 3 | 7 | 14 | **30** | no decay |
|---|---|---|---|---|---|
| Validation PR-AUC | 0.811 | 0.932 | 0.964 | **0.968** | 0.926 |

**Fitted model** (half-life 30 days):

Risk = σ(−11.75 + **1.10**·T_pair + **2.79**·log(1+r) + **0.20**·s + **3.98**·recency + **0.00**·v)

- α (content) = 1.10; β (total TPI weight) = 6.97, with normalised TPI weights w₁ (volume) = 0.40, w₂ (session recurrence) = 0.03 and w₃ (recency) = 0.57; γ (escalation velocity) = 0.
- Standardised importance (coefficient × feature SD on validation): log_r 1.810, recency 1.455, s 0.384, T_pair 0.138, v 0.000.
- Validation PR-AUC 0.968, ROC-AUC 0.999.
- Tier boundaries (decision D10): Medium ≥ 0.461 (recall ≥ 0.95), High ≥ 0.554 (maximum F1), Critical ≥ 0.706 (precision ≥ 0.95).

**Interpretation.** Decayed hostile volume and recency dominate. Session recurrence adds a little on top, and the pair-level content signal adds little once hostility has been counted, because counting already uses T (a message is hostile when T ≥ 0.5; decision D7). The escalation weight γ is exactly zero. The monotone fit drives it to its lower bound because, on this data, escalation velocity carries no information beyond decayed volume and recency (decision D15). This is reported as a finding. Escalation is still computed and shown to moderators as evidence, and the monotonicity constraint means an escalating offender can never *lower* their risk.

---

## 4. Pair-level detection (E1)

**Table 3: Pair-level detection on the held-out test world** (9,481 pairs, 213 harassment pairs)

| System | ROC-AUC | PR-AUC | Precision | Recall | F1 | Pairs flagged |
|---|---|---|---|---|---|---|
| (a) Stateless classifier | 0.963 | 0.354 | 0.036 | 1.000 | 0.070 | 5,898 |
| (b) Frequency rule | 0.997 | 0.907 | 0.117 | 1.000 | 0.210 | 1,815 |
| **(c) Proposed engine** | **0.998** | **0.947** | **0.910** | **0.812** | **0.859** | **190** |
| (d) Upper bound (gradient boosting) | 0.998 | 0.939 | 0.900 | 0.850 | 0.874 | 201 |

Figure: `reports/figures/pr_curve_test_world.png`.

A content-only moderator finds every harasser but has to flag 5,898 pairs to do so, a precision of 3.6%. That is unusable in practice and is exactly the gap behaviour-aware approaches aim to close [3], [5]. The frequency rule is better (precision 0.117) but still flags almost ten pairs for every true harasser. The proposed engine flags 190 pairs at precision 0.910 and recall 0.812 (F1 0.859). Its PR-AUC of 0.947 is slightly *higher* than the gradient-boosting upper bound (0.939), and its F1 is close (0.859 vs 0.874). The transparent, monotone model therefore loses very little against an opaque non-linear model.

**Table 4: Tier distribution by ground truth (test world)**

| Label | Low | Medium | High | Critical |
|---|---|---|---|---|
| 0 (not a repeat harasser) | 9,238 | 13 | 14 | 3 |
| 1 (repeat harasser) | 34 | 6 | 12 | 161 |

161 of the 213 harassment pairs (75.6%) reach Critical, while only 3 of 9,268 non-harassment pairs do. The tiers therefore carry an operational meaning: Critical is very rarely a false alarm.

For comparison, the earlier TF-IDF run of the same pipeline (recorded in the project README) gave F1 0.860 (precision 0.902, recall 0.822) and PR-AUC 0.941. The persistence engine is therefore robust to the choice of content model. DistilBERT's advantage shows mainly in the content-level and cross-domain results.

---

## 5. Ablation study (E2)

**Table 5: PR-AUC on the test world when one component is removed (refitted each time)**

| Variant | PR-AUC |
|---|---|
| Full model | 0.947 |
| without session recurrence s | 0.941 |
| without escalation v | 0.947 |
| without recency term | 0.947 |
| without volume log(1+r) | 0.920 |
| without content T (TPI only) | 0.951 |
| content T only (no TPI) | **0.379** |
| without decay (λ = 0) | 0.914 |

Figure: `reports/figures/ablation_pr_auc.png`.

The key result is the drop from 0.947 to **0.379** when only the content signal is kept. **Persistence, not message content, carries the repeat-harassment signal.** Removing decay (λ = 0) lowers PR-AUC to 0.914, which supports decision D6: with raw counts, reformed users and one-off bursts stay elevated. Removing the volume term costs the most among the TPI parts (0.920). Removing the pair-level content term slightly *raises* PR-AUC (0.951), because content already acts through the hostility gate (T ≥ 0.5) that drives r and s. T_pair is kept because it is cheap, its weight is small, and it lets a pair with very severe content rise faster. Removing v or the separate recency term changes nothing (0.947), consistent with γ = 0 and with recency information already being carried by the decayed counters.

---

## 6. Persona analysis (E3)

**Table 6: Share of pairs (with ≥ 1 hostile message) flagged by each system** (harasser personas: victim pair only)

| Persona | Pairs | Stateless | Frequency | Engine |
|---|---|---|---|---|
| normal | 2,364 | 1.00 | 0.30 | 0.00 |
| banter | 461 | 1.00 | 0.54 | 0.00 |
| one_off | 316 | 1.00 | 0.32 | 0.00 |
| burst | 244 | 1.00 | 0.59 | 0.01 |
| roaming_troll | 1,740 | 1.00 | 0.06 | 0.00 |
| reformed | 124 | 1.00 | 0.63 | 0.01 |
| persistent | 110 | 1.00 | 1.00 | **0.93** |
| persistent_subtle | 75 | 1.00 | 1.00 | **0.60** |
| escalator | 28 | 1.00 | 1.00 | **0.93** |

Figure: `reports/figures/flag_rate_by_persona.png`.

The engine almost never flags the hard negatives that defeat count rules. The frequency rule flags 54% of banter pairs, 59% of burst attackers and 63% of reformed users; the engine flags at most 1% of each. It catches 93% of persistent harassers and 93% of escalators.

**Weak spot: subtle harassers.** Only 60% of subtle-harasser pairs are flagged. Their messages are mostly mild, often below the hostility threshold T ≥ 0.5, so the decayed counters grow slowly. This is a limitation of gating persistence on a binary hostility decision. A graded contribution (for example, weighting mild messages partially) or a lower threshold for repeated low-level hostility are candidate improvements (Section 15).

---

## 7. Streaming detection and the explicit-threat override (E4)

### 7.1 Streaming results

The test world was replayed message by message. A system "detects" a pair at the first message where it raises an alarm (tier ≥ High for the engine).

**Table 7: Streaming detection on the test world** (213 harassment pairs, 9,268 other pairs)

| System | Harasser pairs detected | Median sessions to detect | False-alarm pairs | False-alarm rate |
|---|---|---|---|---|
| Stateless | 100.0% | 1 | 5,685 | 61.34% |
| Frequency | 100.0% | 2 | 1,602 | 17.29% |
| Engine (≥ Medium) | 97.2% | 4 | 710 | 7.66% |
| **Engine (≥ High)** | **95.8%** | **4** | **696** | **7.51%** |
| Engine, override disabled | 91.5% | 6 | 89 | 0.96% |

Figure: `reports/figures/risk_timeline_example.png` (a repeat harasser's risk climbing across sessions while a one-session burst attacker decays).

The engine detects 95.8% of harassers after a median of four sessions, with a false-alarm rate of 7.51%. That is eight times lower than content-only moderation and less than half that of the frequency rule.

### 7.2 Why the false-alarm rate rose with DistilBERT

With the TF-IDF content model the engine's streaming false-alarm rate was 3.0% (project README); with DistilBERT it is 7.51%. The cause is the **explicit-threat safety override** (decision D11). If a single message has P(threat) ≥ 0.9, the pair is held at ≥ High for 24 hours, so a credible threat ("I know where you live") does not wait for persistence evidence.

The engine was replayed with the override disabled (`threat_override_p` = 1.01). The no-override false-alarm rate is 0.96%, so **607 of the 696 false-alarm pairs (87%) are caused only by the override**. DistilBERT assigns P(threat) ≥ 0.9 more often than TF-IDF, in line with its higher threat recall (Section 2).

These "false alarms" are false only with respect to the *repeat-harassment* label. Of the 607 override-only pairs, **452 contain a message carrying a Jigsaw *threat* label**: a human annotator judged the text threatening. Only 155 were triggered by text without that label. For most of these pairs the override did what it was designed to do. They are single, credible threats from users who are not repeat harassers, and in a live deployment they warrant immediate moderator attention.

The override also improves *harasser* detection. Without it, detection falls from 95.8% to 91.5% and the median time to detection rises from four to six sessions, because harassers who issue an explicit threat early are flagged immediately.

### 7.3 Choosing the threshold (validation world only)

`scripts/tune_override.py` sweeps the threshold on the **validation world only**. The test world is not used for this choice. The last column measures safety coverage: the share of pairs that received a Jigsaw-labelled threat message and were at ≥ High within the 24-hour hold.

**Table 8: Override threshold sweep on the validation world** (206 harassment pairs, 8,762 other pairs; `reports/override_tuning.json`)

| threat_override_p | Harassers detected | Median sessions | False-alarm rate | Override-only false alarms (with a labelled threat) | Labelled-threat pairs at High within 24 h |
|---|---|---|---|---|---|
| **0.90 (current)** | 99.5% | 4 | 7.05% | 500 (386) | 582/883 (65.9%) |
| 0.93 | 99.5% | 5 | 6.38% | 441 (358) | 539/883 (61.0%) |
| 0.95 | 99.5% | 5 | 5.99% | 407 (337) | 513/883 (58.1%) |
| 0.97 | 99.0% | 5 | 5.63% | 375 (321) | 486/883 (55.0%) |
| 0.99 | 98.1% | 6 | 3.98% | 231 (210) | 343/883 (38.8%) |
| never | 97.6% | 6 | 1.35% | 0 (0) | 92/883 (10.4%) |

The trade-off is direct. Every step that reduces alert volume also removes alerts on messages a human labelled as threats. Moving from 0.90 to 0.99 lowers the false-alarm rate from 7.05% to 3.98%, but safety coverage falls from 65.9% to 38.8% and harasser detection takes two more sessions. Disabling the override reaches 1.35% but leaves almost 90% of labelled threats without a timely alert.

[**Decision pending supervisor/author approval.** The current value 0.90 is retained in this chapter. If a different value is adopted, record it as decision D18 and update Tables 7 and 8 after re-running `fit_risk_model.py` and `evaluate.py`.]

**Discussion.** The engine combines two different alerting goals: *persistent harassment*, measured by the repeat-harassment label, and *imminent threat*, a safety alert. Counting both against a single "false alarm" figure overstates alert fatigue. The engine now attaches the flag `explicit_threat_override` to such pairs, so a moderator interface can show threat alerts separately from persistence alerts.

---

## 8. Stress tests and account rotation (E5)

**Table 9: Stress tests** (end-of-window pair detection)

| World | System | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| test_noisy (50% mention dropout) | Stateless | 0.949 | 0.253 | 0.038 | 1.000 | 0.074 |
| | Frequency | 0.983 | 0.626 | 0.103 | 0.995 | 0.187 |
| | **Engine** | **0.987** | **0.702** | **0.564** | 0.665 | **0.610** |
| test_rotation (new account per session) | Stateless | 0.733 | 0.372 | 0.281 | 0.845 | 0.422 |
| | Frequency | 0.714 | 0.307 | 0.382 | 0.298 | 0.335 |
| | Engine | 0.715 | 0.317 | 0.000 | 0.000 | 0.000 |

**Target-resolution noise.** When half of the aimed messages lack an @mention, many hostile messages are attributed to the channel owner instead of the true victim. Engine F1 falls from 0.859 to 0.610, but the engine still beats both baselines by a wide margin. The engine is only as good as target resolution, which motivates richer resolution (reply chains, name matching in the session roster; decision D5).

**Account rotation.** When a harasser switches to a new account every session, no per-account history can accumulate. Pair-level detection by the engine falls to **zero**, the expected and honest outcome for any persistence-based method. The victim-centric **coordinated-harassment alert** (decision D12: ≥ 3 distinct hostile senders to one target within 7 days) catches **100%** of the 124 harassment victims, at the cost of alerting on **12.0%** of other targets. This partially mitigates the attack. Linking accounts (behavioural or device fingerprinting) is left as future work, and it raises its own privacy questions.

---

## 9. Adversarial evasion (E6)

**Table 10: Recall of DistilBERT on 1,500 abusive Jigsaw test messages under obfuscation attacks**

| Attack | With normalisation | Without normalisation |
|---|---|---|
| none | 0.858 | 0.861 |
| leetspeak | 0.851 | 0.578 |
| spaced letters | 0.773 | 0.671 |
| zero-width characters | 0.858 | 0.861 |
| homoglyphs | 0.858 | 0.658 |
| character stretching | 0.870 | 0.879 |
| mixed | 0.852 | 0.717 |

Figure: `reports/figures/evasion_recall.png`.

Leetspeak and homoglyph substitution reduce recall from about 0.86 to 0.58 and 0.66. The adversarial normalisation layer (decision D14) restores recall to 0.85–0.86. Spaced letters ("k i l l") remain the hardest attack (0.773 with normalisation).

The zero-width row shows no difference because the **DistilBERT tokenizer already removes zero-width characters**. It was verified that a zero-width-injected sentence produces token IDs identical to the clean sentence. Normalisation therefore matters most for character substitutions, which the WordPiece vocabulary cannot recover by itself. Stretching slightly raises recall without normalisation (0.879), because elongated words remain recognisable sub-words.

---

## 10. Latency (E7)

**Table 11: End-to-end latency per message** (CPU laptop; 600 messages, first 50 excluded as warm-up)

| Store | p50 (ms) | p95 (ms) | max (ms) |
|---|---|---|---|
| Plain SQLite | 8.06 | 9.89 | 11.07 |
| SQLCipher-encrypted | not measured on Windows (sqlcipher3 unavailable) | | |

The engine's own overhead, excluding the content model, is 0.18 ms per message. End-to-end latency at the 95th percentile is about **25 times below** the 250 ms target. Through the REST API under continuous demo traffic (`scripts/demo_stream.py --days 3`, 4,351 messages), the measured engine latency was 17.78 ms (p50) and 25.77 ms (p95), still an order of magnitude below the target. The encrypted-store overhead was measured as about 0.3 ms per message in the earlier TF-IDF run (README) and should be re-measured with DistilBERT on a platform with SQLCipher.

---

## 11. Real-data sanity check (E8)

On the Toxic Comment Moderation dataset (1,050 comments, 308 users), the engine places 305 users at Low and 3 at High. The 3 High users average 5.67 toxic comments, against 1.21 for Low users. The Spearman correlation between toxic-comment count and tier is ρ = 0.159.

This result must be interpreted carefully. The dataset has no victim field, so every comment resolves to the page owner. The comments span March 2021 to December 2024 across five platforms, so with a 30-day half-life the decayed counters never accumulate. The highest pair risk is about 0.006, far below the Medium boundary of 0.461, and the three High users are held there **only by the explicit-threat override**. With the override disabled, all 308 users are Low under both content models. This explains why E8 gives identical tiers with DistilBERT and TF-IDF, even though DistilBERT marks more comments hostile (156 vs 122). E8 confirms that the engine stays quiet on sparse, long-spaced real activity and escalates explicit threats. It does **not** validate the TPI, which requires the cross-session data of Section 12.

---

## 12. SafeChat Lab pilot

SafeChat Lab (decision D17) is the test messaging application requested by the proposal panel. A bots-only pilot (18 bots, 10 sessions, lab clock 2 days per session) produced 1,024 messages and 63 pairs, of which 4 are harassment pairs. Targets were resolved by broadcast (580), reply (257) and mention (187). Re-scored with DistilBERT (`reports/collected/`):

**Table 12: Pilot results (bots only)**

| System | Precision | Recall | F1 | Streaming false-alarm pairs |
|---|---|---|---|---|
| Stateless | 0.085 | 1.000 | 0.157 | 43 |
| Frequency | 0.182 | 1.000 | 0.308 | 18 |
| **Engine** | **0.571** | **1.000** | **0.727** | **8** |

All four harassers were detected after a median of three sessions. All four harassment pairs (harasser, escalator and subtle-harasser roles) ended at High or Critical, while the burst attacker's pairs and all troll pairs stayed Low. The remaining false alarms are mainly the banter pair (two friends trading rough language every session), a known hard case for any content model. The pilot verifies the full pipeline (live chat → engine → research log → export → evaluation). With only four positive pairs it is not statistical evidence. The study with human participants (Chapter X; ethics application pending) is required for that.

Figures: `reports/figures/safechat_chat.png`, `safechat_researcher_feed.png`, `safechat_researcher_pairs.png`, `safechat_researcher_setup.png`, `safechat_join.png`.

---

## 13. Objectives versus targets

| Proposal target | Result | Status |
|---|---|---|
| Precision ≥ 0.85 | 0.910 | **Met** |
| Recall ≥ 0.80 | 0.812 | **Met** (narrow margin) |
| F1 ≥ 0.82 | 0.859 | **Met** |
| AUC ≥ 0.90 | ROC-AUC 0.998 (PR-AUC 0.947) | **Met** |
| Latency ≤ 250 ms per message | 8.06 ms p50, 9.89 ms p95 (in process); 25.8 ms p95 via REST API | **Met** |
| Content-model F1 ≥ 0.82 | 0.898 on Davidson tweets (cross-domain); 0.711 on the Jigsaw test set (abusive vs clean) | **Partly met** |
| SUS ≥ 70 | Moderator walkthrough not yet run (kit prepared; ethics clearance required) | **Pending** |

Additional outcomes not stated as targets: an eightfold reduction in streaming false alarms against content-only moderation (61.34% → 7.51%), a 100% victim alert rate under account rotation, and restored recall under obfuscation attacks.

---

## 14. Limitations and threats to validity

1. **Synthetic behaviour.** The persistence results come from simulated communities. The texts are real but the behaviour patterns are designed. Hard negatives and hard positives, target-resolution noise and separate text pools reduce circularity, and the generator does not use the TPI formula. Even so, the numbers in Sections 4–8 measure performance on the simulated behaviour distribution and must be confirmed on real cross-session data.
2. **Subtle harassers.** Recall is 0.60 for harassers whose messages are mostly mild, against 0.93 for overt harassers.
3. **Account rotation.** Pair-level detection falls to zero. The victim-centric alert recovers all victims, but also alerts on 12% of other targets.
4. **Escalation velocity.** The fitted γ is 0. The proposal's escalation term adds no measurable signal on this data. This may reflect the generator's escalation model rather than real behaviour.
5. **Content model on Jigsaw.** F1 0.711 (abusive vs clean) and 0.510 (threat) are below the 0.82 target. Label noise and rarity explain part of this, but not all.
6. **Explicit-threat override.** It accounts for most streaming false alarms with DistilBERT (607 of 696 on the test world). Most of these involve labelled threats, but 155 were triggered by unlabelled text.
7. **Domain shift.** Jigsaw is long-form Wikipedia text; live chat is shorter, multilingual (Sinhala/Tamil/English code-mixing in Sri Lanka) and rich in emotes. The engine has not been evaluated on code-mixed text.
8. **Real-data checks are weak.** E8 has no victim field and long gaps between comments. The SafeChat pilot used bots only.
9. **Encrypted-store latency** was not re-measured with DistilBERT on Windows.

---

## 15. Future work

1. **Human SafeChat Lab study** (after ethics clearance) with inter-annotator agreement (Cohen's κ) and confidence intervals, followed by the SUS moderator study.
2. **Graded hostility** for subtle harassment: let mild messages contribute partially to the decayed counters instead of applying a binary T ≥ 0.5 gate.
3. **Separate alert channels** for imminent threats (override) and persistent harassment, with their own volume budgets.
4. **Account linking** under privacy constraints (for example, behavioural fingerprints computed on pseudonyms) to address rotation.
5. **Better target resolution** using reply chains and conversational context, given the F1 drop under mention dropout.
6. **Code-mixed Sinhala/Tamil/English** content modelling and evaluation.
7. **Richer escalation modelling** (for example, change-point detection on severity), to test whether the zero γ is a property of the data or of the feature.
8. **Calibration of the content model** per platform, so that the override threshold has a consistent meaning across content models.

---

## References

*Numbering to be aligned with the proposal's reference list. Entries marked † should be checked against the exact entry in the proposal.*

[1] H. Rosa *et al.*, "Automatic cyberbullying detection: A systematic review," *Computers in Human Behavior*, vol. 93, pp. 333–345, 2019.
[2] K. Kumari and J. P. Singh, [title as in proposal] †.
[3] S. Ge, L. Cheng, and H. Liu, "Improving cyberbullying detection with user interaction," in *Proc. Web Conference (WWW)*, 2021, pp. 496–506.
[4] L. Cheng *et al.*, [title as in proposal] †.
[5] D. Chatzakou *et al.*, "Mean birds: Detecting aggression and bullying on Twitter," in *Proc. ACM Web Science Conf.*, 2017, pp. 13–22.
[6] F. Elsafoury, S. Katsigiannis, Z. Pervez, and N. Ramzan, "When the timeline meets the pipeline: A survey on automated cyberbullying detection," *IEEE Access*, vol. 9, 2021.
[7] T. Davidson, D. Warmsley, M. Macy, and I. Weber, "Automated hate speech detection and the problem of offensive language," in *Proc. ICWSM*, 2017.
[8] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter," arXiv:1910.01108, 2019.
[9] J. Brooke, "SUS: A quick and dirty usability scale," in *Usability Evaluation in Industry*, Taylor & Francis, 1996.
[10] Jigsaw/Conversation AI, "Toxic Comment Classification Challenge," Kaggle, 2018.
