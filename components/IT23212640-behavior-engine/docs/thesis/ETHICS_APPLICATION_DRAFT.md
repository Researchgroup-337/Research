# Ethics Application (Draft): SafeChat Lab Cross-Session Chat Study

**Status: DRAFT for supervisor review. Not yet submitted.** Text in [square brackets] must be completed or confirmed before submission. Adapt the headings to the current SLIIT ethics application form if they differ.

| Item | Details |
|---|---|
| Project | J26-CS-337: Privacy-Preserving Real-Time Cyberbullying Detection Using Transformer-Based NLP and Behavioral Analytics |
| Component | Behavior-Based Threat Intelligence Engine for Real-Time Threatening Behavior and Repeated Harassment Detection |
| Applicant | Jayarathna W.D.O.B. (IT23212640), Faculty of Computing, SLIIT |
| Supervisor / co-supervisor | Prof. Harinda Fernando / Ms. Thriyashi Silva |
| Study type | Minimal-risk laboratory study with adult volunteers: role-play text chat in a purpose-built test application, plus a short usability questionnaire |
| Planned dates | [start date] – [end date], after ethics clearance |
| Location | [SLIIT laboratory / room], Malabe campus; local network only |

---

## 1. Background and aims

Online harassment in live-stream chat is often *persistent*: the same person targets the same victim over several sessions. Content-only moderation tools judge each message in isolation and miss this pattern. This project has built a privacy-preserving engine that tracks offender–target persistence across sessions and assigns a risk tier (Low / Medium / High / Critical) to each offender–target pair.

No public dataset records cross-session offender–target interactions with ground truth. The engine has so far been validated on real message texts arranged into simulated multi-session behaviour. The proposal-evaluation panel noted that success depends on a test messaging application for session tracking and on cross-session data. This study collects that data under controlled, ethical conditions.

**Aims**
1. Collect a small cross-session chat dataset in which real people take part over several sessions, with known (role-assigned) ground truth for harassment pairs.
2. Test whether the engine's behaviour on real sessions agrees with its behaviour on simulated data (detection of persistent harassment; low alerting on banter, one-off bursts and normal users).
3. Measure the usability of the moderator dashboard with the System Usability Scale (SUS; target mean ≥ 70).

## 2. Participants and recruitment

| Item | Plan |
|---|---|
| Number | 12–20 chat participants; 5–10 dashboard (SUS) participants (may overlap) |
| Inclusion | SLIIT students aged **18 or over**, comfortable reading informal English chat |
| Exclusion | Under 18; anyone who indicates that exposure to offensive language could cause them distress; students currently assessed by the applicant |
| Recruitment | Open invitation (notice board / student group message) describing the study, its duration and the presence of offensive language. No direct personal approach, so no one feels pressured |
| Incentive | None tied to academic credit. [Optional: refreshments.] Participation or withdrawal has no effect on any grade or relationship with SLIIT |
| Consent | Written informed consent (Appendix B) before the first session, plus an on-screen consent tick-box built into the chat application's join page |

## 3. Procedure

1. **Information and consent** (10 min): the participant reads the information sheet (Appendix A), asks questions and signs the consent form (Appendix B). They receive a random study token on a slip of paper and choose an invented display name.
2. **Role assignment**: each participant receives a private role card. About 75% are *normal*, *bystander* or *banter* roles. About 25% are role-play harasser roles (*harasser*, *subtle harasser*, *escalator*), plus one *burst* and one *troll* role as hard cases. Role-play harassment is **only** directed at scripted **bots** or at the **researcher's persona**, never at another participant (see section 4).
3. **Chat sessions**: either 6–8 sessions of 15 minutes over 2–3 weeks (Option A), or 8 sessions of 10 minutes in one afternoon (Option B). Participants join a chat room in a web browser on the laboratory network. Scripted bots chat alongside them for volume. The researcher starts and ends each session.
4. **Writing hostile messages**: role-players may use a **message bank** of pre-written messages taken from public research datasets, so they do not have to compose abuse themselves. Use of the bank is logged.
5. **Blinding**: participants never see risk scores or tiers. Only the researcher console shows them.
6. **Dashboard usability (optional, separate consent line)**: a 20-minute moderator walkthrough with four tasks on simulated data, followed by the 10-item SUS questionnaire.
7. **Debrief** after the final session (section 4.6).

Approximate total time per participant: Option A 2–2.5 hours over 2–3 weeks; Option B about 2 hours in one afternoon; plus 20 minutes for the optional usability session.

## 4. Risks and mitigations

| Risk | Likelihood / severity | Mitigation |
|---|---|---|
| Distress from **reading** offensive language | Medium / low | Stated clearly in recruitment, information sheet and on-screen consent. Message bank drawn from public research datasets; no content about real people. Participants may stop at any time |
| Distress from **being targeted** | Low / medium | **No human participant is ever a harassment target.** Harassment roles target bots or the researcher persona only. The researcher monitors the live feed and ends the session if a role-player targets a human |
| Discomfort in **writing** hostile messages | Low / low | Role-play is voluntary; the message bank removes the need to compose abuse; any participant may ask for a non-hostile role |
| Role-play spilling into real relationships | Low / low | Invented display names, private role cards, debrief explaining that all hostility was assigned role-play |
| Disclosure of personal information in chat | Low / medium | Consent text and briefing instruct participants not to share real personal information. Study codes only in exports. Researcher can remove such messages (section 5) |
| Data breach | Low / medium | Data held only on the researcher's encrypted laptop; no cloud services; exports contain codes, not names (section 5) |
| Perceived pressure to take part | Low / low | Open recruitment, no academic credit, explicit right to withdraw without reason |

### 4.1 Offensive-language warning
The recruitment notice, the information sheet and the on-screen consent all say that participants *may see offensive language*. The on-screen text currently shown before joining is reproduced in Appendix C.

### 4.2 Message bank
The bank contains short messages from public research corpora (Jigsaw Toxic Comment Classification; Davidson et al. hate-speech tweets). It contains no messages about real, identifiable people. Every use is logged (`from_bank`), so analysis can separate bank messages from free text.

### 4.3 Withdrawal
Participants may stop at any time by closing the browser page, without giving a reason, and may ask for their data to be deleted until [date, e.g. two weeks after the final session] by quoting their study code. [**To do before the study:** the chat application does not yet have a one-click "delete participant" function. Until it is added, deletion is done by the researcher directly in the local research database, and the deletion is recorded in the study log.]

### 4.4 Distress protocol
Any participant can ask the researcher to stop. The researcher then presses *End session*, which closes the chat for everyone. The researcher checks on the participant privately and offers the SLIIT student counselling contact [insert contact]. The participant may withdraw or continue on another day. The incident is recorded (without names) and reported to the supervisor.

### 4.5 Researcher persona
The researcher's own persona is a valid harassment target, so the researcher, not a participant, receives any role-play hostility that is not aimed at bots.

### 4.6 Debrief
After the final session the researcher explains the roles, shows the role cards, confirms that all hostility was scripted role-play, answers questions and repeats the support contacts and the withdrawal deadline.

## 5. Data management

| Aspect | Plan |
|---|---|
| **Identifiers** | Participants are known by study codes (P001...; bots B001...). Tokens are random. Consent forms (with real names) are kept on paper, separately from the data, in [locked cabinet / supervisor's office] |
| **Display names** | Invented by the participant; stored only in the local research database for the live chat. Display names are **not exported**: export files contain study codes only. Names can still appear *inside* message text (for example as @mentions), so any data shared beyond the research team uses the **export without message text** option |
| **What is recorded** | Message text, time, session, sender code, resolved target code and method, the engine's scores, risk and tier, and whether the message came from the bank |
| **What is not recorded** | Real names, student numbers, e-mail addresses, IP addresses or device identifiers are not stored by the application |
| **Engine store** | The behavioural engine itself stores no message text, only pseudonymised (HMAC-SHA256) identifiers, scores and counters |
| **Storage** | On the researcher's password-protected laptop with full-disk encryption [confirm: BitLocker enabled]. The application runs on the local laboratory network only; no cloud services, no third-party APIs |
| **Access** | Applicant and supervisors. A second annotator (a project teammate) sees code-only data for labelling |
| **Sharing / publication** | Only aggregate results (counts, metrics) appear in the thesis. Any released dataset is the code-only export without message text |
| **Retention** | Raw research database and exports kept until the project is assessed [and for the period required by SLIIT policy, e.g. X years]; then securely deleted. Consent forms kept as required by SLIIT |
| **Deletion** | On withdrawal request (section 4.3) and at the end of the retention period. Deletion is recorded in the study log |
| **Legal basis** | Consent; data minimisation and erasure in line with the Sri Lanka Personal Data Protection Act No. 9 of 2022 |

## 6. Benefits
Participants gain insight into how automated moderation works and how harassment can be detected. The wider benefit is evidence that cross-session behavioural analytics can find persistent harassment that content-only tools miss, while protecting privacy.

## 7. Declaration
[Applicant and supervisor signatures and dates per the SLIIT form.]

---

## Appendix A: Participant Information Sheet

**Study title:** Testing a privacy-preserving system that detects repeated online harassment in live chat
**Researcher:** Jayarathna W.D.O.B. (IT23212640), Faculty of Computing, SLIIT. Contact: [SLIIT e-mail]
**Supervisors:** Prof. Harinda Fernando, Ms. Thriyashi Silva

**What is this study about?**
We have built software that looks for people who repeatedly harass the same person in live-stream chat. We want to test it with real people chatting over several sessions.

**What will I do?**
You will chat in a test chat room in your web browser, using an invented name, for [6–8 sessions of 15 minutes over 2–3 weeks / 8 sessions of 10 minutes in one afternoon]. You will get a private role card. Most roles are ordinary chatting. Some people are asked to *role-play* rude behaviour, but only towards computer bots or the researcher's persona, never towards another participant. If you get such a role, you can use ready-made messages instead of writing your own, or ask for a different role. You may also be invited to try a moderator dashboard for 20 minutes and fill in a 10-question form.

**Will I see offensive language?**
Yes, you may. Some messages come from public research datasets of toxic online comments. No message is about you or any real person you know.

**Do I have to take part?**
No. Taking part is voluntary and has no effect on your grades. You can stop at any time without giving a reason, simply by closing the page or telling the researcher.

**What data is collected?**
Your chat messages, when you sent them and who they were addressed to. You are known only by a study code (for example P007). Your invented display name is never exported. Please do not type any real personal information (real names, addresses, phone numbers) in the chat.

**How is my data protected?**
Data stays on the researcher's encrypted laptop and is not sent to any online service. Only summary results appear in the thesis. Data is deleted after the project in line with SLIIT policy.

**Can I withdraw my data?**
Yes, until [date], by quoting your study code to the researcher.

**What if I feel upset?**
Tell the researcher at any time; the session can be stopped for everyone. SLIIT student counselling: [contact].

**Who has approved this study?**
[SLIIT ethics committee reference number.]

## Appendix B: Consent Form

Study: *Testing a privacy-preserving system that detects repeated online harassment in live chat* (J26-CS-337)

Please initial each box.

| | Statement | Initials |
|---|---|---|
| 1 | I have read and understood the information sheet dated [date] and have had the chance to ask questions. | |
| 2 | I am 18 years of age or older. | |
| 3 | I understand that I may see offensive language during the study. | |
| 4 | I understand that I may be asked to role-play rude behaviour towards bots or the researcher persona only, and that I can ask for a different role. | |
| 5 | I understand that taking part is voluntary, that I can stop at any time without giving a reason, and that I can ask for my data to be deleted until [date]. | |
| 6 | I agree that my chat messages, linked only to a study code, may be recorded and analysed for this research. | |
| 7 | I agree not to share any real personal information in the chat. | |
| 8 | (Optional) I agree to take part in the dashboard usability session and questionnaire. | |

Participant name: ______________________  Signature: ______________  Date: ________

Researcher name: Jayarathna W.D.O.B.     Signature: ______________  Date: ________

*Consent forms are stored separately from the research data. Your name is never linked to your chat data in any file that is exported or published.*

## Appendix C: On-screen consent text (current SafeChat Lab join page)

> This chat is part of an undergraduate cybersecurity research study on detecting repeated online harassment. Your messages, the time you send them and who they are addressed to are recorded and analysed. You are known only by a study code (e.g. P007); your display name is never exported. Some participants have been asked to role-play rude or hostile behaviour toward other study personas; none of it is personal. You may see offensive language. You can stop at any time, without giving a reason, by closing this page, and ask the researcher to delete your data. Do not share any real personal information (real names, addresses, phone numbers) in the chat.
>
> ☐ I have read the information above and agree to take part.
