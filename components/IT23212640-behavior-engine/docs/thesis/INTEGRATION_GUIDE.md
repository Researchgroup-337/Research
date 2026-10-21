# Integration Guide: Behavioral Analytics Engine (IT23212640)

For teammates in J26-CS-337:
- **Gunarathna H.M.K.R**: malicious URL detection
- **Mendis W.V.A**: hate speech detection
- **D.M.T. Dissanayaka**: social-engineering detection

The engine runs as a local REST service. You send it one chat message at a time, optionally with your own component's score for that message, in `peer_signals`. It returns the offender → target risk tier and alerts. Your signal raises the message's effective threat score, so your detection also counts towards *repeat-harassment* evidence across sessions.

The contract below is copied from `src/bae/schemas.py`. While the API is running, live interactive documentation is at `http://127.0.0.1:8000/docs`.

---

## 1. Start the API

```bash
pip install -r requirements.txt            # Windows: requirements-windows.txt
# optional settings (defaults shown in brackets)
#   BAE_API_KEY   API key clients must send        [dev-key-change-me]  <- change it for anything shared
#   BAE_DB_PATH   SQLite state file                 [data/bae.db]
#   BAE_DB_KEY    enables SQLCipher encryption      [off; needs sqlcipher3, not available on Windows]
#   BAE_RATE / BAE_BURST  rate limit per key        [50 req/s, burst 200]
python -m uvicorn bae.api:app --app-dir src --host 127.0.0.1 --port 8000
```

On Windows PowerShell, set variables with `$env:BAE_API_KEY="..."`; on macOS/Linux use `export BAE_API_KEY=...`.
Check that it is up: `GET http://127.0.0.1:8000/health` returns
`{"status": "ok", "content_model": "distilbert_onnx", "encrypted_store": false, "half_life_days": 30.0}`.

The service binds to `127.0.0.1`, so only processes on the same machine can reach it. To call it from another machine on the lab network, start it with `--host 0.0.0.0` and a strong `BAE_API_KEY`.

## 2. Endpoints

All `/v1` routes need the header `X-API-Key: <key>`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/score` | Score one message (main integration call) |
| POST | `/v1/score/batch` | Score up to 500 messages (processed in timestamp order) |
| GET | `/v1/users/{platform}/{user_id}/risk` | Current risk report for one user (all their targets) |
| GET | `/v1/top?limit=25` | Highest-risk pairs (max 200) |
| GET | `/v1/alerts?limit=50` | Recent alerts (max 500) |
| GET | `/v1/stats` | Store statistics |
| DELETE | `/v1/users/{platform}/{user_id}` | Right to erasure: delete all state for a user |
| POST | `/v1/admin/purge` | Retention purge of old, decayed state |
| GET | `/health`, `/dashboard` | Health check, moderator dashboard (no key needed to load the page) |

## 3. Request: `ChatEvent`

| Field | Type | Required | Notes |
|---|---|---|---|
| `message_id` | string | yes | Your unique ID for the message |
| `platform` | string | no (default `"youtube"`) | Pseudonyms are scoped per platform |
| `session_id` | string | yes | Live stream / chat-room session. Cross-session persistence is counted per distinct `session_id`, so keep it stable for one stream |
| `channel_owner` | string | no | Streamer / room owner. Used as the target when a message has no reply or @mention |
| `sender_id` | string | yes | Platform user ID of the sender (pseudonymised immediately, see section 7) |
| `text` | string | yes | Max 5,000 characters. Scored in memory and **never stored** |
| `timestamp` | number (Unix seconds) or ISO-8601 string | yes | Message time; decay and session counting depend on it |
| `reply_to` | string | no | User ID being replied to (highest-priority target) |
| `mentions` | list of strings | no | User IDs @mentioned |
| `peer_signals` | object | no | Your component's output; see section 5 |
| `content_scores` | object | no | **Testing only.** Bypasses the content model |

### `peer_signals` (`PeerSignals`)

| Field | Type | Owner | Allowed values |
|---|---|---|---|
| `url_threat` | string | Gunarathna H.M.K.R | `"safe"`, `"suspicious"`, `"malicious"`, `"unknown"` |
| `hate_score` | float | Mendis W.V.A | 0.0–1.0 |
| `hate_type` | string | Mendis W.V.A | free text label, e.g. `"ethnicity"`, `"religion"` |
| `se_score` | float | D.M.T. Dissanayaka | 0.0–1.0 |

All four fields are optional, so send only the ones your component produces.

## 4. Example: request and the real response

This exchange was produced by the running API with the DistilBERT content model.

**Request** `POST /v1/score`
```json
{
  "message_id": "yt-000123",
  "platform": "youtube",
  "session_id": "stream-2026-10-19",
  "channel_owner": "streamer_kasun",
  "sender_id": "viewer_8841",
  "text": "@nimal I know where you live, watch your back",
  "timestamp": 1792396800.0,
  "mentions": ["nimal"],
  "peer_signals": {"url_threat": "safe", "hate_score": 0.12, "hate_type": null, "se_score": 0.05}
}
```

**Response** `200 OK` (`RiskRecord`)
```json
{
  "message_id": "yt-000123",
  "offender": "af40db9c7075bb1cc463b849",
  "target": "32039c4a4b8eb62be0f4d5c5",
  "target_method": "mention",
  "target_confidence": 0.9,
  "message": {
    "T": 0.9023,
    "labels": {"toxic": 0.9557, "severe_toxic": 0.0044, "obscene": 0.0348,
               "threat": 0.9023, "insult": 0.045, "identity_hate": 0.0024},
    "hostile": true,
    "effective_T": 0.9023
  },
  "pair": {
    "risk": 0.0053, "tier": "High", "previous_tier": "Low",
    "tpi": 6.1178, "escalation_v": 0.0, "t_pair": 0.3609,
    "hostile_messages": 1, "sessions_targeted": 1, "total_messages": 1,
    "last_hostile_ts": 1792396800.0
  },
  "user_max_tier": "High",
  "user_max_risk": 0.0053,
  "targets_harassed": 1,
  "target_pressure": 1,
  "coordinated_attack": false,
  "alerts": [{"id": 1, "kind": "tier_escalation", "from": "Low", "to": "High"}],
  "flags": ["explicit_threat_override"],
  "latency_ms": 12.159
}
```

How to read it: this is the first message from this sender to this target, so the persistence-based `risk` is still low (0.0053). Because P(threat) ≥ 0.9, the **explicit-threat safety override** holds the pair at **High** for 24 hours, which is why the response carries the `explicit_threat_override` flag. Repeated hostility across sessions is what drives `risk` itself towards Critical.

| Response field | Meaning |
|---|---|
| `offender`, `target` | HMAC-SHA256 pseudonyms (not reversible without the server's key) |
| `target_method` / `target_confidence` | How the target was found: `reply`, `mention`, `name` (in session roster) or `broadcast` (channel owner) |
| `message.T` | Content threat score from the content model (0–1); `effective_T` is after peer signals |
| `message.hostile` | `effective_T ≥ 0.5`; only hostile messages count towards persistence |
| `pair.tier` | `Low` / `Medium` / `High` / `Critical` for this offender → target pair |
| `pair.tpi`, `hostile_messages`, `sessions_targeted` | Persistence evidence for moderators |
| `user_max_tier` | Highest tier this sender has towards anyone |
| `target_pressure`, `coordinated_attack` | Distinct hostile senders to this target in the last 7 days; true at ≥ 3 |
| `alerts` | `tier_escalation` (tier went up to ≥ Medium) or `coordinated_harassment` |
| `flags` | e.g. `explicit_threat_override`, `url_malicious`, `hate_speech:<type>`, `social_engineering` |

## 5. What your signal does

Peer signals change the **effective threat score** of the message: `effective_T = max(T, your mapped score)`. They never lower it. A message with `effective_T ≥ 0.5` counts as hostile, so it adds to the offender's persistence towards that target.

| Signal | Mapping to effective T | Flag added |
|---|---|---|
| `url_threat` | malicious → 0.75, suspicious → 0.45, safe/unknown → 0 | `url_malicious` / `url_suspicious` |
| `hate_score` | `hate_score` (direct) | `hate_speech:<hate_type>` when ≥ 0.5 (`unspecified` if no type) |
| `se_score` | `0.9 × se_score` | `social_engineering` when ≥ 0.5 |

Consequences:
- A `malicious` URL makes the message hostile on its own (0.75 ≥ 0.5). A `suspicious` URL alone does not (0.45), but its flag is still recorded.
- `se_score` needs to be ≥ 0.556 to make a message hostile on its own (0.9 × 0.556 ≈ 0.5).
- Peer signals do **not** trigger the explicit-threat override. That uses only the content model's *threat* label.
- One hostile message gives low risk. The tier rises with repetition across sessions, which is the point of this component.

## 6. Code examples

### curl (Windows: use `curl.exe` in PowerShell, or Git Bash)
```bash
curl -s -X POST http://127.0.0.1:8000/v1/score \
  -H "X-API-Key: dev-key-change-me" -H "Content-Type: application/json" \
  -d '{"message_id":"m1","session_id":"s1","channel_owner":"streamer_kasun","sender_id":"viewer_8841","text":"check this http://x.example","timestamp":1792396800,"peer_signals":{"url_threat":"malicious"}}'
```

### Python (`httpx`, already in requirements)
```python
import httpx

API = "http://127.0.0.1:8000"
client = httpx.Client(base_url=API, headers={"X-API-Key": "dev-key-change-me"}, timeout=10)

def score(event: dict) -> dict:
    r = client.post("/v1/score", json=event)
    if r.status_code == 429:            # rate limited: back off and retry
        import time; time.sleep(0.1)
        return score(event)
    r.raise_for_status()                # 401 / 422 raise here
    return r.json()

rec = score({
    "message_id": "m42", "platform": "youtube", "session_id": "stream-17",
    "channel_owner": "streamer_kasun", "sender_id": "viewer_8841",
    "text": "@nimal you people should leave this country", "timestamp": 1792396800,
    "mentions": ["nimal"],
    "peer_signals": {"hate_score": 0.86, "hate_type": "ethnicity"},   # from the hate-speech component
})
print(rec["pair"]["tier"], rec["flags"], rec["alerts"])
```

Batch: `client.post("/v1/score/batch", json=[event1, event2, ...])` returns a list of `RiskRecord` objects (max 500 events per call).

## 7. Errors

| Code | When | Example body | What to do |
|---|---|---|---|
| 401 | Missing or wrong `X-API-Key` | `{"detail": "invalid or missing X-API-Key"}` | Send the correct key in the header |
| 422 | Body fails validation (wrong type, score outside 0–1, text over 5,000 chars, unknown `url_threat` value, missing required field) | `{"detail": [{"type": "less_than_equal", "loc": ["body","peer_signals","hate_score"], "msg": "Input should be less than or equal to 1", "input": 1.7, "ctx": {"le": 1.0}}]}` | Fix the field named in `loc` |
| 429 | More than the per-key rate (default 50 req/s, burst 200) | `{"detail": "rate limit exceeded"}` | Back off briefly (e.g. 50–100 ms) and retry, or use `/v1/score/batch` |
| 413 | Batch with more than 500 events | `{"detail": "max 500 events per batch"}` | Split the batch |

## 8. What you must NOT send

The engine is built on data minimisation (Sri Lanka PDPA No. 9 of 2022). Please keep it that way:

- **No real names, phone numbers, e-mail addresses, NIC numbers, IP addresses or device IDs** in any field. `sender_id`, `reply_to` and `mentions` should be platform user IDs or handles, which the engine pseudonymises with HMAC-SHA256 on arrival.
- **Do not put personal data into `hate_type`** or other free-text fields. Use short category labels.
- **Do not use `content_scores` in production.** It bypasses the content model and is for tests only.
- **Do not log the full request bodies on your side**, because `text` contains the message. The engine itself never stores message text.
- Do not expose the API to the internet. Keep it on `127.0.0.1` or the lab network with a non-default `BAE_API_KEY`.

Questions: Jayarathna W.D.O.B. (IT23212640).
