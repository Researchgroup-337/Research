"""Central configuration. Every tunable number in the engine lives here so that the
decision log (proposal Appendix E) can point to one place."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# How each content label contributes to the single threat signal T (0..1).
# T = max_k(weight_k * p_k). Threat and severe toxicity count fully; generic
# profanity (obscene) counts least because it is common in friendly banter.
LABEL_WEIGHTS = {
    "threat": 1.0,
    "severe_toxic": 1.0,
    "identity_hate": 0.9,
    "insult": 0.85,
    "toxic": 0.75,
    "obscene": 0.5,
}


@dataclass
class EngineConfig:
    # A message is "hostile" (counts toward r and s) when T >= hostile_threshold.
    hostile_threshold: float = 0.5
    # Recency decay: half-life in days. lambda = ln2 / half_life. 0 = no decay (ablation).
    half_life_days: float = 14.0
    # Escalation velocity is the slope of T over the last k messages to the target.
    escalation_window: int = 5
    # EWMA smoothing for the pair-level content signal.
    ewma_alpha: float = 0.4
    # Fitted risk-model coefficients (overwritten by models/risk_model.json).
    intercept: float = -4.0
    coef_T: float = 3.0       # alpha
    coef_logr: float = 0.8    # beta * w1
    coef_s: float = 0.9       # beta * w2
    coef_recency: float = 1.5 # beta * w3
    coef_v: float = 1.0       # gamma
    # Tier thresholds on calibrated risk probability.
    tier_medium: float = 0.25
    tier_high: float = 0.5
    tier_critical: float = 0.8
    # Safety override: an explicit, high-confidence threat is never left below High.
    threat_override_p: float = 0.9
    threat_hold_hours: float = 24.0   # the override floor holds this long after the threat
    # Victim-centric coordinated-harassment alert (counters account rotation).
    pressure_window_days: float = 7.0
    pressure_min_senders: int = 3
    # Privacy: drop pair state once it has decayed and is older than this.
    retention_days: int = 90
    # Fallback target when a message has no @mention or reply.
    broadcast_target: str = "__channel_owner__"
    extra: dict = field(default_factory=dict)

    @property
    def decay_lambda(self) -> float:
        import math
        return math.log(2) / (self.half_life_days * 86400.0)  # per second

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | None = None) -> "EngineConfig":
        path = path or MODELS / "risk_model.json"
        cfg = cls()
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        return cfg


def secret_key() -> bytes:
    """HMAC key for pseudonymising user IDs.

    Order: BAE_SECRET_KEY environment variable -> models/.secret_key -> ~/.bae/secret_key.
    The key file is created once with a random value. If no location is writable (e.g. a
    protected or synced folder on Windows), a per-run random key is used with a warning:
    the engine still works, but pseudonyms will differ between runs."""
    env = os.environ.get("BAE_SECRET_KEY")
    if env:
        return env.encode()
    for p in (MODELS / ".secret_key", Path.home() / ".bae" / "secret_key"):
        try:
            if p.exists():
                key = p.read_bytes().strip()
                if key:
                    return key
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(os.urandom(32).hex().encode())
            try:
                os.chmod(p, 0o600)
            except OSError:
                pass  # chmod is limited on Windows; not fatal
            return p.read_bytes().strip()
        except OSError:
            continue
    import warnings
    warnings.warn("Could not save a secret key file; using a temporary key for this run. "
                  "Set the BAE_SECRET_KEY environment variable to make pseudonyms stable.")
    global _TEMP_KEY
    try:
        return _TEMP_KEY
    except NameError:
        _TEMP_KEY = os.urandom(32).hex().encode()
        return _TEMP_KEY
