"""Integration contract (JSON) between this component and the rest of J26-CS-337."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class PeerSignals(BaseModel):
    """Optional signals from teammates' components for the same message."""
    url_threat: Optional[Literal["safe", "suspicious", "malicious", "unknown"]] = Field(
        None, description="Malicious URL component (Gunarathna H.M.K.R)")
    hate_score: Optional[float] = Field(None, ge=0, le=1, description="Hate speech component (Mendis W.V.A)")
    hate_type: Optional[str] = None
    se_score: Optional[float] = Field(None, ge=0, le=1, description="Social-engineering component (D.M.T. Dissanayaka)")


class ChatEvent(BaseModel):
    message_id: str
    platform: str = "youtube"
    session_id: str = Field(..., description="Live stream / chat room session identifier")
    channel_owner: Optional[str] = Field(None, description="Streamer / channel owner username")
    sender_id: str
    text: str = Field(..., max_length=5000)
    timestamp: datetime | float
    reply_to: Optional[str] = None
    mentions: list[str] = []
    peer_signals: Optional[PeerSignals] = None
    content_scores: Optional[dict] = Field(None, description="Pre-computed content scores (testing only)")


class MessageScore(BaseModel):
    T: float
    labels: dict[str, float]
    hostile: bool
    effective_T: float


class PairRisk(BaseModel):
    risk: float
    tier: str
    previous_tier: str
    tpi: float
    escalation_v: float
    t_pair: float
    hostile_messages: int
    sessions_targeted: int
    total_messages: int
    last_hostile_ts: Optional[float]


class RiskRecord(BaseModel):
    message_id: str
    offender: str
    target: str
    target_method: str
    target_confidence: float
    message: MessageScore
    pair: PairRisk
    user_max_tier: str
    user_max_risk: float
    targets_harassed: int
    target_pressure: int
    coordinated_attack: bool
    alerts: list[dict]
    flags: list[str]
    latency_ms: float
