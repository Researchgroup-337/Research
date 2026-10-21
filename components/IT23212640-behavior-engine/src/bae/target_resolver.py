"""Stage A2 - work out WHO a chat message is aimed at.

The Target Persistence Index is defined over (offender, target) pairs, so every message
needs a target. Live chat rarely says so explicitly; the resolver tries, in order:

  1. reply_to      - the platform says this is a reply to a user (highest confidence)
  2. @mention      - first @handle in the text that is not the sender
  3. name mention  - a bare word equal to the name of someone active in this session
  4. broadcast     - nobody named: the message is aimed at the channel owner / streamer

The method is returned as evidence so moderators can see how the target was inferred.
"""
from __future__ import annotations

import re
from collections import defaultdict, deque

MENTION = re.compile(r"@([A-Za-z0-9_\.]{2,32})")
WORD = re.compile(r"[A-Za-z0-9_]{3,32}")

CONFIDENCE = {"reply": 1.0, "mention": 0.9, "name": 0.6, "broadcast": 0.3}


class TargetResolver:
    def __init__(self, broadcast_target: str = "__channel_owner__", roster_size: int = 200):
        self.broadcast_target = broadcast_target
        self.rosters: dict[str, deque] = defaultdict(lambda: deque(maxlen=roster_size))

    def observe(self, session_id: str, user: str) -> None:
        r = self.rosters[session_id]
        if user not in r:
            r.append(user)

    def resolve(self, text: str, sender: str, session_id: str, reply_to: str | None = None,
                mentions: list[str] | None = None, channel_owner: str | None = None) -> tuple[str, str, float]:
        sender_l = sender.lower()
        self.observe(session_id, sender)
        if reply_to and reply_to.lower() != sender_l:
            return reply_to, "reply", CONFIDENCE["reply"]
        for m in (mentions or []) + MENTION.findall(text or ""):
            if m.lower() != sender_l:
                return m, "mention", CONFIDENCE["mention"]
        roster = {u.lower(): u for u in self.rosters[session_id] if u.lower() != sender_l}
        for w in WORD.findall(text or ""):
            if w.lower() in roster:
                return roster[w.lower()], "name", CONFIDENCE["name"]
        return (channel_owner or self.broadcast_target), "broadcast", CONFIDENCE["broadcast"]
