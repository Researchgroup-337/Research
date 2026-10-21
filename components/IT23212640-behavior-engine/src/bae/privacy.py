"""Privacy controls: pseudonymisation of identities.

User identifiers are replaced by HMAC-SHA256(secret_key, platform:user_id) before
anything is stored. Without the key the hashes cannot be reversed or linked to a
username by brute force over known usernames (unlike a plain SHA-256). Rotating the
key unlinks all stored history (crypto-shredding).
"""
from __future__ import annotations

import hashlib
import hmac

from .config import secret_key


class Pseudonymizer:
    def __init__(self, key: bytes | None = None, length: int = 24):
        self.key = key or secret_key()
        self.length = length

    def __call__(self, user_id: str, platform: str = "") -> str:
        msg = f"{platform.lower()}:{str(user_id).strip().lower()}".encode()
        return hmac.new(self.key, msg, hashlib.sha256).hexdigest()[: self.length]
