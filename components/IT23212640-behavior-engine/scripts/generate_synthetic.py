"""Step 3 - Synthetic multi-session harassment corpus (proposal Appendix C).

No public dataset records the same offender targeting the same victim across separate
chat sessions, so we simulate live-stream chat and place REAL labelled messages (from the
held-out Jigsaw/Davidson pools - never seen by the content model) into it.

World
  * C channels (streamers), each streaming every 1-3 days for 2-3 hours over D days.
  * N viewers with home channels and friends; each attends some sessions and chats.
  * Messages are aimed at a friend / another viewer (@mention or reply) or at the streamer.
    With probability `mention_dropout` an aimed message has no mention -> the resolver can
    only attribute it to the streamer (realistic target-resolution noise).

Personas (share of viewers)                                     pair label
  normal            ordinary viewers, rare toxicity                     0
  banter            friends trading profanity across many sessions      0  (hard negative)
  one_off           one bad session aimed at one person                 0
  burst             15+ hostile messages at one person, one session     0  (hard negative for counts)
  roaming_troll     hostile to many different people, rarely repeats    0  (hard negative for user-level)
  reformed          harassed someone early, then stopped               0 at the end (tests decay)
  persistent        returns session after session to target one victim  1
  persistent_subtle as above but mostly mild/clean wording              1  (hard positive for content-only)
  escalator         starts mild, escalates to severe threats            1

Ground truth is assigned at the (offender, victim) pair level.

Outputs data/processed/synthetic_<name>.parquet  (one row per message)
        data/processed/synthetic_<name>_pairs.parquet (ground-truth pairs)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bae.config import DATA_PROCESSED  # noqa: E402
from bae.content_model import load_scorer, threat_signal  # noqa: E402

BUCKETS = ["clean", "mild", "severe", "threat"]
BASE_TS = pd.Timestamp("2026-07-01", tz="Asia/Colombo").timestamp()
DAY = 86400.0

PERSONA_SHARE = {
    "normal": 0.60, "banter": 0.08, "one_off": 0.07, "burst": 0.04, "roaming_troll": 0.05,
    "reformed": 0.03, "persistent": 0.07, "persistent_subtle": 0.04, "escalator": 0.02,
}
POSITIVE = {"persistent", "persistent_subtle", "escalator"}

# Bucket probabilities for messages aimed at the victim / friends.
MIX = {
    "normal": [0.95, 0.045, 0.004, 0.001],
    "banter": [0.55, 0.43, 0.02, 0.0],
    "attack": [0.15, 0.45, 0.25, 0.15],
    "persistent": [0.30, 0.40, 0.18, 0.12],
    "subtle": [0.55, 0.37, 0.06, 0.02],
}


@dataclass
class Params:
    n_users: int = 1500
    n_channels: int = 6
    days: int = 60
    mention_dropout: float = 0.25
    attend_p: float = 0.30
    msgs_per_session: float = 4.0
    account_rotation: bool = False   # harassers switch to a new account every session
    seed: int = 0


def load_pool(pool_name: str, scorer_kind: str) -> pd.DataFrame:
    """Pool texts with pre-computed content scores (cached per scorer)."""
    scorer = load_scorer(scorer_kind)
    cache = DATA_PROCESSED / f"{pool_name}_scores_{scorer.name}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    pool = pd.read_parquet(DATA_PROCESSED / f"{pool_name}.parquet")
    probs = scorer.predict_proba(pool.text.tolist())
    pool["T"] = threat_signal(probs)
    pool["p_threat"] = probs[:, 3]
    pool.to_parquet(cache)
    return pool


class World:
    def __init__(self, p: Params, pool: pd.DataFrame):
        self.p = p
        self.rng = np.random.default_rng(p.seed)
        self.pool = {b: pool[pool.bucket == b].reset_index(drop=True) for b in BUCKETS}
        self.rows: list[dict] = []
        self.msg_id = 0

    # -------------------------------------------------------------- setup
    def build(self):
        rng, p = self.rng, self.p
        self.channels = [f"streamer_{c}" for c in range(p.n_channels)]
        # sessions per channel
        self.sessions = []  # (session_id, channel, start, end)
        for ch in self.channels:
            day = rng.uniform(0, 2)
            k = 0
            while day < p.days:
                start = BASE_TS + day * DAY + rng.uniform(17, 21) * 3600
                dur = rng.uniform(2, 3) * 3600
                self.sessions.append((f"{ch}_s{k:03d}", ch, start, start + dur))
                k += 1
                day += rng.choice([1, 2, 2, 3])
        self.sessions.sort(key=lambda s: s[2])

        users = [f"user_{i:05d}" for i in range(p.n_users)]
        personas = rng.choice(list(PERSONA_SHARE), size=p.n_users, p=list(PERSONA_SHARE.values()))
        self.persona = dict(zip(users, personas))
        self.home = {u: list(rng.choice(self.channels, size=rng.choice([1, 1, 2]), replace=False)) for u in users}
        by_channel = {c: [u for u in users if c in self.home[u]] for c in self.channels}
        self.by_channel = by_channel
        # friends: 2-5 people sharing a home channel
        self.friends = {}
        for u in users:
            cand = [v for c in self.home[u] for v in by_channel[c] if v != u]
            self.friends[u] = list(rng.choice(cand, size=min(len(cand), rng.integers(2, 6)), replace=False)) if cand else []

        # victims for harassers: another viewer of the same channel, or the streamer (30%)
        self.victim, self.active_until, self.active_from = {}, {}, {}
        for u in users:
            per = self.persona[u]
            if per in POSITIVE or per in ("reformed", "one_off", "burst"):
                ch = self.home[u][0]
                if rng.random() < 0.3 and per != "burst":
                    v = ch
                else:
                    cand = [x for x in by_channel[ch] if x != u and self.persona[x] == "normal"]
                    v = rng.choice(cand) if cand else ch
                self.victim[u] = v
                if per == "reformed":
                    self.active_from[u], self.active_until[u] = 0, p.days * 0.4
                elif per in POSITIVE:
                    start = rng.uniform(0, p.days * 0.5)
                    self.active_from[u], self.active_until[u] = start, p.days + 1
        # one_off / burst happen in exactly one session of their home channel
        self.single_session = {}
        for u, per in self.persona.items():
            if per in ("one_off", "burst"):
                ss = [s for s in self.sessions if s[1] == self.home[u][0]]
                self.single_session[u] = ss[rng.integers(len(ss))][0]

    # -------------------------------------------------------------- messages
    def _text(self, bucket: str) -> tuple[str, float, float, str]:
        df = self.pool[bucket]
        i = self.rng.integers(len(df))
        r = df.iloc[i]
        return r.text, float(r["T"]), float(r.p_threat), bucket

    def _emit(self, ts, sess, ch, sender, intended, bucket, aimed: bool, rotation_suffix: str = ""):
        rng = self.rng
        text, T, p_threat, bucket = self._text(bucket)
        is_owner = intended == ch
        resolvable = aimed and not is_owner and rng.random() >= self.p.mention_dropout
        resolved = intended if (resolvable or is_owner) else ch
        if resolvable:
            text = f"@{intended} {text}"
        self.msg_id += 1
        self.rows.append({
            "message_id": f"m{self.msg_id:07d}", "ts": ts, "session_id": sess, "channel": ch,
            "sender": sender + rotation_suffix, "sender_true": sender, "intended_target": intended,
            "target": resolved, "mentioned": resolvable, "text": text, "bucket": bucket, "T": T,
            "p_threat": p_threat, "persona": self.persona.get(sender, "streamer"),
        })

    def _bucket(self, mix):
        return self.rng.choice(BUCKETS, p=mix)

    def simulate(self):
        rng, p = self.rng, self.p
        for sess, ch, start, end in self.sessions:
            day = (start - BASE_TS) / DAY
            viewers = self.by_channel[ch]
            for u in viewers:
                per = self.persona[u]
                harassing = per in POSITIVE and self.active_from[u] <= day <= self.active_until[u]
                if per == "reformed":
                    harassing = day <= self.active_until[u]
                attend = p.attend_p
                if harassing:
                    attend = 0.65
                if per in ("one_off", "burst") and self.single_session[u] == sess:
                    attend = 1.0
                if rng.random() > attend:
                    continue
                suffix = f"_alt{sess[-3:]}" if (p.account_rotation and harassing) else ""
                n = rng.poisson(p.msgs_per_session) + 1
                times = np.sort(rng.uniform(start, end, size=n))
                for k, ts in enumerate(times):
                    # default chatter
                    r = rng.random()
                    if r < 0.55:
                        intended, aimed = ch, False
                    elif self.friends[u]:
                        intended, aimed = rng.choice(self.friends[u]), True
                    else:
                        intended, aimed = ch, False
                    mix = MIX["banter"] if (per == "banter" and aimed) else MIX["normal"]
                    if per == "roaming_troll" and rng.random() < 0.6:
                        others = [v for v in viewers if v != u]
                        intended, aimed, mix = rng.choice(others), True, MIX["attack"]
                    if harassing and rng.random() < 0.7:
                        intended, aimed = self.victim[u], True
                        if per == "persistent_subtle":
                            mix = MIX["subtle"]
                        elif per == "escalator":
                            prog = min(1.0, (day - self.active_from[u]) / max(1.0, p.days - self.active_from[u]))
                            mix = list((1 - prog) * np.array(MIX["subtle"]) + prog * np.array(MIX["attack"]))
                        else:
                            mix = MIX["persistent"]
                    self._emit(ts, sess, ch, u, intended, self._bucket(mix), aimed, suffix)
                # single bad session
                if per in ("one_off", "burst") and self.single_session[u] == sess:
                    m = rng.integers(2, 5) if per == "one_off" else rng.integers(15, 26)
                    for ts in np.sort(rng.uniform(start, end, size=m)):
                        self._emit(ts, sess, ch, u, self.victim[u], self._bucket(MIX["attack"]), True)
        df = pd.DataFrame(self.rows).sort_values("ts").reset_index(drop=True)
        return df

    def ground_truth(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label every observed (sender, resolved target) pair."""
        pairs = df.groupby(["sender", "target"]).agg(
            sender_true=("sender_true", "first"), persona=("persona", "first"), n=("T", "size"),
            first_ts=("ts", "min"), last_ts=("ts", "max")).reset_index()
        pairs["label"] = [int(self.persona.get(st) in POSITIVE and self.victim.get(st) == t)
                          for st, t in zip(pairs.sender_true, pairs.target)]
        return pairs


def generate(name: str, pool_name: str, params: Params, scorer_kind: str = "auto") -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = load_pool(pool_name, scorer_kind)
    w = World(params, pool)
    w.build()
    df = w.simulate()
    gt = w.ground_truth(df)
    df.to_parquet(DATA_PROCESSED / f"synthetic_{name}.parquet")
    gt.to_parquet(DATA_PROCESSED / f"synthetic_{name}_pairs.parquet")
    return df, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", default="auto")
    a = ap.parse_args()
    worlds = {
        # name: (pool, params)
        "train": ("pool_gen_train", Params(seed=1)),
        "val": ("pool_gen_train", Params(seed=2)),
        "test": ("pool_gen_test", Params(seed=3)),
        # stress tests on unseen text
        "test_noisy": ("pool_gen_test", Params(seed=4, mention_dropout=0.5)),
        "test_rotation": ("pool_gen_test", Params(seed=5, account_rotation=True)),
    }
    for name, (pool, params) in worlds.items():
        df, gt = generate(name, pool, params, a.scorer)
        print(f"{name:14s} messages={len(df):,} sessions={df.session_id.nunique()} senders={df.sender.nunique()} "
              f"pairs={len(gt):,} positive_pairs={gt.label.sum()}  hostile_msgs={(df['T']>=0.5).mean():.1%}")
    print(df.groupby("persona").size().to_string())


if __name__ == "__main__":
    main()
