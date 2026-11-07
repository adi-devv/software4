"""Config and result types."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- plasticity ---
    half_life: float = 7 * 86400.0      # forgetting half-life (same unit as the clock)
    node_rate: float = 0.3              # fraction of the gap to max a verified success closes (memory)
    link_rate: float = 0.3              # same, for links between memories
    max_strength: float = 1.0           # strengths are bounded in [0, max_strength]
    # --- fast index ---
    fast_size: int = 200                # bounded size of the fast index
    fast_recent_cap: int = 64           # memories added since the last nap that are also searched fast
    recent_half_life: float = 86400.0
    recency_weight: float = 0.5         # fast-index priority = strength + recency_weight * recency
    confidence: float = 0.5             # fast best score needed to skip the deep pass
    hop_min_link: float = 0.15          # a link must be at least this strong to be followed in the fast pass
    hop_decay: float = 0.9
    hop_seeds: int = 3                  # how many top fast hits get a one-hop expansion
    hop_reserve: int = 2                # strongest linked neighbours always kept in the fast result
    # --- graph ---
    max_entities: int = 12
    syn_threshold: float = 0.85         # phrase-phrase synonym edge if cosine >= this
    syn_topk: int = 5
    cooc_scale: float = 0.5
    syn_scale: float = 1.0
    node_boost: float = 1.0             # passage<->phrase edge weight * (1 + node_boost * strength)
    link_scale: float = 2.0             # passage<->passage plastic edge weight = link_scale * strength
    # --- deep pass (PPR) ---
    damping: float = 0.6
    ppr_iters: int = 50
    ppr_tol: float = 1e-9
    passage_seed_k: int = 5
    phrase_seed_weight: float = 1.0
    passage_seed_weight: float = 0.2
    phrase_match_threshold: float = 0.8
    # --- trust / quarantine ---
    trusted_sources: tuple = ("user", "tests", "system")
    default_trust: float = 1.0
    untrusted_trust: float = 0.2


@dataclass
class Hit:
    id: str
    version: int
    text: str
    score: float          # comparable only within one pass
    via: str              # "fast" or "deep"
    kind: str = ""
    trust: float = 1.0


@dataclass
class ReplayItem:
    task_id: str
    kind: str                         # "success" or "corrected_failure"
    text: str                         # episode text ("" if the outcome had no episode memory)
    correction: str | None
    memory_ids: list = field(default_factory=list)
    episode_id: str | None = None
    t: float = 0.0
    new: bool = False                 # happened after the last sleep
    trust: float = 1.0                # minimum trust over the memories involved
