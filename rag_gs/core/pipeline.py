from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence

from .config import Config


@dataclass
class StageSpec:
    name: str

from ..stages.s1_embed.run import run_embed_stage
from ..stages.s2_retrieve.run import run_retrieve_stage
from ..stages.s3_merge.run import run_merge_stage
from ..stages.s4_score.run import run_score_stage
from ..stages.s5_prune.run import run_prune_stage
from ..stages.s6_rank.run import run_rank_stage

DISPATCH: Dict[str, Callable[[Sequence[str], str | None, Config], None]] = {
    "embed": lambda q, r, cfg: run_embed_stage(qids=q, tags=[], questions_pack=None, run_id=r, override_max_batch=None, cfg=cfg),
    "retrieve": lambda q, r, cfg: run_retrieve_stage(qids=q, run_id=r, override_dense_k=None, override_sparse_k=None, cfg=cfg),
    "merge": lambda q, r, cfg: run_merge_stage(qids=q, run_id=r, override_rrf_c=None, cfg=cfg),
    "score": lambda q, r, cfg: run_score_stage(qids=q, run_id=r, override_model=None, override_temperature=None, override_top_p=None, cfg=cfg),
    "prune": lambda q, r, cfg: run_prune_stage(qids=q, run_id=r, override_min=None, cfg=cfg),
    "rank": lambda q, r, cfg: run_rank_stage(qids=q, run_id=r, override_stability=None, cfg=cfg),
}


def run_pipeline(stages: Sequence[str], qids: Sequence[str], run_id: str | None, cfg: Config) -> None:
    q_list = list(qids)
    for name in stages:
        fn = DISPATCH.get(name.lower())
        if not fn:
            raise ValueError(f"Unknown stage: {name}")
        fn(q_list, run_id, cfg)
