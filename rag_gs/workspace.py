from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
# Keep run artifacts inside the repository under ./data to simplify inspection
DATA_ROOT = ROOT / "data"


def default_run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def q_segment(qid: str) -> str:
    return f"q{int(qid[1:]):02d}"


@dataclass(frozen=True)
class RunPaths:
    run_id: str

    @property
    def root(self) -> Path:
        return DATA_ROOT / "runs" / self.run_id

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    def q_dir(self, qid: str) -> Path:
        return self.root / q_segment(qid)

    # Stage outputs
    def s1_rewrite_path(self, qid: str) -> Path:
        seg = q_segment(qid)
        return self.q_dir(qid) / "s1_rewrites" / f"{seg}.json"

    def s2_dense_path(self, qid: str, k: int) -> Path:
        seg = q_segment(qid)
        return self.q_dir(qid) / "s2_candidates" / "dense" / f"{seg}.top{k}.jsonl"

    def s2_sparse_path(self, qid: str, k: int) -> Path:
        seg = q_segment(qid)
        return self.q_dir(qid) / "s2_candidates" / "sparse" / f"{seg}.top{k}.jsonl"

    def s3_merge_path(self, qid: str) -> Path:
        seg = q_segment(qid)
        return self.q_dir(qid) / "s3_merge" / f"{seg}.merge.jsonl"

    def s4_scores_path(self, qid: str) -> Path:
        seg = q_segment(qid)
        return self.q_dir(qid) / "s4_scores" / f"{seg}.gpt5_scores.jsonl"

    def s5_pruned_path(self, qid: str) -> Path:
        seg = q_segment(qid)
        return self.q_dir(qid) / "s5_pruned" / f"{seg}.top35plus.jsonl"

    def s6_ranked_path(self, qid: str) -> Path:
        seg = q_segment(qid)
        return self.q_dir(qid) / "s6_ranked" / f"{seg}.top20.jsonl"
