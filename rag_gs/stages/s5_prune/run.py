from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rag_gs.core.config import Config
from rag_gs.core.io import read_jsonl, write_jsonl
from rag_gs.core.manifest import write_stage_manifest
from rag_gs.core.utils import parse_qids
from rag_gs.workspace import RunPaths, default_run_id


def _scores_path(paths: RunPaths, qid: str) -> Optional[Path]:
    seg_path = paths.q_dir(qid) / "s4_scores"
    enriched = seg_path / f"{paths.q_dir(qid).name}.gpt5_scores.enriched.jsonl"
    base = seg_path / f"{paths.q_dir(qid).name}.gpt5_scores.jsonl"
    if enriched.exists():
        return enriched
    if base.exists():
        return base
    return None


def _prune_min(records: List[Dict[str, Any]], min_grade: int) -> List[Dict[str, Any]]:
    buckets: Dict[int, List[Dict[str, Any]]] = {g: [] for g in [5, 4, 3, 2, 1]}
    for r in records:
        g = r.get("grade_1_5")
        if isinstance(g, int) and 1 <= g <= 5:
            buckets[g].append(r)
    for g in buckets:
        buckets[g].sort(key=lambda x: (x.get("rank_rrf") if x.get("rank_rrf") is not None else 10**9))
    selected: List[Dict[str, Any]] = []
    for g in [5, 4, 3, 2, 1]:
        if len(selected) >= min_grade:
            break
        selected.extend(buckets[g])
    ts = datetime.now(timezone.utc).isoformat()
    cut_size = len(selected)
    for r in selected:
        # Keep legacy label for contract parity
        r.setdefault("cut_stage", "35plus")
        r.setdefault("cut_size", cut_size)
        r.setdefault("cut_ts", ts)
    return selected


def run_prune_stage(*, qids: Sequence[str], run_id: Optional[str], override_min: Optional[int], cfg: Config) -> None:
    run_id = run_id or default_run_id()
    paths = RunPaths(run_id)
    min_grade = int(override_min or cfg.s5.prune_min)

    # Optional env fallback to mirror legacy behavior (S5_QIDS)
    if not qids:
        env_qids = os.getenv("S5_QIDS") if 'os' in globals() else None
        if env_qids:
            qids = parse_qids(env_qids, default=[f"Q{i}" for i in range(1, 6)])

    for qid in (list(qids) if qids else [f"Q{i}" for i in range(1, 6)]):
        s4p = _scores_path(paths, qid)
        if s4p is None:
            continue
        records = read_jsonl(s4p)
        selected = _prune_min(records, min_grade)
        write_jsonl(paths.s5_pruned_path(qid), selected)
        write_stage_manifest(
            paths.q_dir(qid) / "s5_pruned",
            {
                "run_id": run_id,
                "qid": qid,
                "stage": "s5_prune",
                "total_in": len(records),
                "selected_out": len(selected),
                "min_grade": min_grade,
            },
        )
