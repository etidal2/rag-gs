from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from rag_gs.core.config import Config
from rag_gs.core.io import read_json, read_jsonl, write_jsonl
from rag_gs.core.manifest import write_stage_manifest
from rag_gs.workspace import RunPaths, default_run_id


def _coerce_int(v: Any) -> Optional[int]:
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def _coerce_float(v: Any) -> Optional[float]:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _resolve_candidates_path(dir_path: Any, seg: str, expected_path: Any) -> Any:
    # If expected path exists, use it; otherwise pick the highest-K file matching pattern
    expected = expected_path
    try:
        if expected.exists():
            return expected
    except Exception:
        pass
    base_dir = dir_path
    if not base_dir.exists():
        raise FileNotFoundError(f"Candidates directory missing: {base_dir}")
    candidates = list(base_dir.glob(f"{seg}.top*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No candidates found under {base_dir} for segment {seg}")
    def _extract_k(p: Any) -> int:
        name = p.name
        try:
            middle = name.split('.top', 1)[1]
            num = ''
            for ch in middle:
                if ch.isdigit():
                    num += ch
                else:
                    break
            return int(num) if num else -1
        except Exception:
            return -1
    candidates.sort(key=_extract_k, reverse=True)
    return candidates[0]


def run_merge_stage(
    *, qids: Sequence[str], run_id: Optional[str], override_rrf_c: Optional[float], cfg: Config
) -> None:
    run_id = run_id or default_run_id()
    paths = RunPaths(run_id)
    rrf_c = float(override_rrf_c or cfg.s3.rrf_c)

    for qid in (list(qids) if qids else [f"Q{i}" for i in range(1, 6)]):
        rewrite = read_json(paths.s1_rewrite_path(qid))
        qtext = rewrite.get("text", "")
        rtext = rewrite.get("text_rewrite", "")

        # Resolve dense/sparse candidate paths, falling back to existing topK on disk if config differs
        seg = paths.q_dir(qid).name
        dense_dir = paths.q_dir(qid) / "s2_candidates" / "dense"
        sparse_dir = paths.q_dir(qid) / "s2_candidates" / "sparse"
        dense_path = _resolve_candidates_path(dense_dir, seg, paths.s2_dense_path(qid, cfg.s2.dense_top_k))
        sparse_path = _resolve_candidates_path(sparse_dir, seg, paths.s2_sparse_path(qid, cfg.s2.sparse_top_k))
        dense = read_jsonl(dense_path)
        sparse = read_jsonl(sparse_path)

        merged: Dict[str, Dict[str, Any]] = {}
        for rec in dense:
            doc_id = str(rec.get("doc_id"))
            entry = merged.get(doc_id)
            if entry is None:
                entry = {
                    "qid": qid,
                    "question": qtext,
                    "query_rewrite": rtext,
                    "doc_id": doc_id,
                    "text": rec.get("text"),
                    "metadata": rec.get("metadata") or {},
                    "source": "dense",
                    "rank_dense": None,
                    "rank_sparse": None,
                    "score_cos": None,
                    "score_es": None,
                    "score_bm25": None,
                }
                merged[doc_id] = entry
            entry["rank_dense"] = _coerce_int(rec.get("rank_dense"))
            entry["score_cos"] = _coerce_float(rec.get("score_cos"))
            entry["score_es"] = _coerce_float(rec.get("score_es"))

        for rec in sparse:
            doc_id = str(rec.get("doc_id"))
            entry = merged.get(doc_id)
            if entry is None:
                entry = {
                    "qid": qid,
                    "question": qtext,
                    "query_rewrite": rtext,
                    "doc_id": doc_id,
                    "text": rec.get("text"),
                    "metadata": rec.get("metadata") or {},
                    "source": "sparse",
                    "rank_dense": None,
                    "rank_sparse": None,
                    "score_cos": None,
                    "score_es": None,
                    "score_bm25": None,
                }
                merged[doc_id] = entry
            else:
                entry["source"] = "both"
                # Back-fill text/metadata if missing (parity with legacy)
                if not entry.get("text"):
                    entry["text"] = rec.get("text")
                if not entry.get("metadata"):
                    entry["metadata"] = rec.get("metadata") or {}
            entry["rank_sparse"] = _coerce_int(rec.get("rank_sparse"))
            entry["score_bm25"] = _coerce_float(rec.get("score_bm25"))

        entries = list(merged.values())
        for e in entries:
            r = 0.0
            if e["rank_dense"] is not None:
                r += 1.0 / (rrf_c + e["rank_dense"])
            if e["rank_sparse"] is not None:
                r += 1.0 / (rrf_c + e["rank_sparse"])
            e["rrf_score"] = r

        entries.sort(
            key=lambda it: (
                -it["rrf_score"],
                it["rank_dense"] if it["rank_dense"] is not None else 10**9,
                it["rank_sparse"] if it["rank_sparse"] is not None else 10**9,
                it["doc_id"],
            )
        )
        for i, e in enumerate(entries, start=1):
            e["rank_rrf"] = i

        write_jsonl(paths.s3_merge_path(qid), entries)

        write_stage_manifest(
            paths.q_dir(qid) / "s3_merge",
            {
                "run_id": run_id,
                "qid": qid,
                "stage": "s3_merge",
                "rrf_c": rrf_c,
                "dense_hits": len(dense),
                "sparse_hits": len(sparse),
                "merged_docs": len(entries),
                "rewrite_source": str(paths.s1_rewrite_path(qid)),
                "dense_source": str(dense_path),
                "sparse_source": str(sparse_path),
                "output": str(paths.s3_merge_path(qid)),
            },
        )
