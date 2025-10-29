from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rag_gs.core.config import Config
from rag_gs.core.io import write_json
from rag_gs.core.logging import setup_logger
from rag_gs.core.manifest import write_stage_manifest
from rag_gs.core.utils import chunked
from rag_gs.core.questions import load_questions_pack, filter_questions, Question as PackQuestion
from rag_gs.plugins.embedders.voyage import request_embeddings as voyage_embeddings
from rag_gs.workspace import RunPaths, default_run_id


@dataclass(frozen=True)
class Question:
    qid: str
    text: str
    text_rewrite: str
    bm25_query: Optional[dict] = None

    def cache_key(self, model_name: str, input_type: str, dim: int) -> str:
        raw = f"{self.text_rewrite}|{model_name}|{input_type}|{dim}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _select_questions(pack: str, qids: Sequence[str], tags: Sequence[str]) -> List[Question]:
    raw = load_questions_pack(pack)
    filtered = filter_questions(raw, qids=qids or None, tags=tags or None)
    return [Question(qid=q.qid, text=q.text, text_rewrite=q.text_rewrite, bm25_query=q.bm25_query) for q in filtered]


def _read_cache(path: Path, expected_dim: int) -> Optional[List[float]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    vec = payload.get("embedding")
    if not isinstance(vec, list) or len(vec) != expected_dim:
        return None
    try:
        floats = [float(x) for x in vec]
    except Exception:
        return None
    if any((not math.isfinite(v)) for v in floats):
        return None
    return floats


def run_embed_stage(
    *,
    qids: Sequence[str],
    tags: Sequence[str] | None = None,
    questions_pack: str | None = None,
    run_id: Optional[str],
    override_max_batch: Optional[int],
    cfg: Config,
) -> None:
    run_id = run_id or default_run_id()
    paths = RunPaths(run_id)
    setup_logger(paths.logs_dir / f"s1_embed_{run_id}.log")

    model = cfg.s1.model_name
    input_type = cfg.s1.input_type
    dim = cfg.s1.output_dimension
    trunc = cfg.s1.truncation
    max_batch = int(override_max_batch or cfg.s1.max_voyage_batch)
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY is required for embeddings")

    # Cache dir inside workspace cache
    cache_dir = (paths.root.parent / "cache" / "embeddings")
    cache_dir.mkdir(parents=True, exist_ok=True)

    pack = questions_pack or "example"
    questions = _select_questions(pack, qids, tags or [])

    # Load cache / decide what to fetch
    embeddings: Dict[str, List[float]] = {}
    cache_hits: set[str] = set()
    to_fetch: List[Question] = []
    for q in questions:
        ck = q.cache_key(model, input_type, dim)
        cp = cache_dir / f"{ck}.json"
        cached = _read_cache(cp, dim)
        if cached is not None:
            embeddings[q.qid] = cached
            cache_hits.add(q.qid)
        else:
            to_fetch.append(q)

    # Fetch in batches
    for batch in chunked(to_fetch, max_batch):
        inputs = [q.text_rewrite for q in batch]
        idx_to_vec = voyage_embeddings(
            api_key=api_key,
            model=model,
            input_type=input_type,
            output_dimension=dim,
            inputs=inputs,
            truncation=trunc,
        )
        for idx, q in enumerate(batch):
            vec = idx_to_vec.get(idx)
            if not vec:
                raise RuntimeError(f"Missing embedding for {q.qid}")
            # Validate embedding shape and values
            if len(vec) != dim:
                raise ValueError(f"Embedding for {q.qid} has length {len(vec)} (expected {dim})")
            if any((v is None) or not math.isfinite(float(v)) for v in vec):
                raise ValueError(f"Embedding for {q.qid} contains invalid values.")
            embeddings[q.qid] = vec
            # write cache
            ck = q.cache_key(model, input_type, dim)
            cp = cache_dir / f"{ck}.json"
            write_json(
                cp,
                {
                    "embedding": vec,
                    "model": model,
                    "input_type": input_type,
                    "output_dimension": dim,
                },
                indent=None,
            )

    # Persist per-qid outputs
    for q in questions:
        vec = embeddings[q.qid]
        # Validate cached vectors as well (parity with legacy checks)
        if len(vec) != dim:
            raise ValueError(f"Embedding for {q.qid} has length {len(vec)} (expected {dim})")
        if any((v is None) or not math.isfinite(float(v)) for v in vec):
            raise ValueError(f"Embedding for {q.qid} contains invalid values.")
        write_json(
            paths.s1_rewrite_path(q.qid),
            {
                "qid": q.qid,
                "text": q.text,
                "text_rewrite": q.text_rewrite,
                "embedding": vec,
                "model": model,
                "input_type": input_type,
                "output_dimension": dim,
                "truncation": trunc,
                # Ensure bm25_query exists; default to simple match on rewrite if absent
                "bm25_query": q.bm25_query if q.bm25_query else {"query": {"match": {"text": q.text_rewrite}}},
            },
        )

    # Questions snapshot for provenance
    write_json(
        paths.root / "questions_manifest.json",
        {
            "run_id": run_id,
            "pack": pack,
            "selected_qids": [q.qid for q in questions],
            "count": len(questions),
            "items": [
                {"qid": q.qid, "text": q.text, "text_rewrite": q.text_rewrite}
                for q in questions
            ],
        },
    )

    # Stage manifests per qid
    for q in questions:
        write_stage_manifest(
            paths.q_dir(q.qid) / "s1_rewrites",
            {
                "run_id": run_id,
                "qid": q.qid,
                "stage": "s1_embed",
                "model": model,
                "input_type": input_type,
                "output_dimension": dim,
                "truncation": trunc,
                "max_batch": max_batch,
                "pack": pack,
                "cached": q.qid in cache_hits,
            },
        )
