from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI

from rag_gs.core.config import Config
from rag_gs.core.io import read_jsonl, write_jsonl
from rag_gs.core.manifest import write_stage_manifest
from rag_gs.core.utils import parse_qids
from rag_gs.plugins.judges.openai import chat_json
from rag_gs.plugins.judges.prompts import SCORE_SYSTEM_PROMPT as SYSTEM_PROMPT
from rag_gs.workspace import RunPaths, default_run_id


"""SYSTEM_PROMPT is imported from rag_gs.plugins.judges.prompts as SCORE_SYSTEM_PROMPT"""


def _build_user_prompt(question: str, text: str) -> str:
    return (
        "Question:\n"
        f"{question}\n\n"
        "Texte:\n"
        f"{text}\n\n"
        'Réponds en JSON minifié EXACTEMENT:\n{"grade_1_5": <entier entre 1 et 5>}'
    )


def run_score_stage(
    *,
    qids: Sequence[str],
    run_id: Optional[str],
    override_model: Optional[str],
    override_temperature: Optional[float],
    override_top_p: Optional[float],
    cfg: Config,
) -> None:
    run_id = run_id or default_run_id()
    paths = RunPaths(run_id)

    model = override_model or cfg.s4.model
    temperature = float(override_temperature if override_temperature is not None else cfg.s4.temperature)
    top_p = float(override_top_p if override_top_p is not None else cfg.s4.top_p)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set for scoring")
    client = OpenAI(api_key=api_key)

    # Optional env fallback to mirror legacy behavior (S4_QIDS)
    if not qids:
        env_qids = os.getenv("S4_QIDS")
        if env_qids:
            qids = parse_qids(env_qids, default=[f"Q{i}" for i in range(1, 6)])

    for qid in (list(qids) if qids else [f"Q{i}" for i in range(1, 6)]):
        inputs = read_jsonl(paths.s3_merge_path(qid))
        total = len(inputs)
        print(f"[Script4] Scoring {qid}: {total} items -> {paths.s4_scores_path(qid)}", flush=True)
        outputs: List[Dict[str, Any]] = []
        for idx, rec in enumerate(inputs, start=1):
            question = rec.get("question") or ""
            text = rec.get("text") or ""
            print(
                f"[{qid}] {idx}/{total} doc_id={rec.get('doc_id')} rank_rrf={rec.get('rank_rrf')}",
                flush=True,
            )
            result = chat_json(
                client,
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(question, text)},
                ],
                temperature=temperature,
                top_p=top_p,
            )
            grade = result.get("grade_1_5")
            if not isinstance(grade, int):
                raise ValueError(f"Invalid grade in judge output: {result!r}")
            out = {
                "qid": rec.get("qid"),
                "doc_id": rec.get("doc_id"),
                "grade_1_5": int(grade),
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "ts": datetime.now(timezone.utc).isoformat(),
                # passthrough
                "source": rec.get("source"),
                "rank_dense": rec.get("rank_dense"),
                "rank_sparse": rec.get("rank_sparse"),
                "score_cos": rec.get("score_cos"),
                "score_es": rec.get("score_es"),
                "score_bm25": rec.get("score_bm25"),
                "rrf_score": rec.get("rrf_score"),
                "rank_rrf": rec.get("rank_rrf"),
            }
            outputs.append(out)
        write_jsonl(paths.s4_scores_path(qid), outputs)

        write_stage_manifest(
            paths.q_dir(qid) / "s4_scores",
            {
                "run_id": run_id,
                "qid": qid,
                "stage": "s4_score",
                "count_scored": len(outputs),
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
            },
        )
