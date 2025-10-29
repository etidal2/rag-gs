from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from rag_gs.core.config import Config
from rag_gs.core.io import read_json, write_jsonl
from rag_gs.core.manifest import write_stage_manifest
from rag_gs.plugins.retrievers.elastic import search as es_search
from rag_gs.workspace import RunPaths, default_run_id


def _build_dense_query(embedding: List[float], vector_field: str, top_k: int) -> Dict[str, Any]:
    return {
        "size": top_k,
        "track_total_hits": False,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": f"cosineSimilarity(params.query_vector, '{vector_field}') + 1.0",
                    "params": {"query_vector": embedding},
                },
            }
        },
    }


def _build_sparse_query(bm25_query: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    if "query" not in bm25_query:
        raise ValueError("bm25_query must contain a 'query' key with the Elasticsearch DSL.")
    body: Dict[str, Any] = {"size": top_k, "track_total_hits": False}
    for k, v in bm25_query.items():
        if k in {"size", "track_total_hits"}:
            continue
        body[k] = v
    return body


def _proc_dense(qid: str, qtext: str, rewrite: str, hits: List[Dict[str, Any]], cfg: Config) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rank, h in enumerate(hits, start=1):
        src = h.get("_source", {})
        score_es = float(h.get("_score", 0.0))
        out.append({
            "qid": qid,
            "question": qtext,
            "query_rewrite": rewrite,
            "rank_dense": rank,
            "score_cos": score_es - 1.0,
            "score_es": score_es,
            "doc_id": h.get("_id"),
            "text": src.get("text"),
            "metadata": src.get("metadata", {}),
            "vector_field": cfg.s2.vector_field,
            "similarity": cfg.s2.similarity,
            "embed_dim": int(cfg.s1.output_dimension),
            "retrieval": "dense_vector",
            "generated_from": "text_rewrite",
        })
    return out


def _proc_sparse(qid: str, qtext: str, rewrite: str, bm25_query: Any, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rank, h in enumerate(hits, start=1):
        src = h.get("_source", {})
        out.append({
            "qid": qid,
            "question": qtext,
            "query_rewrite": rewrite,
            "bm25_query": bm25_query,
            "rank_sparse": rank,
            "score_bm25": float(h.get("_score", 0.0)),
            "doc_id": h.get("_id"),
            "text": src.get("text"),
            "metadata": src.get("metadata", {}),
            "similarity": "bm25",
            "retrieval": "sparse_bm25",
        })
    return out


def _validate_rewrite_payload(payload: Dict[str, Any], expected_dim: int, src_path: Path) -> None:
    required = {"qid", "text", "text_rewrite", "embedding", "bm25_query"}
    missing = required - set(payload.keys())
    if missing:
        raise ValueError(f"{src_path} is missing required keys: {missing}")
    emb = payload.get("embedding")
    if not isinstance(emb, list) or len(emb) != expected_dim:
        raise ValueError(f"Embedding in {src_path} must be a list of length {expected_dim}.")
    for v in emb:
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise ValueError(f"Embedding in {src_path} contains invalid values.")
    bm25_query = payload.get("bm25_query")
    if not isinstance(bm25_query, dict):
        raise ValueError(f"bm25_query in {src_path} must be a JSON object containing the DSL query.")


def run_retrieve_stage(
    *,
    qids: Sequence[str],
    run_id: Optional[str],
    override_dense_k: Optional[int],
    override_sparse_k: Optional[int],
    cfg: Config,
) -> None:
    run_id = run_id or default_run_id()
    paths = RunPaths(run_id)

    dense_k = int(override_dense_k or cfg.s2.dense_top_k)
    sparse_k = int(override_sparse_k or cfg.s2.sparse_top_k)

    # Fail-fast on ES_URL to provide clear error early
    es_url = (cfg.s2.es_url or "").strip()
    if not es_url:
        raise RuntimeError("ES_URL is required (configure in configs/local.yaml or environment)")
    if not (es_url.startswith("http://") or es_url.startswith("https://")):
        raise RuntimeError(f"ES_URL must start with http(s)://, got: {es_url}")

    for qid in (list(qids) if qids else [f"Q{i}" for i in range(1, 6)]):
        s1p = paths.s1_rewrite_path(qid)
        payload = read_json(s1p)
        _validate_rewrite_payload(payload, int(cfg.s1.output_dimension), s1p)
        qtext = payload.get("text", "")
        rewrite = payload.get("text_rewrite", "")
        embedding = payload.get("embedding") or []
        bm25_query = payload.get("bm25_query")

        dense_body = _build_dense_query(embedding, cfg.s2.vector_field, dense_k)
        dres = es_search(es_url, cfg.s2.es_index, dense_body)
        dhits = dres.get("hits", {}).get("hits", [])
        dense_records = _proc_dense(qid, qtext, rewrite, dhits, cfg)
        write_jsonl(paths.s2_dense_path(qid, dense_k), dense_records)

        sparse_body = _build_sparse_query(bm25_query, sparse_k)
        sres = es_search(es_url, cfg.s2.es_index, sparse_body)
        shits = sres.get("hits", {}).get("hits", [])
        sparse_records = _proc_sparse(qid, qtext, rewrite, bm25_query, shits)
        write_jsonl(paths.s2_sparse_path(qid, sparse_k), sparse_records)

        write_stage_manifest(
            paths.q_dir(qid) / "s2_candidates",
            {
                "run_id": run_id,
                "qid": qid,
                "stage": "s2_retrieve",
                "dense_hits": len(dense_records),
                "sparse_hits": len(sparse_records),
                "dense_k": dense_k,
                "sparse_k": sparse_k,
                "index": cfg.s2.es_index,
                "vector_field": cfg.s2.vector_field,
                "similarity": cfg.s2.similarity,
            },
        )
