# rag-gs — RAG Golden Sets

An open-source pipeline to build and manage small, high‑quality “golden sets” for RAG evaluation. It organizes the workflow into clear stages — embeddings → retrieval → merge → LLM judge → prune → rank — with shared plumbing, YAML configs, and per-run workspaces.

The package lives under `rag_gs/` and exposes CLI commands for each stage. Legacy scripts remain for reference.

## Features

- Installable Python package with CLI entrypoints for each stage.
- Structured per-run outputs under `data/runs/<run_id>/...`.
- YAML configuration with environment overrides and optional pipeline profiles.
- Centralized judge prompts and consistent scoring/ranking flows.
- Pre-commit hooks with formatters, linters, and secret scanning.

## Installation

Requirements: Python 3.10+

- Editable install (recommended for development):
  - `pip install -e .[dev]`
- User install:
  - `pip install -e .`

Copy `.env.example` to `.env` and fill required keys.

## Quickstart

Run the full pipeline with defaults:

- `raggs --stages embed,retrieve,merge,score,prune,rank --qids ALL --run-id demo`

Or run individual stages:

- `raggs-embed --qids Q1,Q2 --run-id demo`
- `raggs-retrieve --qids ALL --run-id demo`
- `raggs-merge --qids ALL --run-id demo`
- `raggs-score --qids Q1 --run-id demo`
- `raggs-prune --qids ALL --run-id demo`
- `raggs-rank --qids Q1,Q2 --run-id demo`

## Configuration

Create `.env` with required keys and optional tuning knobs:

- Required: `VOYAGE_API_KEY`, `ES_URL`, `OPENAI_API_KEY`
- Optional: `MAX_VOYAGE_BATCH`, `S4_QIDS`, `S5_QIDS`, `S6_QIDS`, `S6_*` knobs

YAML config files live in `configs/`:

- `configs/default.yaml` (base)
- `configs/local.yaml` (local overrides; ignored by git)
- `configs/pipelines/<name>.yaml` (profile overrides; select via `--profile` or `RAGGS_PROFILE`)

Environment variables have highest precedence for key settings.

## Data Layout

Per-run artifacts are written under `data/runs/<run_id>/...` by the package. Legacy scripts also write to structured folders under `data/`.

Key artifacts by stage:

- S1 (embed): question rewrites and embeddings
- S2 (retrieve): dense/sparse candidate lists
- S3 (merge): RRF fused candidates with scores and ranks
- S4 (score): LLM grades (1–5)
- S5 (prune): grade-bucketed subset (35+)
- S6 (rank): stabilized Top‑20 ordering

## Legacy Scripts (reference)

This repo includes earlier standalone scripts that perform the same stages step by step. They remain for reference and for simple runs; the package CLIs are the preferred interface.

- `script1_query_embedding.py`: generate Voyage embeddings for question rewrites
- `script2_retrieval.py`: kNN retrieval from Elasticsearch
- `script3_merge.py`: reciprocal-rank fusion (RRF)
- `script4_score_gpt5.py`: LLM grading on 1–5 scale
- `script5_prune_40plus.py`: build 35+ subset by grade buckets
- `script6_rank_top20.py`: listwise ranking to a stable Top‑20

See comments in each script and the artifacts described below.

## Data Artifacts (script outputs)

- `data/qXX/s1_rewrites/qXX.json`: `qid`, `text`, `text_rewrite`, `embedding`
- `data/qXX/s2_candidates/qXX.top100.jsonl`: top‑K candidates from ES
- `data/qXX/s3_merge/qXX.merge.jsonl`: fused candidates with RRF scores/ranks
- `data/qXX/s4_scores/qXX.gpt5_scores.jsonl`: LLM grades (1–5)
- `data/qXX/s5_pruned/qXX.top35plus.jsonl`: pruned subset by grade buckets
- `data/qXX/s6_ranked/qXX.top20.jsonl`: stabilized Top‑20
- Plus caches, logs, and manifests per stage

## Setup (scripts)

Set required environment variables in `.env`:

- `VOYAGE_API_KEY`
- `ES_URL`
- `OPENAI_API_KEY`

Optional filters and tuning:

- `S4_QIDS`, `S5_QIDS`, `S6_QIDS`
- `MAX_VOYAGE_BATCH`, `S6_MODEL_NAME`, `S6_STABILITY_TURNS`, `S6_CONFIRMATIONS_BEFORE_LOCK`, `S6_*`

## Usage (scripts)

- Embeddings: `python script1_query_embedding.py`
- Retrieval: `python script2_retrieval.py`
- Merge (RRF): `python script3_merge.py`
- Grade (LLM): `S4_QIDS=Q1 python script4_score_gpt5.py`
- Prune (35+): `S5_QIDS=Q1 python script5_prune_40plus.py`
- Rank (Top‑20): `S6_QIDS=Q1 python script6_rank_top20.py`
- Stats/report: `python stats.py`

## CLI Summary (package)

- `raggs` — pipeline runner (`--stages`, `--qids`, `--run-id`, `--profile`)
- `raggs-embed` — embeddings for question rewrites
- `raggs-retrieve` — dense/sparse retrieval
- `raggs-merge` — RRF merge and rank
- `raggs-score` — LLM grading (1–5)
- `raggs-prune` — build 35+ subset by grade buckets
- `raggs-rank` — listwise ranking to stabilized Top‑20

## Viewer

- `viewer_scores.html` provides a quick UI to explore scores:
  - Serve locally from repo root: `python -m http.server 8000`
  - Open: `http://localhost:8000/viewer_scores.html`
  - Features: select Q1–Q5, filter by min grade, sort by grade or by source+grade, show question title and chunk text.
- Migrate embeddings in Elasticsearch:
  - `python migrate_embeddings.py`

## Security

- Do not commit real keys; `.env` is gitignored and `.env.example` holds placeholders.
- Pre-commit includes `detect-secrets` with `.secrets.baseline`. Run hooks locally before pushing.

## License

MIT — see `LICENSE`.
