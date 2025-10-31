# rag-gs — RAG Golden Sets

An open-source pipeline to build and manage small, high‑quality “golden sets” for RAG evaluation. It organizes the workflow into clear stages — embeddings → retrieval → merge → LLM judge → prune → rank — with shared plumbing, YAML configs, and per-run workspaces.

The package lives under `rag_gs/` and exposes CLI commands for each stage.

## Features

- Installable Python package with CLI entrypoints for each stage.
- Structured per-run outputs under `data/runs/<run_id>/...`.
- YAML configuration with environment overrides and optional pipeline profiles.

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

Create `.env` with only the required secrets. All other tuning lives in YAML.

- Required env: `VOYAGE_API_KEY`, `ES_URL`, `OPENAI_API_KEY`

YAML config files live in `configs/` and contain all knobs (models, batch sizes, ranking settings):

- `configs/default.yaml` (base)
- `configs/local.yaml` (local overrides; ignored by git)
- `configs/pipelines/<name>.yaml` (profile overrides; select via `--profile` or `RAGGS_PROFILE`)

Environment variables can override some keys, but the recommended source of truth is YAML (use `configs/local.yaml` for machine‑local overrides).

## Data Layout

Per-run artifacts are written under `data/runs/<run_id>/...` by the package. Legacy scripts also write to structured folders under `data/`.

Key artifacts by stage:

- S1 (embed): question rewrites and embeddings
- S2 (retrieve): dense/sparse candidate lists
- S3 (merge): RRF fused candidates with scores and ranks
- S4 (score): LLM grades (1–5)
- S5 (prune): grade-bucketed subset (35+)
- S6 (rank): stabilized Top‑20 ordering

## Ranking Refinement (S6)

Stage 6 runs a ranking refinement engine that repeatedly samples small batches of passages (listwise, typically 5 at a time), asks an LLM judge to order them, and updates passage scores using a Plackett–Luce‑style learning rule. It accumulates stable pairwise preferences and locks them into a DAG when confirmed or sufficiently separated by a margin, then recomputes the global order consistent with those locks. The process continues until the Top‑20 list remains unchanged for a configured number of turns, yielding a stable, high‑confidence Top‑20.


## Data Artifacts (script outputs)

- `data/qXX/s1_rewrites/qXX.json`: `qid`, `text`, `text_rewrite`, `embedding`
- `data/qXX/s2_candidates/qXX.top100.jsonl`: top‑K candidates from ES
- `data/qXX/s3_merge/qXX.merge.jsonl`: fused candidates with RRF scores/ranks
- `data/qXX/s4_scores/qXX.gpt5_scores.jsonl`: LLM grades (1–5)
- `data/qXX/s5_pruned/qXX.top35plus.jsonl`: pruned subset by grade buckets
- `data/qXX/s6_ranked/qXX.top20.jsonl`: stabilized Top‑20
- Plus caches, logs, and manifests per stage

## Setup

- Put secrets in `.env`. By default the app does not auto‑load it. You can:
  - Export in your shell: `set -a; source .env; set +a`
  - Or opt‑in auto‑load by setting `RAGGS_LOAD_DOTENV=1` (requires `python-dotenv`), which reads `.env` without overriding existing env.
- Adjust non‑secret settings in YAML:
  - Global defaults: `configs/default.yaml`
  - Local overrides: `configs/local.yaml` (git‑ignored)

## Usage

Prefer the CLI commands provided by the package (see Quickstart above):

- `raggs-embed` — embeddings for question rewrites
- `raggs-retrieve` — dense/sparse retrieval
- `raggs-merge` — RRF merge and rank
- `raggs-score` — LLM grading (1–5)
- `raggs-prune` — build 35+ subset by grade buckets
- `raggs-rank` — listwise ranking to stabilized Top‑20

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
