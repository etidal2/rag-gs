# ADR-001: Monorepo layout for RAG GS

Context
- Multiple scripts with duplicated utilities and ad-hoc paths made iteration slower and brittle.

Decision
- Create an installable package `rag_gs` with clear separation:
  - `core/` for shared infra (config, IO, logging, manifests, pipeline runner).
  - `stages/` for business logic per stage (s1..s6).
  - `plugins/` for backend wrappers (embeddings, elastic, judges).
  - `workspace.py` for run-aware path resolution under `data/runs/<run_id>/...`.
- YAML configs with env overrides.
- Per-stage manifests + `_SUCCESS` marker for safe resume.

Status
- Accepted.

Consequences
- Easier reuse and consistency across stages; safer resumption; clearer CLIs.

