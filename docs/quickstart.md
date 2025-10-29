Quickstart
1) Copy `.env.example` to `.env` and fill required keys.
   - VOYAGE_API_KEY, ES_URL, OPENAI_API_KEY
2) Install: `pip install -e .`
3) Run a pipeline profile (later CLI wiring):
   - `raggs-embed --qids Q1,Q2 --run-id demo`
   - `raggs-retrieve --qids ALL --run-id demo`
   - `raggs-merge --qids ALL --run-id demo`
   - `raggs-score --qids Q1,Q3 --run-id demo`
   - `raggs-prune --qids ALL --run-id demo`
   - `raggs-rank --qids Q1,Q2 --run-id demo`

Artifacts live under `data/runs/<run_id>/...` with per-stage manifests.

