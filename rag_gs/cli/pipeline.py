import argparse
import os
from rag_gs.core.config import load_config
from rag_gs.core.pipeline import run_pipeline
from rag_gs.core.utils import parse_qids


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG GS pipeline runner")
    parser.add_argument("--stages", type=str, required=False, default="embed,retrieve,merge,score,prune,rank")
    parser.add_argument("--qids", type=str, required=False, default="ALL")
    parser.add_argument("--run-id", type=str, required=False, default=None)
    parser.add_argument("--questions-pack", type=str, required=False, default=None)
    parser.add_argument("--tags", type=str, required=False, default=None)
    parser.add_argument("--profile", type=str, required=False, default=None)
    args = parser.parse_args()

    if args.profile:
        os.environ["RAGGS_PROFILE"] = args.profile  # make load_config pick it up

    cfg = load_config()
    qids = parse_qids(args.qids)
    stages = [s.strip() for s in args.stages.split(',') if s.strip()]
    # Note: wiring of questions-pack/tags to pipeline happens when runner orchestrates S1.
    run_pipeline(stages, qids, args.run_id, cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
