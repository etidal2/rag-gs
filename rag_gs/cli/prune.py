import argparse
from rag_gs.core.config import load_config
from rag_gs.core.utils import parse_qids
from rag_gs.stages.s5_prune.run import run_prune_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="S5: prune ≥min")
    parser.add_argument("--qids", type=str, required=False, default="ALL")
    parser.add_argument("--run-id", type=str, required=False, default=None)
    parser.add_argument("--min", dest="min_grade", type=int, required=False, default=None)
    args = parser.parse_args()

    cfg = load_config()
    qids = parse_qids(args.qids)
    run_prune_stage(qids=qids, run_id=args.run_id, override_min=args.min_grade, cfg=cfg)


if __name__ == "__main__":  # pragma: no cover
    main()

