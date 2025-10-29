import argparse
from rag_gs.core.config import load_config
from rag_gs.core.utils import parse_qids
from rag_gs.stages.s3_merge.run import run_merge_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="S3: RRF merge")
    parser.add_argument("--qids", type=str, required=False, default="ALL")
    parser.add_argument("--run-id", type=str, required=False, default=None)
    parser.add_argument("--rrf-c", type=float, required=False, default=None)
    args = parser.parse_args()

    cfg = load_config()
    qids = parse_qids(args.qids)
    run_merge_stage(qids=qids, run_id=args.run_id, override_rrf_c=args.rrf_c, cfg=cfg)


if __name__ == "__main__":  # pragma: no cover
    main()

