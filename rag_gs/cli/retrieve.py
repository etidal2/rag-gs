import argparse
from rag_gs.core.config import load_config
from rag_gs.core.utils import parse_qids
from rag_gs.stages.s2_retrieve.run import run_retrieve_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="S2: retrieval (dense + sparse)")
    parser.add_argument("--qids", type=str, required=False, default="ALL")
    parser.add_argument("--run-id", type=str, required=False, default=None)
    parser.add_argument("--dense-k", type=int, required=False, default=None)
    parser.add_argument("--sparse-k", type=int, required=False, default=None)
    args = parser.parse_args()

    cfg = load_config()
    qids = parse_qids(args.qids)
    run_retrieve_stage(qids=qids, run_id=args.run_id, override_dense_k=args.dense_k, override_sparse_k=args.sparse_k, cfg=cfg)


if __name__ == "__main__":  # pragma: no cover
    main()

