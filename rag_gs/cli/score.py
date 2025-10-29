import argparse
from rag_gs.core.config import load_config
from rag_gs.core.utils import parse_qids
from rag_gs.stages.s4_score.run import run_score_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="S4: GPT-5 scoring")
    parser.add_argument("--qids", type=str, required=False, default="ALL")
    parser.add_argument("--run-id", type=str, required=False, default=None)
    parser.add_argument("--model", type=str, required=False, default=None)
    parser.add_argument("--temp", type=float, required=False, default=None)
    parser.add_argument("--top-p", type=float, required=False, default=None)
    args = parser.parse_args()

    cfg = load_config()
    qids = parse_qids(args.qids)
    run_score_stage(qids=qids, run_id=args.run_id, override_model=args.model, override_temperature=args.temp, override_top_p=args.top_p, cfg=cfg)


if __name__ == "__main__":  # pragma: no cover
    main()

