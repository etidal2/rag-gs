import argparse
from rag_gs.core.config import load_config
from rag_gs.core.utils import parse_qids
from rag_gs.stages.s1_embed.run import run_embed_stage


def main() -> None:
    parser = argparse.ArgumentParser(description="S1: query embeddings")
    parser.add_argument("--qids", type=str, required=False, default="ALL")
    parser.add_argument("--tags", type=str, required=False, default=None)
    parser.add_argument("--questions-pack", type=str, required=False, default="example")
    parser.add_argument("--run-id", type=str, required=False, default=None)
    parser.add_argument("--max-batch", type=int, required=False, default=None)
    args = parser.parse_args()

    cfg = load_config()
    qids = parse_qids(args.qids)
    from rag_gs.core.utils import parse_tags

    tags = parse_tags(args.tags)
    run_embed_stage(
        qids=qids,
        tags=tags,
        questions_pack=args.questions_pack,
        run_id=args.run_id,
        override_max_batch=args.max_batch,
        cfg=cfg,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
