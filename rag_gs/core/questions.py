from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import yaml

from .config import ROOT


@dataclass(frozen=True)
class Question:
    qid: str
    text: str
    text_rewrite: str
    bm25_query: Optional[dict] = None
    tags: Optional[List[str]] = None


def _resolve_pack_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix.lower() in {".yml", ".yaml"} and p.exists():
        return p
    # treat as pack name
    return ROOT / "configs" / "questions" / f"{name_or_path}.yaml"


def load_questions_pack(name_or_path: str) -> List[Question]:
    path = _resolve_pack_path(name_or_path)
    if not path.exists():
        raise FileNotFoundError(f"Question pack not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = raw.get("questions") or []
    out: List[Question] = []
    for it in items:
        qid = str(it.get("qid"))
        text = it.get("text") or ""
        rewrite = it.get("text_rewrite") or ""
        bm25_query = it.get("bm25_query")
        tags = it.get("tags") or None
        out.append(Question(qid=qid, text=text, text_rewrite=rewrite, bm25_query=bm25_query, tags=tags))
    return out


def filter_questions(
    questions: Sequence[Question], *, qids: Optional[Sequence[str]] = None, tags: Optional[Sequence[str]] = None
) -> List[Question]:
    out = list(questions)
    if qids:
        keep = {q.upper() for q in qids}
        out = [q for q in out if q.qid.upper() in keep]
    if tags:
        want = {t.strip().lower() for t in tags if t}
        out = [q for q in out if q.tags and want.intersection({t.strip().lower() for t in q.tags})]
    return out

