from __future__ import annotations

import hashlib
from typing import Iterable, List, Optional, Sequence


def normalize_qid(token: str) -> Optional[str]:
    t = (token or "").strip().upper()
    if not t:
        return None
    if t.startswith("Q"):
        t = t[1:]
    try:
        n = int(t)
    except ValueError:
        return None
    return f"Q{n}"


def parse_qids(raw: str, *, default: Optional[Sequence[str]] = None) -> List[str]:
    if not raw:
        return list(default or [])
    up = raw.strip().upper()
    if up in {"ALL", "*"}:
        return list(default or [])
    out: List[str] = []
    for tok in up.split(','):
        q = normalize_qid(tok)
        if q:
            out.append(q)
    return out


def chunked(seq: Sequence, size: int) -> Iterable[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_tags(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    return [t.strip() for t in raw.split(',') if t.strip()]
