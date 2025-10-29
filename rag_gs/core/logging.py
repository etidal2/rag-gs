from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict


def setup_logger(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def log_event(event: str, **fields: Any) -> None:
    payload: Dict[str, Any] = {"event": event}
    payload.update(fields)
    logging.info(json.dumps(payload, ensure_ascii=False))

