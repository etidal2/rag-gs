from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .io import write_json


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_stage_manifest(dir_path: Path, payload: Dict[str, Any]) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "finished_at": _now()}
    write_json(dir_path / "manifest.json", payload)
    (dir_path / "_SUCCESS").write_text("", encoding="utf-8")


def write_run_manifest(run_dir: Path, payload: Dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "created_at": _now()}
    write_json(run_dir / "manifest.json", payload)

