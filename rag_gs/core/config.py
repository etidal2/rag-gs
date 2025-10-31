from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Optional .env support (opt-in via RAGGS_LOAD_DOTENV)
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None  # type: ignore


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
TRUTHY_FLAGS = {"1", "true", "yes", "on"}


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        return {}
    return raw


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)


def _maybe_load_dotenv() -> None:
    """Opt-in loading of .env without overriding existing env vars."""
    flag = (os.getenv("RAGGS_LOAD_DOTENV", "").strip().lower())
    if flag not in TRUTHY_FLAGS or load_dotenv is None:
        return
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    try:
        load_dotenv(env_path, override=False)
    except Exception:
        pass


@dataclass
class S1Config:
    model_name: str
    input_type: str
    output_dimension: int
    truncation: bool
    max_voyage_batch: int


@dataclass
class S2Config:
    es_url: str
    es_index: str
    vector_field: str
    similarity: str
    dense_top_k: int
    sparse_top_k: int


@dataclass
class S3Config:
    rrf_c: float


@dataclass
class S4Config:
    model: str
    temperature: float
    top_p: float


@dataclass
class S5Config:
    prune_min: int


@dataclass
class S6Config:
    model_name: str
    stability_turns: int
    confirmations_before_lock: int
    max_turns: int
    margin_z: float
    temperature: float
    top_p: float
    pl_step_size: float
    pl_step_min: float
    pl_clip: float
    pl_recenter_every: int
    pl_decay: str
    seed: int


@dataclass
class Config:
    profile: str
    s1: S1Config
    s2: S2Config
    s3: S3Config
    s4: S4Config
    s5: S5Config
    s6: S6Config


def _apply_env_overrides(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # S1
    max_batch = _env("MAX_VOYAGE_BATCH")
    if max_batch:
        cfg.setdefault("s1", {})["max_voyage_batch"] = int(max_batch)

    # S2
    es_url = _env("ES_URL")
    if es_url:
        cfg.setdefault("s2", {})["es_url"] = es_url

    # S4
    if (m := _env("S4_MODEL")):
        cfg.setdefault("s4", {})["model"] = m

    # S6
    s6 = cfg.setdefault("s6", {})
    for key_env, key_cfg, cast in [
        ("S6_MODEL_NAME", "model_name", str),
        ("S6_STABILITY_TURNS", "stability_turns", int),
        ("S6_CONFIRMATIONS_BEFORE_LOCK", "confirmations_before_lock", int),
        ("S6_MAX_TURNS", "max_turns", int),
        ("S6_MARGIN_Z", "margin_z", float),
        ("S6_TEMPERATURE", "temperature", float),
        ("S6_TOP_P", "top_p", float),
        ("S6_PL_STEP_SIZE", "pl_step_size", float),
        ("S6_PL_STEP_MIN", "pl_step_min", float),
        ("S6_PL_CLIP", "pl_clip", float),
        ("S6_PL_RECENTER_EVERY", "pl_recenter_every", int),
        ("S6_PL_DECAY", "pl_decay", str),
        ("S6_SEED", "seed", int),
    ]:
        val = _env(key_env)
        if val is not None and val != "":
            s6[key_cfg] = cast(val)

    return cfg


def load_config() -> Config:
    # Load .env early so env-based profile/overrides can see it
    _maybe_load_dotenv()

    default = _read_yaml(CONFIG_DIR / "default.yaml")
    local = _read_yaml(CONFIG_DIR / "local.yaml")
    cfg = _merge_dicts(default, local)

    # Allow env override for profile
    env_profile = _env("RAGGS_PROFILE")
    profile_name = (env_profile or cfg.get("profile") or "default")
    profile_path = CONFIG_DIR / "pipelines" / f"{profile_name}.yaml"
    if profile_path.exists():
        profile = _read_yaml(profile_path)
        # Profile can override the top-level keys (like s6.stability_turns)
        cfg = _merge_dicts(cfg, profile)

    # If s2.* not set, inherit from global.* defaults when present
    glob = cfg.get("global", {}) or {}
    s2d = cfg.setdefault("s2", {})
    if "dense_top_k" not in s2d and "dense_k" in glob:
        s2d["dense_top_k"] = int(glob["dense_k"])
    if "sparse_top_k" not in s2d and "sparse_k" in glob:
        s2d["sparse_top_k"] = int(glob["sparse_k"])

    cfg = _apply_env_overrides(cfg)

    s1 = cfg.get("s1", {})
    s2 = cfg.get("s2", {})
    s3 = cfg.get("s3", {})
    s4 = cfg.get("s4", {})
    s5 = cfg.get("s5", {})
    s6 = cfg.get("s6", {})

    return Config(
        profile=cfg.get("profile", "default"),
        s1=S1Config(
            model_name=s1.get("model_name", "voyage-3.5"),
            input_type=s1.get("input_type", "query"),
            output_dimension=int(s1.get("output_dimension", 1024)),
            truncation=bool(s1.get("truncation", True)),
            max_voyage_batch=int(s1.get("max_voyage_batch", 16)),
        ),
        s2=S2Config(
            es_url=s2.get("es_url") or "",
            es_index=s2.get("es_index", "journal_articles_voyage1024_knn_f32_i8"),
            vector_field=s2.get("vector_field", "vector_f32"),
            similarity=s2.get("similarity", "cosine"),
            dense_top_k=int(s2.get("dense_top_k", 50)),
            sparse_top_k=int(s2.get("sparse_top_k", 50)),
        ),
        s3=S3Config(rrf_c=float(s3.get("rrf_c", 60.0))),
        s4=S4Config(
            model=s4.get("model", "gpt-5"),
            temperature=float(s4.get("temperature", 1.0)),
            top_p=float(s4.get("top_p", 1.0)),
        ),
        s5=S5Config(prune_min=int(s5.get("prune_min", 35))),
        s6=S6Config(
            model_name=s6.get("model_name", "gpt-5-judge"),
            stability_turns=int(s6.get("stability_turns", 15)),
            confirmations_before_lock=int(s6.get("confirmations_before_lock", 2)),
            max_turns=int(s6.get("max_turns", 0)),
            margin_z=float(s6.get("margin_z", 1.0)),
            temperature=float(s6.get("temperature", 1.0)),
            top_p=float(s6.get("top_p", 1.0)),
            pl_step_size=float(s6.get("pl_step_size", 0.05)),
            pl_step_min=float(s6.get("pl_step_min", 0.01)),
            pl_clip=float(s6.get("pl_clip", 0.20)),
            pl_recenter_every=int(s6.get("pl_recenter_every", 50)),
            pl_decay=str(s6.get("pl_decay", "inv_sqrt")),
            seed=int(s6.get("seed", 12345)),
        ),
    )
