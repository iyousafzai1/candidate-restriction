# Paper D: Tensor-Completion Warm-Started Bayesian Optimization
# Shared utilities: paths, config, logging, seeding, atomic IO, hashing.

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


def project_root() -> Path:
    """Repository root = two levels above this file (src/atc → repo)."""
    return Path(__file__).resolve().parents[1]


def ensure_project_dirs(root: Path | None = None) -> Path:
    root = Path(root) if root else project_root()
    for d in ["data/raw", "data/grids", "results/raw", "results/agg",
              "figures", "logs"]:
        (root / d).mkdir(parents=True, exist_ok=True)
    return root


def load_config(root: Path | None = None) -> dict:
    root = Path(root) if root else project_root()
    cfg_path = root / "configs" / "paper_d.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def setup_logging(name: str, root: Path | None = None) -> logging.Logger:
    root = ensure_project_dirs(root)
    day_dir = root / "logs" / datetime.now().strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%H%M%S")
    log_path = day_dir / f"{name}_{ts}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)):
        h.setFormatter(fmt)
        logger.addHandler(h)
    logger.info("Logging to %s", log_path)
    return logger


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def config_hash(cfg: dict) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    blob = json.dumps(clean, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def scoped_hash(cfg: dict, keys: list[str]) -> str:
    return config_hash({k: cfg.get(k) for k in keys})


def atomic_write_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2, default=str)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def atomic_save_npz(path: Path, **arrays) -> None:
    """Atomically write a (compressed) .npz of named arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npz.tmp")
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_hash(root: Path | None = None) -> str:
    """Short git commit hash of the repo, or 'nogit' if unavailable."""
    import subprocess
    root = Path(root) if root else project_root()
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"
