"""Cartesian grid construction and tensor assembly (MVE plan §2/§4).

A grid axis is built per hyperparameter from the live config-space bounds
(optionally overridden in the YAML). Querying the surrogate at every grid
point x task yields a DENSE ground-truth tensor of shape (n1,...,nd, N).
"""
from __future__ import annotations

import itertools
import logging
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from benchmarks import ScenarioSpec, BenchmarkNameError
import utils


# --------------------------------------------------------------------------- #
# Axes
# --------------------------------------------------------------------------- #

def make_axis(hp_spec: dict, grid_cfg: dict, name: str) -> np.ndarray:
    """Grid values for one hyperparameter."""
    kind = hp_spec["kind"]
    if kind == "categorical":
        choices = grid_cfg.get("choices", hp_spec["choices"])
        bad = [c for c in choices if c not in hp_spec["choices"]]
        if bad:
            raise BenchmarkNameError(
                f"Axis '{name}': choices {bad} not in {hp_spec['choices']}"
            )
        return np.array(choices, dtype=object)

    if kind not in ("int", "float"):
        raise BenchmarkNameError(f"Axis '{name}': unsupported kind '{kind}'")

    n = int(grid_cfg.get("n", 5))
    lo = float(grid_cfg.get("lower", hp_spec["lower"]))
    hi = float(grid_cfg.get("upper", hp_spec["upper"]))
    if not (hi > lo):
        raise ValueError(f"Axis '{name}': upper ({hi}) must exceed lower ({lo})")

    scale = grid_cfg.get("scale", "auto")
    use_log = (scale == "log") or (scale == "auto" and hp_spec.get("log", False))
    if use_log and lo <= 0:
        logging.getLogger(__name__).warning(
            "Axis '%s': log scale requested but lower=%s <= 0; using linear.", name, lo
        )
        use_log = False

    vals = np.geomspace(lo, hi, n) if use_log else np.linspace(lo, hi, n)
    if kind == "int":
        vals = np.unique(np.round(vals).astype(np.int64))
        if len(vals) < n:
            logging.getLogger(__name__).info(
                "Axis '%s': %d unique ints after rounding (requested %d).",
                name, len(vals), n,
            )
    return vals


def build_axes(bench, spec: ScenarioSpec) -> list[tuple[str, np.ndarray]]:
    """Ordered [(hp_name, values)] following the config's grid order.
    Validates names/conditionals via bench.active_grid_hps()."""
    hp_specs = bench.active_grid_hps(spec)
    return [(name, make_axis(hp_specs[name], spec.grid[name], name))
            for name in spec.grid]


def grid_configs(axes: list[tuple[str, np.ndarray]], extra: dict) -> list[dict]:
    """All grid points as config dicts (C-order, matching reshape below)."""
    names = [n for n, _ in axes]
    out = []
    for combo in itertools.product(*[vals for _, vals in axes]):
        cfg = dict(zip(names, combo))
        for k, v in cfg.items():
            if isinstance(v, np.generic):
                cfg[k] = v.item()
        cfg.update(extra)
        out.append(cfg)
    return out


# --------------------------------------------------------------------------- #
# Tensor build
# --------------------------------------------------------------------------- #

def _build_one_instance(scenario: str, data_path: str, instance: str,
                        configs: list[dict], metric: str,
                        shape: tuple) -> np.ndarray:
    """Worker: fresh benchmark (own ONNX session), query all grid points.
    multithread=False: each worker stays single-threaded under joblib."""
    from benchmarks import YahpoBenchmark
    bench = YahpoBenchmark(scenario, Path(data_path), multithread=False)
    vals = bench.query(instance, configs, metric)
    return vals.reshape(shape)


def select_tasks(instances: list[str], max_tasks: int | None, seed: int) -> list[str]:
    if max_tasks is None or max_tasks >= len(instances):
        return list(instances)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(instances), size=max_tasks, replace=False)
    return [instances[i] for i in sorted(idx)]


def build_tensor(spec: ScenarioSpec, data_path: Path, n_jobs: int = 1,
                 logger: logging.Logger | None = None):
    """Returns (tensor[n1..nd, N], axes, task_ids, build_info)."""
    log = logger or logging.getLogger(__name__)
    from benchmarks import YahpoBenchmark
    bench = YahpoBenchmark(spec.scenario, data_path, logger=log)

    axes = build_axes(bench, spec)
    shape = tuple(len(v) for _, v in axes)

    fidelity = bench.fidelity_defaults()
    fidelity.update(spec.fidelity)           # explicit pins win
    defaults, unresolved = bench.default_fill(spec, fidelity)
    extra = {**defaults, **spec.fixed, **fidelity}
    configs = grid_configs(axes, extra)

    tasks = select_tasks(bench.instances, spec.max_tasks, spec.task_seed)
    log.info("[%s] grid shape=%s (%d cells), %d tasks, fixed=%s, fidelity=%s",
             spec.key, shape, len(configs), len(tasks), spec.fixed, fidelity)

    t0 = time.time()
    slices = Parallel(n_jobs=n_jobs)(
        delayed(_build_one_instance)(
            spec.scenario, str(data_path), t, configs, spec.metric, shape
        )
        for t in tasks
    )
    elapsed = time.time() - t0
    tensor = np.stack(slices, axis=-1)       # (n1,...,nd, N)
    log.info("[%s] built tensor %s in %.1fs (%.2f ms/query)",
             spec.key, tensor.shape, elapsed,
             1000 * elapsed / max(1, tensor.size))

    # Filter constant task slices: no recommendation signal, and they break
    # normalized-regret arithmetic downstream (y* - y_min = 0).
    tensor, tasks, removed, near_constant = filter_constant_tasks(
        tensor, tasks, log=log, key=spec.key)

    info = {"build_seconds": round(elapsed, 2), "fidelity_used": fidelity,
            "fixed_used": spec.fixed, "defaults_used": defaults,
            "unset_hyperparameters": unresolved,
            "n_queries": int(np.prod(tensor.shape[:-1])) * (len(tasks) + len(removed)),
            "removed_constant_task_ids": removed,
            "near_constant_task_ids": near_constant}
    return tensor, axes, tasks, info


CONSTANT_RANGE_TOL = 1e-8     # slice dropped: max-min below this
NEAR_CONSTANT_RANGE = 1e-3    # slice kept but flagged (regret will be noisy)


def filter_constant_tasks(tensor: np.ndarray, tasks: list[str],
                          log: logging.Logger | None = None, key: str = ""):
    """Drop task slices with (numerically) zero performance range.
    Returns (tensor, kept_tasks, removed_ids, near_constant_ids)."""
    log = log or logging.getLogger(__name__)
    rng = np.array([np.nanmax(tensor[..., k]) - np.nanmin(tensor[..., k])
                    for k in range(tensor.shape[-1])])
    keep = rng > CONSTANT_RANGE_TOL
    removed = [t for t, k in zip(tasks, keep) if not k]
    near = [t for t, k, r in zip(tasks, keep, rng)
            if k and r < NEAR_CONSTANT_RANGE]
    if removed:
        log.warning("[%s] removed %d CONSTANT task slice(s): %s",
                    key, len(removed), removed)
    if near:
        log.warning("[%s] kept %d NEAR-constant task slice(s) (range < %g): %s",
                    key, len(near), NEAR_CONSTANT_RANGE, near)
    kept_tasks = [t for t, k in zip(tasks, keep) if k]
    return tensor[..., keep], kept_tasks, removed, near


# --------------------------------------------------------------------------- #
# Split / save / load
# --------------------------------------------------------------------------- #

def split_tasks(task_ids: list[str], test_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    n = len(task_ids)
    n_test = max(1, int(round(n * test_frac)))
    test_idx = set(rng.choice(n, size=n_test, replace=False).tolist())
    train = [t for i, t in enumerate(task_ids) if i not in test_idx]
    test = [t for i, t in enumerate(task_ids) if i in test_idx]
    return train, test


def grid_dir(root: Path, mode: str, key: str) -> Path:
    """Mode-namespaced grid location. Debug and MVE must NEVER share paths:
    a stale debug grid silently masquerading as MVE data caused a bad
    Gate 1/2 reading once already."""
    return Path(root) / "data" / "grids" / mode / key


def save_grid(root: Path, spec: ScenarioSpec, tensor: np.ndarray,
              axes, task_ids, split_cfg: dict, cfg: dict, info: dict) -> Path:
    out_dir = grid_dir(root, cfg.get("mode", "default"), spec.key)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = {"tensor": tensor.astype(np.float32),
              "task_ids": np.array(task_ids, dtype=object)}
    for name, vals in axes:
        arrays[f"axis__{name}"] = vals
    npz_path = out_dir / "tensor.npz"
    utils.atomic_save_npz(npz_path, **arrays)

    train, test = split_tasks(task_ids, split_cfg["test_frac"], split_cfg["seed"])
    finite = np.isfinite(tensor)
    meta = {
        "mode": cfg.get("mode", "default"),
        "removed_constant_task_ids": info.get("removed_constant_task_ids", []),
        "near_constant_task_ids": info.get("near_constant_task_ids", []),
        "scenario_key": spec.key,
        "yahpo_scenario": spec.scenario,
        "metric": spec.metric,
        "axis_names": [n for n, _ in axes],
        "axes": {n: v.tolist() for n, v in axes},
        "shape": list(tensor.shape),
        "n_cells": int(np.prod(tensor.shape[:-1])),
        "n_tasks": len(task_ids),
        "task_ids": list(task_ids),
        "train_task_ids": train,
        "test_task_ids": test,
        "split": split_cfg,
        "density": float(finite.mean()),
        "value_min": float(np.nanmin(tensor)),
        "value_max": float(np.nanmax(tensor)),
        "build_info": info,
        "config_hash": utils.config_hash(cfg),
        "git_hash": utils.git_hash(root),
        "created_at": utils.now_iso(),
        "npz_sha256": utils.sha256_of_file(npz_path),
    }
    utils.atomic_write_json(meta, out_dir / "meta.json")
    return out_dir


def load_grid(grid_dir: Path) -> dict:
    import json
    grid_dir = Path(grid_dir)
    with open(grid_dir / "meta.json") as f:
        meta = json.load(f)
    data = np.load(grid_dir / "tensor.npz", allow_pickle=True)
    axes = [(n, data[f"axis__{n}"]) for n in meta["axis_names"]]
    return {"tensor": data["tensor"].astype(np.float64),
            "axes": axes,
            "task_ids": [str(t) for t in data["task_ids"]],
            "meta": meta}


def normalize_per_task(tensor: np.ndarray) -> np.ndarray:
    """Z-score each task slice (last mode). Used by diagnostics so that
    cross-task scale differences do not masquerade as rank."""
    out = np.empty_like(tensor, dtype=np.float64)
    for k in range(tensor.shape[-1]):
        sl = tensor[..., k]
        mu, sd = np.nanmean(sl), np.nanstd(sl)
        out[..., k] = (sl - mu) / (sd if sd > 1e-12 else 1.0)
    return out
