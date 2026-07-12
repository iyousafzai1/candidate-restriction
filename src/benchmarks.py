"""YAHPO-Gym access layer.

Design rules (MVE plan §2/§3):
- All scenario / hyperparameter / metric names coming from configs are treated
  as *candidates* and validated against the live YAHPO package at runtime.
  On mismatch we raise BenchmarkNameError listing the available choices.
- Fidelity parameters not pinned in the config are set to their MAXIMUM
  (full-budget evaluation; no multi-fidelity in the MVE).
- Conditional hyperparameters are resolved against the `fixed` assignment;
  grid hyperparameters must be unconditional or active under `fixed`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


class BenchmarkNameError(ValueError):
    """A scenario / hyperparameter / metric name was not found; message lists options."""


# --------------------------------------------------------------------------- #
# Scenario specification (parsed from YAML)
# --------------------------------------------------------------------------- #

@dataclass
class ScenarioSpec:
    key: str                      # name used for folders (e.g. rbv2_svm_radial)
    scenario: str                 # YAHPO scenario id (e.g. rbv2_svm)
    metric: str
    grid: dict                    # hp_name -> {n: int, scale: auto|linear|log, lower?, upper?}
    fixed: dict = field(default_factory=dict)
    fidelity: dict = field(default_factory=dict)
    max_tasks: int | None = None
    task_seed: int = 0

    @staticmethod
    def from_config(key: str, d: dict) -> "ScenarioSpec":
        return ScenarioSpec(
            key=key,
            scenario=d["scenario"],
            metric=d["metric"],
            grid={k: dict(v or {}) for k, v in d["grid"].items()},
            fixed=dict(d.get("fixed") or {}),
            fidelity=dict(d.get("fidelity") or {}),
            max_tasks=d.get("max_tasks"),
            task_seed=int(d.get("task_seed", 0)),
        )


# --------------------------------------------------------------------------- #
# ConfigSpace compatibility helpers (old 0.x API and new 1.x API)
# --------------------------------------------------------------------------- #

def _cs_hyperparameters(cs):
    if hasattr(cs, "get_hyperparameters"):
        return list(cs.get_hyperparameters())
    return list(cs.values())


def _cs_conditions(cs):
    if hasattr(cs, "get_conditions"):
        return list(cs.get_conditions())
    return list(getattr(cs, "conditions", []))


def _hp_spec(hp) -> dict:
    """Normalize a ConfigSpace hyperparameter into a plain dict."""
    spec = {"name": hp.name, "type": type(hp).__name__}
    if hasattr(hp, "choices"):                       # categorical
        spec.update(kind="categorical", choices=list(hp.choices))
    elif hasattr(hp, "lower"):                       # numeric
        is_int = "Integer" in type(hp).__name__
        spec.update(
            kind="int" if is_int else "float",
            lower=hp.lower, upper=hp.upper, log=bool(getattr(hp, "log", False)),
        )
    if hasattr(hp, "default_value"):
        spec["default"] = hp.default_value
    elif hasattr(hp, "value"):                       # constant
        spec.update(kind="constant", value=hp.value)
    else:
        spec.update(kind="unknown")
    return spec


def _condition_satisfied(cond, assignment: dict) -> bool | None:
    """True/False if decidable from `assignment`, None if the parent is unset.

    ConfigSpace 0.6.x (Cython): InCondition has .value = list (same as
    .values), so membership must be checked with `in`, not `==`.  We resolve
    this by checking if the stored reference is iterable first.
    """
    parent = cond.parent.name
    if parent not in assignment:
        return None
    val = assignment[parent]
    ref = getattr(cond, "values", None)
    if ref is None:
        ref = getattr(cond, "value", None)
    if ref is None:
        return None
    if isinstance(ref, (list, set, frozenset)):
        return val in ref
    return val == ref


# --------------------------------------------------------------------------- #
# YAHPO wrapper
# --------------------------------------------------------------------------- #

class YahpoBenchmark:
    """Thin, validated wrapper around yahpo_gym.benchmark_set.BenchmarkSet."""

    def __init__(self, scenario: str, data_path: Path | str,
                 logger: logging.Logger | None = None,
                 multithread: bool = True, check: bool = False):
        """check=False skips per-config validation in objective_function
        (we validate names/bounds once at axis construction); multithread=False
        avoids ONNX thread contention when running under joblib workers."""
        self.log = logger or logging.getLogger(__name__)
        self.scenario = scenario
        self.data_path = Path(data_path)

        try:
            from yahpo_gym import benchmark_set, local_config
            import yahpo_gym.benchmarks  # noqa: F401  (registers scenarios)
        except ImportError as e:
            raise ImportError(
                "yahpo-gym is not installed or failed to import "
                f"({e}). Install with: pip install yahpo-gym "
                "(and if ConfigSpace errors appear: pip install 'ConfigSpace==0.6.1')"
            ) from e

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"YAHPO data path not found: {self.data_path}. "
                "Run scripts/download_data.py first."
            )
        local_config.init_config()
        local_config.set_data_path(str(self.data_path))

        available = self._available_scenarios()
        if available and scenario not in available:
            raise BenchmarkNameError(
                f"Scenario '{scenario}' not found in YAHPO-Gym. "
                f"Available scenarios: {sorted(available)}"
            )
        try:
            self.bench = benchmark_set.BenchmarkSet(
                scenario, active_session=True,
                multithread=multithread, check=check)
        except Exception as e:
            raise BenchmarkNameError(
                f"Could not load scenario '{scenario}' "
                f"(data missing under {self.data_path}?): {e}. "
                f"Available scenarios: {sorted(available) if available else 'unknown'}"
            ) from e

        cfg = self.bench.config
        self.instance_name: str = self._first_attr(
            cfg, ["instance_names"], default="task_id"
        )
        if isinstance(self.instance_name, (list, tuple)):
            self.instance_name = self.instance_name[0]
        self.y_names: list = list(self._first_attr(cfg, ["y_names"], default=[]))
        self.fidelity_names: list = list(
            self._first_attr(cfg, ["fidelity_params"], default=[])
        )

    # -- introspection ------------------------------------------------------ #

    @staticmethod
    def _available_scenarios() -> list:
        try:
            from yahpo_gym.configuration import list_scenarios
            return list(list_scenarios())
        except Exception:
            try:
                from yahpo_gym.configuration import config_dict
                return list(config_dict.configs.keys())
            except Exception:
                return []

    @staticmethod
    def _first_attr(obj, names: Iterable[str], default=None):
        for n in names:
            if hasattr(obj, n) and getattr(obj, n) is not None:
                return getattr(obj, n)
        return default

    @property
    def instances(self) -> list:
        return [str(i) for i in self.bench.instances]

    def opt_space(self):
        try:
            return self.bench.get_opt_space(drop_fidelity_params=False)
        except TypeError:
            return self.bench.get_opt_space()

    def hyperparameters(self) -> dict:
        """name -> normalized spec dict; excludes the instance id parameter."""
        specs = {}
        for hp in _cs_hyperparameters(self.opt_space()):
            if hp.name == self.instance_name:
                continue
            specs[hp.name] = _hp_spec(hp)
        return specs

    def fidelity_defaults(self) -> dict:
        """Every fidelity parameter at its maximum (full budget)."""
        out = {}
        hps = self.hyperparameters()
        for name in self.fidelity_names:
            if name in hps and hps[name].get("upper") is not None:
                up = hps[name]["upper"]
                out[name] = int(up) if hps[name]["kind"] == "int" else float(up)
        return out

    def active_grid_hps(self, spec: ScenarioSpec) -> dict:
        """Validate grid/fixed/metric names; resolve conditionals; return specs
        of the grid hyperparameters (in the order given in the config)."""
        hps = self.hyperparameters()

        for name in list(spec.fixed) + list(spec.grid):
            if name not in hps:
                raise BenchmarkNameError(
                    f"[{spec.key}] hyperparameter '{name}' not in scenario "
                    f"'{self.scenario}'. Available: {sorted(hps)}"
                )
        if self.y_names and spec.metric not in self.y_names:
            raise BenchmarkNameError(
                f"[{spec.key}] metric '{spec.metric}' not available for "
                f"'{self.scenario}'. Available metrics: {sorted(self.y_names)}"
            )

        conditions = _cs_conditions(self.opt_space())
        assignment = dict(spec.fixed)
        for name in spec.grid:
            for cond in conditions:
                if cond.child.name != name:
                    continue
                sat = _condition_satisfied(cond, assignment)
                if sat is None:
                    raise BenchmarkNameError(
                        f"[{spec.key}] grid hyperparameter '{name}' is conditional on "
                        f"'{cond.parent.name}', which is neither fixed nor decidable. "
                        f"Pin '{cond.parent.name}' under 'fixed:' in the config."
                    )
                if sat is False:
                    raise BenchmarkNameError(
                        f"[{spec.key}] grid hyperparameter '{name}' is INACTIVE under "
                        f"fixed={spec.fixed} (condition on '{cond.parent.name}')."
                    )
        return {name: hps[name] for name in spec.grid}

    def default_fill(self, spec: ScenarioSpec, fidelity: dict):
        """Hyperparameters that are neither gridded, fixed, fidelity, nor the
        instance id would be silently imputed (0 / '#na#') by yahpo's NA
        handling — misleading data. Pin every such *active* hyperparameter to
        its ConfigSpace default instead. Returns (defaults, unresolved):
        `unresolved` are conditional hps whose activity cannot be decided
        from `fixed` (left unset; recorded in meta; loud warning)."""
        hps = self.hyperparameters()
        covered = set(spec.grid) | set(spec.fixed) | set(fidelity)
        conditions = _cs_conditions(self.opt_space())
        assignment = dict(spec.fixed)

        defaults, unresolved = {}, []
        for name, h in hps.items():
            if name in covered:
                continue
            status = True                       # unconditional -> active
            for cond in conditions:
                if cond.child.name != name:
                    continue
                sat = _condition_satisfied(cond, assignment)
                status = sat if sat is not None else None
                if status in (None, False):
                    break
            if status is True:
                if h.get("default") is None:
                    unresolved.append(name)
                else:
                    defaults[name] = h["default"]
            elif status is None:
                unresolved.append(name)
            # status False -> inactive under fixed: correctly left unset
        if unresolved:
            self.log.warning(
                "[%s] hyperparameters left UNSET (yahpo will impute 0/'#na#'): "
                "%s — pin them under 'fixed:' if this matters.",
                spec.key, unresolved)
        if defaults:
            self.log.info("[%s] pinned to ConfigSpace defaults: %s",
                          spec.key, defaults)
        return defaults, unresolved

    # -- querying ------------------------------------------------------------ #

    def query(self, instance: str, configs: list[dict], metric: str,
              chunk: int = 2000) -> np.ndarray:
        """Evaluate `configs` (dicts WITHOUT instance id) on one instance.
        Returns float array aligned with `configs`."""
        if self.y_names and metric not in self.y_names:
            raise BenchmarkNameError(
                f"Metric '{metric}' not in {sorted(self.y_names)} for '{self.scenario}'."
            )
        self.bench.set_instance(instance)
        out = np.empty(len(configs), dtype=np.float64)
        for start in range(0, len(configs), chunk):
            batch = configs[start:start + chunk]
            payload = [{self.instance_name: instance, **c} for c in batch]
            try:
                res = self.bench.objective_function(payload)
            except Exception:
                res = [self.bench.objective_function(p) for p in payload]
                res = [r[0] if isinstance(r, list) else r for r in res]
            if isinstance(res, dict):
                res = [res]
            for j, r in enumerate(res):
                out[start + j] = float(r[metric])
        return out


def make_bench_factory(scenario: str, data_path: Path, multithread: bool = False):
    """Picklable factory so each joblib worker builds its own ONNX session."""
    def _factory():
        return YahpoBenchmark(scenario, data_path, multithread=multithread)
    return _factory
