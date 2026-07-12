#!/usr/bin/env python3
"""Paper D experiment runner.

Run unit = (scenario, method, test_task, seed).
Resume-safe: one atomic JSON per unit.

Examples:
  python scripts/run_experiments.py --experiment main --n_jobs 8
  python scripts/run_experiments.py --experiment acq_ablation --scenario lcbench
"""
from __future__ import annotations

import argparse, json, os, sys, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")

from joblib import Parallel, delayed

# src/ is on sys.path — import as flat modules
import utils
import tensorize


def unit_path(root, experiment, scenario, method, task, seed) -> Path:
    return (root / "results" / "raw" / experiment / scenario / method /
            f"{task}__seed{seed}.json")


def unit_done(path, cfg_hash, accept_any=False):
    if not path.exists():
        return False
    try:
        with open(path) as f:
            return accept_any or json.load(f).get("_config_hash") == cfg_hash
    except Exception:
        return False


def run_one(grid_dir, scenario, method, task, seed, exp_cfg, out_path,
            cfg_hash, sweep_overrides=None):
    from prep import Problem
    from protocol import run_unit
    import utils as ut

    try:
        grid = tensorize.load_grid(Path(grid_dir))
        meta = grid["meta"]
        axes = meta.get("axes", None)
        axis_names = meta.get("axis_names", None)

        ecfg = dict(exp_cfg)
        if axis_names:
            ecfg["_axis_names"] = axis_names

        # split_robustness re-splits tasks per split_seed: the runner injects
        # the split's train set and anchor seed so the historical context (and
        # informed anchors) genuinely change across splits.
        train_ids = ecfg.get("_train_task_ids") or meta["train_task_ids"]
        anchor_seed = int(ecfg.get("_anchor_seed",
                                   ecfg.get("split", {}).get("seed", 1234)))

        problem = Problem(grid["tensor"], grid["task_ids"],
                          train_ids, task, k0=ecfg.get("k0", 3),
                          anchor_mode=ecfg.get("anchor_mode", "informed"),
                          anchor_seed=anchor_seed)

        rec = run_unit(method, problem, seed, ecfg, axes=axes)
        # Ceiling-effect diagnostic: regret reachable from the k0 anchors
        # alone, before any sequential search. Emitted for every method/unit.
        anchor_best = float(problem.evaluate(problem.anchors).max())
        rec["regret_at_anchors"] = float(problem.normalized_regret(anchor_best))
        rec["anchor_mode"] = problem.anchor_mode
        rec.update(scenario=scenario, task=task,
                   _config_hash=cfg_hash, _created_at=ut.now_iso())
        ut.atomic_write_json(rec, Path(out_path))
        return f"ok {scenario}/{method}/{task}/s{seed}"
    except Exception as e:
        return f"FAIL {scenario}/{method}/{task}/s{seed}: {e}\n{traceback.format_exc(limit=3)}"


_BO_OVERRIDE_KEYS = {
    "kernel", "acquisition", "portfolio_size", "xi", "pibo_beta",
    "prior_corruption", "prior_corruption_strength", "length_scale_bounds",
    "refit_budget",
}


def _apply_overrides(ecfg, overrides):
    """Route a sweep's overrides into the right config section.

    Deep-copies the mutated nested sections so concurrent jobs never share
    state. Recognised routings: lrtc_rank/lrtc_beta -> lrtc.*, ucb_kappa ->
    bo (and switches acquisition to ucb), the BO keys above -> bo.*, k0 and
    anchor_mode -> top level. Unknown keys land at the top level.
    """
    import copy
    ecfg = dict(ecfg)
    ecfg["lrtc"] = copy.deepcopy(ecfg.get("lrtc", {}))
    ecfg["bo"] = copy.deepcopy(ecfg.get("bo", {}))
    for k, v in overrides.items():
        if k == "lrtc_rank":
            ecfg["lrtc"]["rank"] = v
        elif k == "lrtc_beta":
            ecfg["lrtc"]["beta"] = v
        elif k == "ucb_kappa":
            ecfg["bo"]["ucb_kappa"] = v
            ecfg["bo"]["acquisition"] = "ucb"
        elif k in _BO_OVERRIDE_KEYS:
            ecfg["bo"][k] = v
        else:                              # k0, anchor_mode, misc -> top level
            ecfg[k] = v
    return ecfg


def expand_sweep(base_cfg):
    """Expand sweep config into list of (overrides, label) tuples."""
    sweep = base_cfg.get("sweep", {})
    if not sweep:
        return [({}, "")]
    import itertools
    keys = list(sweep.keys())
    values = list(sweep.values())
    results = []
    for combo in itertools.product(*values):
        overrides = dict(zip(keys, combo))
        label = "_".join(f"{k}={v}" for k, v in overrides.items())
        results.append((overrides, label))
    return results


def merged_exp_cfg(cfg, experiment):
    """Merge shared top-level model sections into the experiment block."""
    exp_key = f"experiment_{experiment}"
    exp_cfg = dict(cfg[exp_key])
    for shared in ("pmf", "lrtc", "bo", "active_pmf", "tpe", "rgpe", "split"):
        if shared in cfg:
            exp_cfg.setdefault(shared, cfg[shared])
    cfg_hash = utils.scoped_hash(cfg, [exp_key, "split", "scenarios", "pmf",
                                       "lrtc", "bo", "active_pmf", "rgpe"])
    return exp_cfg, cfg_hash


def iter_units(root, experiment, scenarios=None, methods=None, max_tasks=None):
    """Enumerate every (scenario, method, task, seed, sweep) unit as a job tuple
    compatible with run_one. Single source of truth for the joblib runner AND
    the elastic workers (worker.py)."""
    cfg = utils.load_config(root)
    exp_cfg, cfg_hash = merged_exp_cfg(cfg, experiment)
    scenarios = scenarios or exp_cfg["scenarios"]
    methods = methods or exp_cfg.get("methods",
                                     exp_cfg.get("method_base", [exp_cfg.get("method")]))
    if isinstance(methods, str):
        methods = [methods]
    STOCHASTIC = {"rs", "tpe", "active_pmf", "rgpe"}
    n_seeds = int(exp_cfg["n_seeds"])
    n_seeds_det = int(exp_cfg.get("n_seeds_deterministic", n_seeds))

    def seeds_for(m):
        return list(range(n_seeds if m in STOCHASTIC else n_seeds_det))

    sweeps = expand_sweep(exp_cfg)
    jobs = []
    for scen in scenarios:
        gdir = tensorize.grid_dir(root, "paper_d", scen)
        if not (gdir / "tensor.npz").exists():
            continue
        with open(gdir / "meta.json") as _mf:   # meta only — don't load tensor
            meta = json.load(_mf)
        split_seeds = exp_cfg.get("split_seeds")
        if split_seeds:
            tf = float(exp_cfg.get("split", {}).get("test_frac", 0.3))
            contexts = []
            for ss in split_seeds:
                tr, te = tensorize.split_tasks(meta["task_ids"], tf, ss)
                contexts.append((f"__split{ss}", tr, te, int(ss)))
        else:
            contexts = [("", meta["train_task_ids"], meta["test_task_ids"], None)]
        for split_label, train_ids, test_tasks, ss in contexts:
            if max_tasks is not None:
                test_tasks = test_tasks[:max_tasks]
            for task in test_tasks:
                for method in methods:
                    for seed in seeds_for(method):
                        for overrides, sweep_label in sweeps:
                            m_label = (f"{method}_{sweep_label}" if sweep_label
                                       else method) + split_label
                            out = unit_path(root, experiment, scen,
                                            m_label, task, seed)
                            ecfg = _apply_overrides(dict(exp_cfg), overrides)
                            if ss is not None:
                                ecfg["_train_task_ids"] = train_ids
                                ecfg["_anchor_seed"] = ss
                            jobs.append((str(gdir), scen, method, task, seed,
                                         ecfg, str(out), cfg_hash, overrides))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True,
                    choices=["main", "acq_ablation", "kernel_ablation",
                             "portfolio_ablation", "k0_sensitivity",
                             "lrtc_sensitivity", "split_robustness",
                             "prior_robustness", "anchor_robustness",
                             "is_external_validation",
                             "is_hardcase_recovery",
                             "full_noniaml_multik0_rerun",
                             "is_external_validation_multisplit",
                             "real_svm_validation",
                             "smoke"])
    ap.add_argument("--scenario", action="append", default=None)
    ap.add_argument("--method", action="append", default=None)
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--max_tasks", type=int, default=None,
                    help="Subsample at most N test tasks per scenario "
                         "(deterministic by task order) — for fast diagnostics.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--accept-any-hash", action="store_true")
    args = ap.parse_args()

    root = utils.ensure_project_dirs()
    log = utils.setup_logging(f"run_{args.experiment}")
    cfg = utils.load_config(root)

    exp_key = f"experiment_{args.experiment}"
    if exp_key not in cfg:
        log.error("Config section '%s' not found.", exp_key)
        sys.exit(1)

    # Merge the shared top-level model sections into the experiment config:
    # protocol.run_unit reads exp_cfg["lrtc"], exp_cfg["bo"], etc., which live
    # at the top level of the YAML, not inside each experiment_* block.
    exp_cfg = dict(cfg[exp_key])
    for shared in ("pmf", "lrtc", "bo", "active_pmf", "tpe", "rgpe", "split"):
        if shared in cfg:
            exp_cfg.setdefault(shared, cfg[shared])
    cfg_hash = utils.scoped_hash(cfg, [exp_key, "split", "scenarios", "pmf",
                                        "lrtc", "bo", "active_pmf", "rgpe"])

    scenarios = args.scenario or exp_cfg["scenarios"]
    methods = args.method or exp_cfg.get("methods",
                                         exp_cfg.get("method_base", [exp_cfg.get("method")]))
    if isinstance(methods, str):
        methods = [methods]

    # B5: evaluations are noise-free surrogate look-ups and anchors are
    # fixed per task, so deterministic methods barely vary across seeds —
    # their statistical power comes from the *task* population, not seeds.
    # Run them with fewer seeds to save compute (configurable; defaults to
    # the full count so behaviour is unchanged unless set). Genuinely
    # stochastic methods (rs, tpe, active_pmf, rgpe) keep all seeds.
    STOCHASTIC = {"rs", "tpe", "active_pmf", "rgpe"}
    n_seeds = int(exp_cfg["n_seeds"])
    n_seeds_det = int(exp_cfg.get("n_seeds_deterministic", n_seeds))

    def seeds_for(method_name):
        is_stoch = method_name in STOCHASTIC
        return list(range(n_seeds if is_stoch else n_seeds_det))

    sweeps = expand_sweep(exp_cfg)

    jobs = iter_units(root, args.experiment, args.scenario,
                      args.method, args.max_tasks)
    todo, skipped = [], 0
    for j in jobs:
        if not args.force and unit_done(Path(j[6]), j[7], args.accept_any_hash):
            skipped += 1
            continue
        todo.append(j)
    log.info("%d to run, %d skipped.", len(todo), skipped)
    if not todo:
        log.info("Nothing to do.")
        return

    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_one)(*j) for j in todo)
    fails = [r for r in results if r.startswith("FAIL")]
    for f in fails:
        log.error(f)
    log.info("Done: %d ok, %d failed.", len(results) - len(fails), len(fails))
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    main()
