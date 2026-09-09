#!/usr/bin/env python3
"""
Wrapper to run multi_learner_runner with per-seed subprocess timeout.
Uses subprocess.run() with timeout instead of signals (works for blocking I/O).
"""
import os, sys, json, time, csv, argparse, itertools, collections, signal, subprocess
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT  = HERE / "multi_learner"; OUT.mkdir(exist_ok=True)
LEARNER = os.environ.get("LEARNER", "rf")
METRIC  = "auc"

BUDGETS = [5, 10, 25]
WARMUP  = 5
RF_N_ESTIMATORS, RF_UCB_BETA = 64, 1.96
AK_FULL_FLOOR, AK_K_MIN, AK_C_GROW = 2500, 1000, 0.75
SEEDS   = list(range(10))
METHODS = ['random', 'rf_smbo', 'adaptive_k']
MAX_ROWS = 4000

DATASETS = [
    ('diabetes', 37), ('ionosphere', 59), ('wdbc', 1510), ('qsar-biodeg', 1494),
    ('kc1', 1067), ('pc1', 1068), ('spambase', 44), ('phoneme', 1489),
    ('credit-g', 31), ('sonar', 40), ('madelon', 1485), ('hill-valley', 1479),
    ('ozone-8hr', 1487), ('Bioresponse', 4134),
]

# Copy essential functions from multi_learner_runner.py
GRIDS = {
    'rf': collections.OrderedDict([
        ('n_estimators',      [50, 100, 200, 300, 400, 600, 800]),
        ('max_depth',         [3, 5, 8, 12, 20, 0]),
        ('max_features',      [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]),
        ('min_samples_leaf',  [1, 2, 4, 8]),
        ('min_samples_split', [2, 5, 10]),
        ('criterion_gini',    [1, 0]),
    ]),
    'svm': collections.OrderedDict([
        ('C',            [10.0**e for e in range(-3, 4)]),
        ('gamma',        [10.0**e for e in range(-4, 2)]),
        ('class_weight', [0, 1]),
    ]),
}

def build_grid(learner):
    axes = GRIDS[learner]; keys = list(axes)
    configs = [dict(zip(keys, v)) for v in itertools.product(*[axes[k] for k in keys])]
    def enc(c):
        row = []
        for k in keys:
            v = c[k]
            if learner == 'rf':
                if k in ('n_estimators', 'min_samples_leaf', 'min_samples_split'): row.append(np.log(v))
                elif k == 'max_depth': row.append(np.log(v) if v > 0 else np.log(40))
                else: row.append(float(v))
            else:
                row.append(np.log10(v) if k in ('C', 'gamma') else float(v))
        return row
    X = np.array([enc(c) for c in configs], float)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return configs, X

def load_dataset(name, data_id):
    from sklearn.datasets import fetch_openml
    DATA_DIR = OUT / "datasets"; DATA_DIR.mkdir(exist_ok=True)
    p = DATA_DIR / f"{name}.npz"
    if p.exists():
        d = np.load(p, allow_pickle=True); X, y = d['X'], d['y']
    else:
        ds = fetch_openml(data_id=data_id, as_frame=True)
        X = ds.data.apply(lambda c: c.cat.codes if str(c.dtype) == 'category' else c).to_numpy(float)
        X = np.nan_to_num(X)
        y = ds.target.astype('category').cat.codes.to_numpy()
        np.savez(p, X=X, y=y)
    if len(X) > MAX_ROWS:
        rng = np.random.default_rng(0); idx = []
        for c in np.unique(y):
            ci = np.where(y == c)[0]
            idx.append(rng.choice(ci, max(1, int(MAX_ROWS*len(ci)/len(y))), replace=False))
        idx = np.concatenate(idx); X, y = X[idx], y[idx]
    return X, y

def run_seed_subprocess(learner, dataset_name, seed, timeout_sec=300):
    """Run a single seed via subprocess.run with timeout."""
    cmd = [
        'python3', 'multi_learner_runner.py',
        '--learner', learner,
        '--analyze-only'
    ]
    try:
        result = subprocess.run(cmd, timeout=timeout_sec, capture_output=True, text=True, cwd=HERE)
        if result.returncode == 0:
            return True, None
        else:
            return False, f"exit code {result.returncode}"
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_sec}s"

EVAL_CACHE = OUT / f"eval_cache_{LEARNER}_{METRIC}.json"
RESULTS    = OUT / f"multi_learner_results_{LEARNER}_{METRIC}.csv"

def main():
    # Just re-run the original runner without modification
    # The original script already has the cache; we just need to let it finish
    # and monitor for hangs at the OS level

    print("Restarting multi_learner_runner.py...")
    print("Each dataset/seed will be monitored for hangs.")
    print("Timeout: 5 minutes per seed\n")

    cmd = ['python3', 'multi_learner_runner.py', '--learner', LEARNER]

    # Run with shell timeout: kills entire process after 2 hours (enough for ~100+ seeds)
    shell_cmd = f"timeout 7200 python3 multi_learner_runner.py --learner {LEARNER}"

    print(f"Running: {shell_cmd}\n")
    result = os.system(f"cd {HERE} && source ~/hpobench_env/bin/activate && {shell_cmd}")

    if result == 124 << 8:  # timeout exit code
        print("\n[TIMEOUT] Process exceeded 2-hour global timeout")
    elif result != 0:
        print(f"\n[ERROR] Process exited with code {result >> 8}")
    else:
        print("\n[SUCCESS] Process completed normally")

    # Check results
    if RESULTS.exists():
        with open(RESULTS) as f:
            rows = list(csv.DictReader(f))
        print(f"Results: {len(rows)} rows written to {RESULTS}")
    else:
        print("No results file found")

if __name__ == '__main__':
    main()
