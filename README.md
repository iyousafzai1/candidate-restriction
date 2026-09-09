# Adaptive Candidate Restriction for RF-SMBO

Reference implementation and reproduction code for:

> **Adaptive Candidate Restriction for Scalable Random-Forest Bayesian Optimisation:
> Matched Accuracy at a Fraction of the Acquisition Cost**
> Irfan Khan, Saqib Alam, Ramesh Kumar Ayyasamy, Miao Zhang.
> Submitted to *Computer Modeling in Engineering & Sciences* (CMES).

---

## What this is

Random-forest sequential model-based optimisation (RF-SMBO) normally scores **every**
unevaluated configuration with the acquisition function at each step. On grids of tens of
thousands of configurations that scoring — not the objective evaluations — dominates the
per-iteration cost.

This repository implements **randomised candidate restriction**: maximise the acquisition
over a small random subset of `K` candidates instead of the full pool, with `K` sized online
by the forest's own predictive dispersion (**adaptive-K**).

The rule is a one-line change to any RF-SMBO loop:

```python
# instead of scoring all U unevaluated configurations ...
K_t  = int(min(U, max(K_MIN, round(K_MIN * (s0 / s_hat) ** C_GROW))))
cand = rng.choice(uneval, K_t, replace=False)
```

`s_hat` is the mean per-tree prediction standard deviation over a random probe, and `s0` is
its value at the first surrogate step. While the surrogate is unreliable (`s_hat` large) few
candidates are scored; as it becomes reliable `K_t` grows. Below a pool-size floor `F` the
method reverts to exact full scoring, so small pools are unaffected.

The rule is available as an installable package, `candidate_restriction`, and the
experiment scripts that produced the published numbers live in `experiments/`.

### Frozen constants

Set **once** on a single development space (HPO-B `6767`) and never retuned. The selection
grid is `SWEEP_SETTINGS` in `adaptive_k_runner.py`; reproduce it with `--sweep`.

| Constant | Value | Meaning |
|---|---|---|
| `FULL_FLOOR` (F) | 2500 | below this pool size, score the full pool exactly |
| `K_MIN` | 1000 | early candidate floor |
| `C_GROW` (c) | 0.75 | growth exponent |
| `RF_N_ESTIMATORS` | 64 | trees in the surrogate |
| `RF_UCB_BETA` (β) | 1.96 | UCB exploration weight |
| warm-start | 5 steps | random initial design |

The fixed-`K` variant (`rf_candidate_2k`) uses `K = 2000`.

---

## Installation

```bash
git clone https://github.com/iyousafzai1/candidate-restriction.git
cd candidate-restriction
pip install -e .
```

Python 3.9+. Runtime dependencies are `numpy`, `scipy` and `scikit-learn` only.

For the live-training studies and the test suite:

```bash
pip install -e ".[experiments,dev]"
pytest
```

### HPO-B data

The benchmark experiments need the [HPO-B](https://github.com/machinelearningnanoscale/HPO-B)
repository (for `hpob_handler`) and its v3 data:

```bash
git clone https://github.com/machinelearningnanoscale/HPO-B /tmp/HPO-B
# download and unpack the HPO-B v3 data under /tmp/hpob_data/hpob-data/
```

Locations are configurable:

| Variable | Default |
|---|---|
| `HPOB_REPO` | `/tmp/HPO-B` |
| `HPOB_DATA_ROOT` | `/tmp/hpob_data/hpob-data/` |
| `CACHE_DIR` | per-script default |
| `RUNS_DIR` | `./runs` |

The live-training studies additionally need `xgboost` and `openml`.

---

## Using the package

Apply the schedule inside your own optimiser — the rule is one call and has no
dependency beyond the standard library:

```python
from candidate_restriction import adaptive_k_size, SCORE_ALL

K_t = adaptive_k_size(
    n_unevaluated=len(pool),      # U_t
    dispersion=sigma_hat,         # mean per-tree prediction std over a probe
    reference_dispersion=sigma_0, # the same, at the first restricted step
)
if K_t is SCORE_ALL:              # pool <= F: score everything, exactly as before
    candidates = pool
else:
    candidates = rng.choice(pool, K_t, replace=False)
```

Or run the whole loop:

```python
import numpy as np
from candidate_restriction import RFSMBO

rng = np.random.default_rng(0)
X = rng.normal(size=(30_000, 6))
y = -(X ** 2).sum(axis=1)

for mode in ("full", "adaptive"):
    res = RFSMBO(mode=mode).optimize(X, y, budget=20, seed=0)
    print(f"{mode:9s} best={res.best:7.4f}  "
          f"candidates/step={res.mean_candidates_scored:8.1f}")
```

```
full      best=-1.1866  candidates/step= 29988.0
adaptive  best=-1.4095  candidates/step=  1015.1
```

Same budget, comparable result, **~30x less acquisition work** on this 30,000-point
grid. `mode` is one of `"full"` (exhaustive), `"fixed"` (`rf-cand-2k`, K=2000) or
`"adaptive"`.

### API

| Object | Purpose |
|---|---|
| `adaptive_k_size(...)` | the schedule; returns `K_t`, or `SCORE_ALL` below the floor |
| `fixed_k_size(...)` | the frozen fixed-K variant |
| `RFSMBO(mode=...)` | the full RF-SMBO loop in all three modes |
| `OptimizationResult` | `.best`, `.trajectory`, `.evaluated`, `.mean_candidates_scored` |
| `FULL_FLOOR`, `K_MIN`, `C_GROW`, `FIXED_K`, ... | the frozen constants |

---

## Quick start

Smoke test on the development space only (`6767`), a few minutes on one core:

```bash
export HPOB_REPO=/tmp/HPO-B
export HPOB_DATA_ROOT=/tmp/hpob_data/hpob-data/

python experiments/adaptive_k_runner.py --smoke
```

A single space:

```bash
python experiments/adaptive_k_runner.py --space 5636
```

Re-derive the frozen constants (runs the five-triple grid on `6767` only, in memory):

```bash
python experiments/adaptive_k_runner.py --sweep
```

Re-analyse an existing cache without re-running:

```bash
python experiments/adaptive_k_runner.py --analyze-only
```

---

## Reproducing the paper

The frozen constants are the defaults, so no environment overrides are needed.

| Paper claim | Script | Output in `results/` |
|---|---|---|
| Main accuracy comparison (14 spaces, 157 tasks, 20 seeds) | `experiments/adaptive_k_runner.py` | `table1_main_results.csv`, `canonical_macro.csv` |
| Per-space results at B=25 | `experiments/hpob_phase2_runner.py` | `table2_per_space_b25.csv`, `canonical_per_space_b25.csv` |
| Non-inferiority / statistical tests | `analysis/hpob_phase2_analysis.py` | `table4_statistical_tests.csv`, `adaptive_k_stat_tests.csv` |
| Candidate-count reduction (20–27×) | `experiments/log_candidates.py` | `candidate_cost.csv` |
| Saturation sweep — accuracy is flat in K | `experiments/ksweep_runner.py` | `ksweep_results.csv`, `ksweep_density.csv` |
| Wall-clock acquisition timings | `experiments/timing_benchmark.py` | `timing_results.csv` |
| Coverage mechanism (subset retains a near-max candidate) | `experiments/coverage_mechanism.py` | `coverage_mechanism_raw.csv` |
| Optimiser's-curse measurement | `experiments/curse_instrumentation.py` | `curse_realized_raw.csv`, `curse_scaling_raw.csv` |
| Where restriction loses | `analysis/reanalyze.py` | `table7_failure_analysis.csv`, `supp_loss_table.csv` |
| Live-training validation (XGBoost / RF on OpenML) | `experiments/real_training_runner.py` | `validation_per_task_deltas.csv` |
| Fixed-K across surrogates (TPE, GP-UCB) | `experiments/multi_surrogate_runner.py` | `multi_surrogate_results.csv` |
| Adaptive-K across surrogates | `experiments/multi_surrogate_adaptive_runner.py` | `multi_surrogate_adaptive_results.csv` |
| Final tables and figure data | `analysis/build_final_outputs.py` | `table*.csv`, `fig*.csv` |

### Headline numbers

From `results/candidate_cost.csv` and `results/timing_results.csv`:

| Quantity | Value |
|---|---|
| Mean candidates scored per step (full → adaptive) | 24,248 → 1,210 (**20.0×** fewer) |
| Same, large pools only | 37,364 → 1,371 (**27.2×** fewer) |
| Scoring speedup at N = 52,000 | **16.4×** |
| End-to-end per-step speedup at N = 52,000 | **4.7×** |
| Normalised regret at B=25 (rf-smbo → adaptive) | 0.0859 → 0.0767 |

---

## Layout

```
src/candidate_restriction/   the installable package
    schedule.py              the adaptive-K rule (pure, dependency-free)
    optimizer.py             RF-SMBO in full / fixed / adaptive modes
    constants.py             the frozen constants
experiments/                 run the optimisers and produce raw results
analysis/                    turn raw results into the paper's tables
results/                     the CSVs behind every table and figure
tests/                       unit tests for the rule and the loop
```

`results/` is committed so the paper's numbers can be checked without re-running
anything — every table in the manuscript traces to a file here.

---

## Citation

```bibtex
@article{khan2026candidate,
  author  = {Khan, Irfan and Alam, Saqib and Ayyasamy, Ramesh Kumar and Zhang, Miao},
  title   = {Adaptive Candidate Restriction for Scalable Random-Forest Bayesian
             Optimisation: Matched Accuracy at a Fraction of the Acquisition Cost},
  journal = {Computer Modeling in Engineering \& Sciences},
  year    = {2026},
  note    = {Submitted}
}
```

## License

MIT — see [LICENSE](LICENSE).
