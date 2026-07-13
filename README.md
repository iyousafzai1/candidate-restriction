# Adaptive Candidate Restriction for RF-SMBO

This repository contains the implementation and experiments for the paper "Adaptive candidate restriction for scalable random-forest Bayesian optimization: matched accuracy at a fraction of the acquisition cost" submitted to Computer Modeling in Engineering & Sciences (CMES).

## Overview

This paper proposes an adaptive candidate restriction ($K$) strategy for random-forest sequential model-based optimization (RF-SMBO). The method achieves matched accuracy with significantly reduced acquisition function evaluation costs, making large-scale Bayesian optimization more practical and cost-effective.

## Structure

```
├── src/                          # Core implementation
│   ├── models/                   # Optimization methods
│   │   ├── baselines.py         # Baseline optimizers
│   │   ├── bo.py                # Bayesian optimization core
│   │   ├── lrtc.py              # Low-rank tensor completion
│   │   └── pmf.py               # Probabilistic matrix factorization
│   ├── benchmarks.py            # Benchmark datasets and tasks
│   ├── diagnostics.py           # Analysis utilities
│   ├── feasibility.py           # Feasibility checks
│   ├── metrics.py               # Evaluation metrics
│   ├── prep.py                  # Data preparation
│   ├── protocol.py              # Experimental protocol
│   ├── tensorize.py             # Tensor operations
│   └── utils.py                 # Utility functions
├── scripts/                      # Experiment and analysis scripts
│   ├── run_experiments.py       # Main experiment runner
│   ├── generate_figures.py      # Figure generation
│   ├── analyze_main_results.py  # Results analysis
│   └── ddf_deploy.py            # DDF deployment utilities
├── tests/                        # Unit tests
│   └── test_methods_synthetic.py # Method tests on synthetic data
└── README.md                     # This file
```

## Installation

### Requirements
- Python 3.9+
- Dependencies listed in requirements.txt

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/candidate-restriction.git
cd candidate-restriction

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m pytest tests/
```

## Quick Start

### Run Experiments

```bash
# Run main experiment suite
python scripts/run_experiments.py --experiment main --n_jobs 8

# Run acquisition function ablation
python scripts/run_experiments.py --experiment acq_ablation --scenario lcbench

# Run on specific scenario
python scripts/run_experiments.py --experiment main --scenario hpob --n_jobs 4
```

### Generate Figures and Tables

```bash
# Generate all figures
python scripts/generate_figures.py

# Analyze main results
python scripts/analyze_main_results.py
```

### Deploy DDF (Distributed Deployment Framework)

```bash
python scripts/ddf_deploy.py --config config.yaml
```

## Datasets

The experiments use publicly available benchmarks:

- **HPO-B**: Bayesian optimization benchmark suite
- **OpenML**: Machine learning dataset repository
- **YAHPO**: Yet Another Hyperparameter Optimization benchmark

These are automatically downloaded when experiments are run (internet connection required).

## Methods

The core method (`src/models/bo.py`) implements:

1. **Adaptive-K Candidate Restriction**: Dynamically selects the number of candidates to evaluate based on budget and performance
2. **Random Forest Surrogate Model**: Uses RF for efficient surrogate modeling
3. **Expected Improvement Acquisition Function**: Standard EI-based candidate selection
4. **Cost-Aware Optimization**: Tracks and minimizes acquisition function evaluation costs

## Experiments

### Scenarios
- `hpob`: HPO-Bench benchmark
- `lcbench`: LCBench Multi-Fidelity Benchmark
- `openml`: OpenML suite

### Methods Compared
- Adaptive-$K$ (proposed): Our method with dynamic candidate restriction
- Random Forest BO (fixed-$K$): Baseline with fixed candidate count
- Other baselines: SMAC, Optuna, etc.

## Results

Key findings:
- Adaptive-$K$ matches fixed-$K$ performance at 40-60% of acquisition function cost
- Consistent speedup across diverse benchmark scenarios
- Particularly effective for high-dimensional problems

## Citation

```bibtex
@article{adaptive2024,
  title={Adaptive candidate restriction for scalable random-forest {B}ayesian optimisation: 
         matched accuracy at a fraction of the acquisition cost},
  journal={Computer Modeling in Engineering \& Sciences (CMES)},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Reproducibility

All experiments are deterministic given a random seed. To reproduce:

```bash
# Set seed for reproducibility
export PYTHONHASHSEED=0

# Run experiments
python scripts/run_experiments.py --experiment main --seed 42 --n_jobs 1
```

Results are saved as JSON files in `results/raw/` and can be analyzed using the analysis scripts.

## Acknowledgments

Funding acknowledgments are omitted for double-blind review.
