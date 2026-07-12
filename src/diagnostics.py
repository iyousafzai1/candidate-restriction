"""Gate 2 — low-rank diagnostics (MVE plan §5).

For each mode of the (per-task z-scored) tensor:
  - singular value spectrum of the mode unfolding,
  - effective rank at 90%/99% Frobenius energy (as fraction of max rank),
  - same for an entry-shuffled control tensor (structure destroyed,
    marginal distribution preserved).

Gate 2 (per scenario), revised after the first MVE run:
The rank-fraction criterion (r90_frac <= max_r90_frac) is statistically
meaningless on small modes — at dim 7 it admits only 2 components, which a
merely smooth nonlinear response already exceeds. Criteria are therefore
applied by mode size:
  - separation from shuffled control >= min_separation:
        gated for every mode with dim >= gate_separation_min_dim (default 4);
        this is the actual structure test and works at any dimension.
  - r90_frac <= max_r90_frac:
        additionally gated only where max_rank >= gate_frac_min_rank
        (default 10), where "low-rank" is a meaningful notion.
A mode passes if all criteria applicable to it pass; the scenario passes
Gate 2 if all gated modes pass. Both numbers are always reported.
"""
from __future__ import annotations

import numpy as np

from tensorize import normalize_per_task

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------- #
# Core linear algebra
# --------------------------------------------------------------------------- #

def unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def sv_spectrum(mat: np.ndarray) -> np.ndarray:
    return np.linalg.svd(mat, compute_uv=False)


def energy_rank(s: np.ndarray, energy: float) -> int:
    """Smallest r with sum(s[:r]^2) >= energy * sum(s^2)."""
    e = np.cumsum(s ** 2)
    total = e[-1]
    if total <= 0:
        return len(s)
    return int(np.searchsorted(e / total, energy) + 1)


def shuffled_copy(tensor: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    flat = tensor.ravel().copy()
    rng.shuffle(flat)
    return flat.reshape(tensor.shape)


# --------------------------------------------------------------------------- #
# Diagnosis
# --------------------------------------------------------------------------- #

def diagnose(tensor: np.ndarray, axis_names: list[str], cfg: dict) -> dict:
    """cfg: the `diagnostics` section of the YAML config."""
    energy = cfg.get("energy", 0.90)
    energy_hi = cfg.get("energy_hi", 0.99)
    max_frac = cfg.get("max_r90_frac", 0.30)
    min_sep = cfg.get("min_separation", 0.15)
    sep_min_dim = cfg.get("gate_separation_min_dim", 4)
    frac_min_rank = cfg.get("gate_frac_min_rank", 10)
    rng = np.random.default_rng(cfg.get("shuffle_seed", 7))

    z = normalize_per_task(tensor)
    z_shuf = shuffled_copy(z, rng)
    mode_names = list(axis_names) + ["tasks"]

    modes, gated_pass = [], []
    for mode, name in enumerate(mode_names):
        dim = z.shape[mode]
        s = sv_spectrum(unfold(z, mode))
        s_sh = sv_spectrum(unfold(z_shuf, mode))
        max_rank = min(unfold(z, mode).shape)

        r90 = energy_rank(s, energy)
        r99 = energy_rank(s, energy_hi)
        r90_sh = energy_rank(s_sh, energy)
        frac = r90 / max_rank
        frac_sh = r90_sh / max_rank

        criteria = []
        if dim >= sep_min_dim:
            criteria.append(("separation", (frac_sh - frac) >= min_sep))
        if max_rank >= frac_min_rank:
            criteria.append(("r90_frac", frac <= max_frac))
        gated = len(criteria) > 0
        passed = bool(all(ok for _, ok in criteria)) if gated else True
        if gated:
            gated_pass.append(passed)

        modes.append({
            "mode": mode, "name": name, "dim": int(dim), "max_rank": int(max_rank),
            "r90": int(r90), "r99": int(r99),
            "r90_frac": round(frac, 4), "r90_frac_shuffled": round(frac_sh, 4),
            "separation": round(frac_sh - frac, 4),
            "criteria": {c: bool(ok) for c, ok in criteria},
            "gated": bool(gated), "passed": passed,
            "sv_norm": (s / s[0]).tolist() if s[0] > 0 else s.tolist(),
            "sv_norm_shuffled": (s_sh / s_sh[0]).tolist() if s_sh[0] > 0 else s_sh.tolist(),
        })

    # Per-task slice effective rank (first-mode matricization of each slice).
    task_ranks = []
    for k in range(z.shape[-1]):
        sl = z[..., k]
        mat = sl if sl.ndim == 2 else sl.reshape(sl.shape[0], -1)
        task_ranks.append(energy_rank(sv_spectrum(mat), energy))

    return {
        "thresholds": {"energy": energy, "max_r90_frac": max_frac,
                       "min_separation": min_sep,
                       "gate_separation_min_dim": sep_min_dim,
                       "gate_frac_min_rank": frac_min_rank},
        "modes": modes,
        "per_task_slice_r90": task_ranks,
        "gate2_pass": bool(gated_pass and all(gated_pass)),
        "n_gated_modes": int(len(gated_pass)),
    }


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot_diagnostics(report: dict, tensor: np.ndarray,
                     axes: list, out_dir, scenario_key: str) -> list:
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # 1) singular value decay per mode, with shuffled overlay
    modes = report["modes"]
    ncols = len(modes)
    fig, axs = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.2), squeeze=False)
    for ax, m in zip(axs[0], modes):
        ax.semilogy(np.array(m["sv_norm"]) + 1e-16, "o-", ms=3, label="real")
        ax.semilogy(np.array(m["sv_norm_shuffled"]) + 1e-16, "s--", ms=3,
                    alpha=0.6, label="shuffled")
        ax.axvline(m["r90"] - 1, color="k", lw=0.8, ls=":")
        ax.set_title(f"mode '{m['name']}' (r90={m['r90']}/{m['max_rank']})",
                     fontsize=9)
        ax.set_xlabel("component")
        ax.legend(fontsize=7)
    fig.suptitle(f"{scenario_key}: singular value decay (z-scored per task)")
    fig.tight_layout()
    p = out_dir / f"diag_{scenario_key}_sv_decay.png"
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # 2) cumulative energy per mode
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for m in modes:
        s = np.array(m["sv_norm"])
        e = np.cumsum(s ** 2) / np.sum(s ** 2)
        ax.plot(np.arange(1, len(e) + 1) / m["max_rank"], e,
                label=f"{m['name']}")
    ax.axhline(report["thresholds"]["energy"], color="k", lw=0.8, ls=":")
    ax.axvline(report["thresholds"]["max_r90_frac"], color="r", lw=0.8, ls=":")
    ax.set_xlabel("rank fraction"); ax.set_ylabel("cumulative energy")
    ax.legend(fontsize=8); ax.set_title(f"{scenario_key}: cumulative energy")
    fig.tight_layout()
    p = out_dir / f"diag_{scenario_key}_cum_energy.png"
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # 3) histogram of per-task slice ranks
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(report["per_task_slice_r90"], bins=20)
    ax.set_xlabel("per-task slice r90"); ax.set_ylabel("#tasks")
    ax.set_title(f"{scenario_key}: per-task effective rank")
    fig.tight_layout()
    p = out_dir / f"diag_{scenario_key}_task_ranks.png"
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    # 4) example slices as heatmaps (2-D view; higher modes fixed at middle)
    n_show = min(6, tensor.shape[-1])
    fig, axs = plt.subplots(1, n_show, figsize=(2.6 * n_show, 2.8), squeeze=False)
    for j in range(n_show):
        sl = tensor[..., j]
        while sl.ndim > 2:                      # fix trailing modes at middle
            sl = sl[..., sl.shape[-1] // 2]
        if sl.ndim == 1:
            sl = sl[None, :]
        im = axs[0][j].imshow(sl, aspect="auto", cmap="viridis")
        axs[0][j].set_title(f"task {j}", fontsize=8)
        fig.colorbar(im, ax=axs[0][j], fraction=0.046)
    fig.suptitle(f"{scenario_key}: example performance slices (raw)")
    fig.tight_layout()
    p = out_dir / f"diag_{scenario_key}_slices.png"
    fig.savefig(p, dpi=150); plt.close(fig); paths.append(p)

    return paths
