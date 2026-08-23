"""Generate the evaluation figures from outputs/experiments.csv.

Every figure is derived from the measured sweep. Nothing here synthesizes,
smooths or extrapolates data.

Usage:
    python scripts/make_plots.py [--csv outputs/experiments.csv] [--outdir outputs/figures]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display in the sandbox or in CI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perceptionguard.evaluation.metrics import auroc  # noqa: E402

SIGNAL_PREFIX = "s_"
TEST_SCENE = "approach"  # held out during weight fitting


def _signal_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(SIGNAL_PREFIX)]


def plot_reliability_vs_severity(df: pd.DataFrame, outdir: Path) -> Path:
    """Does reliability fall as the degradation gets worse? (Question 1)"""
    degs = [d for d in sorted(df.degradation.unique()) if d != "none"]
    ncol = 3
    nrow = int(np.ceil(len(degs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    clean = float(df[df.degradation == "none"].reliability.mean())

    for ax, deg in zip(axes, degs):
        sub = df[df.degradation == deg].groupby("severity")
        sev = sorted(df[df.degradation == deg].severity.unique())
        rel = sub.reliability.mean().reindex(sev)
        fail = sub.failure.mean().reindex(sev)

        ax.plot(sev, rel, "o-", color="#1f77b4", label="reliability")
        ax.plot(
            sev,
            1.0 - fail,
            "s--",
            color="#d62728",
            alpha=0.75,
            label="1 - failure rate",
        )
        ax.axhline(clean, color="gray", lw=0.8, ls=":", label="clean baseline")
        ax.set_title(deg, fontsize=10)
        ax.set_xlabel("severity")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)

    for ax in axes[len(degs) :]:
        ax.axis("off")
    axes[0].set_ylabel("score")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "Q1: reliability vs degradation severity (red = actual perception quality)",
        fontsize=12,
    )
    fig.tight_layout()
    path = outdir / "reliability_vs_severity.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_signal_auroc(df: pd.DataFrame, outdir: Path) -> Path:
    """Which individual signals carry information about failure? (Question 3)"""
    cols = _signal_columns(df)
    labels = df.failure.astype(int).to_numpy()
    scores = {c: auroc((-df[c]).to_numpy(), labels) for c in cols}
    scores = {k: v for k, v in scores.items() if np.isfinite(v)}
    order = sorted(scores, key=lambda k: scores[k])

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    vals = [scores[k] for k in order]
    # Below 0.5 means the signal is anti-correlated with failure -- a real
    # finding for this backend, not a bug, so it is drawn in a different colour
    # rather than hidden.
    colors = ["#d62728" if v < 0.5 else "#2ca02c" for v in vals]
    ax.barh([k.removeprefix(SIGNAL_PREFIX) for k in order], vals, color=colors)
    ax.axvline(0.5, color="black", lw=1.0, ls="--")
    ax.set_xlabel("AUROC for predicting a perception failure")
    ax.set_xlim(0.3, 1.0)
    ax.set_title("Q3: per-signal discriminative power (red = anti-correlated)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path = outdir / "signal_auroc.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_roc(df: pd.DataFrame, outdir: Path) -> Path:
    """Fused reliability vs raw detector confidence. (Question 4)"""

    def roc(score: np.ndarray, label: np.ndarray):
        order = np.argsort(-score)
        lab = label[order]
        tp = np.cumsum(lab)
        fp = np.cumsum(1 - lab)
        npos, nneg = max(lab.sum(), 1), max((1 - lab).sum(), 1)
        return np.r_[0, fp / nneg], np.r_[0, tp / npos]

    test = df[df.scene == TEST_SCENE]
    lab = test.failure.astype(int).to_numpy()

    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    for col, name, color in (
        ("reliability", "fused reliability", "#1f77b4"),
        ("s_confidence", "raw detector confidence", "#ff7f0e"),
    ):
        s = (-test[col]).to_numpy()  # low score = likely failure
        x, y = roc(s, lab)
        ax.plot(x, y, label=f"{name} (AUROC {auroc(s, lab):.3f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"Q4: failure detection on the held-out '{TEST_SCENE}' scene")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = outdir / "roc_fused_vs_confidence.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_reliability_distribution(df: pd.DataFrame, outdir: Path) -> Path:
    """Separation between frames that failed and frames that did not."""
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    bins = np.linspace(0, 1, 41)
    ax.hist(
        df[~df.failure].reliability,
        bins=bins,
        alpha=0.65,
        label="perception OK",
        color="#2ca02c",
    )
    ax.hist(
        df[df.failure].reliability,
        bins=bins,
        alpha=0.65,
        label="perception FAILED",
        color="#d62728",
    )
    ax.set_xlabel("reliability score")
    ax.set_ylabel("frames")
    ax.set_title("Score distribution by measured outcome (overlap = irreducible error)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = outdir / "reliability_distribution.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_latency(df: pd.DataFrame, outdir: Path) -> Path:
    """Runtime cost: is the safety mechanism affordable?"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax1.hist(df.latency_ms, bins=50, color="#1f77b4", alpha=0.85)
    p95 = float(np.percentile(df.latency_ms, 95))
    ax1.axvline(
        float(df.latency_ms.mean()),
        color="black",
        ls="--",
        label=f"mean {df.latency_ms.mean():.1f} ms",
    )
    ax1.axvline(p95, color="#d62728", ls="--", label=f"p95 {p95:.1f} ms")
    ax1.set_xlabel("end-to-end frame latency (ms)")
    ax1.set_ylabel("frames")
    ax1.set_title("Total pipeline latency (CPU, 2 vCPU)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)

    monitor = float(df.monitor_ms.mean())
    rest = float(df.latency_ms.mean()) - monitor
    ax2.bar(
        ["perception", "reliability monitor"],
        [rest, monitor],
        color=["#1f77b4", "#ff7f0e"],
    )
    ax2.set_ylabel("mean ms/frame")
    ax2.set_title(
        f"Cost of the monitor: {monitor / (monitor + rest) * 100:.1f}% of the budget"
    )
    ax2.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    path = outdir / "latency.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", type=Path, default=ROOT / "outputs" / "experiments.csv"
    )
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs" / "figures")
    args = parser.parse_args(argv)

    if not args.csv.exists():
        print(
            f"missing {args.csv} -- run scripts/run_experiments.py first",
            file=sys.stderr,
        )
        return 1

    df = pd.read_csv(args.csv)
    if df.failure.dtype != bool:
        df["failure"] = df.failure.astype(bool)
    args.outdir.mkdir(parents=True, exist_ok=True)

    made = [
        plot_reliability_vs_severity(df, args.outdir),
        plot_signal_auroc(df, args.outdir),
        plot_roc(df, args.outdir),
        plot_reliability_distribution(df, args.outdir),
        plot_latency(df, args.outdir),
    ]
    for p in made:
        print(f"wrote {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
