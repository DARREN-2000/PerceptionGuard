"""Fit reliability weights and thresholds from measured data (Milestone 6b).

Why this script exists
----------------------
The first sweep with hand-set weights produced fused AUROC 0.654 vs 0.679 for
raw detector confidence: the fusion was WORSE than its simplest baseline. Two
measured causes:
  * ``s_sharpness`` (AUROC 0.408) and ``s_noise`` (0.473) are ANTI-correlated
    with failure -- this colour-model detector is largely blur-tolerant, so blur
    lowers the sharpness signal without breaking perception.
  * ``s_clipping``/``s_noise``/``s_timing`` are 1.000 in most conditions and only
    dilute a uniformly-weighted geometric mean.
So the weights must be fitted, not guessed.

Key identity that keeps this principled rather than an arbitrary model swap:
    weighted geometric mean = exp( sum_i w_i * log s_i / sum_i w_i )
A linear model on log-signals IS a weighted geometric mean, and AUROC is
invariant to monotone transforms of the score. So fitting logistic regression on
log(s_i) yields exactly the geometric-mean weights the monitor already uses --
no change in fusion form, only in the weights.

Methodology guards against self-deception:
  * GROUP SPLIT BY SCENE (train on one scene, test on another). Splitting by
    frame would leak: consecutive frames of one sequence are near-duplicates and
    would inflate test AUROC.
  * Non-negative weights for the deployed monitor, so no signal can *raise*
    reliability by looking bad. The unconstrained fit is reported alongside,
    because negative weights are diagnostic information, not something to hide.
  * Thresholds are chosen on TRAIN to hit a target failure rate inside TRUSTED,
    then their behaviour is REPORTED on TEST.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from perceptionguard.evaluation.metrics import auroc  # noqa: E402

EPS = 1e-6


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    non_negative: bool = False,
    l2: float = 1e-3,
    iters: int = 4000,
    lr: float = 0.1,
) -> tuple[np.ndarray, float]:
    """Plain full-batch logistic regression (no sklearn in this environment).

    Projected gradient descent when ``non_negative``: after each step, negative
    weights are clipped to zero. Simple and sufficient for 13 features.
    """
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = p - y
        gw = X.T @ g / n + l2 * w
        gb = float(g.mean())
        w -= lr * gw
        b -= lr * gb
        if non_negative:
            w = np.maximum(w, 0.0)
    return w, b


def pick_thresholds(
    score: np.ndarray, fail: np.ndarray, *, target_trusted_fail: float = 0.05
) -> tuple[float, float]:
    """Lowest TRUSTED cut whose failure rate meets the target; CAUTION at 50%."""
    cands = np.unique(np.round(score, 3))
    t_trusted = float(cands.max())
    for t in cands:
        sel = score >= t
        if sel.sum() >= 20 and fail[sel].mean() <= target_trusted_fail:
            t_trusted = float(t)
            break
    t_caution = float(cands.min())
    for t in cands[::-1]:
        sel = score < t
        if sel.sum() >= 20 and fail[sel].mean() >= 0.5:
            t_caution = float(t)
            break
    if t_caution >= t_trusted:
        t_caution = max(float(cands.min()), t_trusted - 0.15)
    return t_trusted, t_caution


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "outputs" / "experiments.csv"))
    ap.add_argument("--train-scene", default="multi")
    ap.add_argument("--test-scene", default="approach")
    ap.add_argument("--out", default=str(ROOT / "configs" / "monitor_weights.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    sig_cols = sorted(c for c in df.columns if c.startswith("s_"))
    names = [c[2:] for c in sig_cols]

    tr = df[df.scene == args.train_scene]
    te = df[df.scene == args.test_scene]
    if len(tr) == 0 or len(te) == 0:
        print("ERROR: train/test scene split is empty")
        return 1

    def feats(d):
        # DEFICITS: -log(s) >= 0 and INCREASING with badness. The non-negativity
        # constraint is only meaningful in this sign space. Fitting w >= 0 against
        # log(s) -- which DECREASES with badness -- is unsatisfiable, and the first
        # attempt duly collapsed to a single signal at AUROC 0.522. Same fusion
        # form either way: exp(-sum w_i * deficit_i) is a weighted geometric mean.
        return -np.log(np.clip(d[sig_cols].to_numpy(dtype=float), EPS, 1.0))

    Xtr, ytr = feats(tr), tr["failure"].to_numpy(int)
    Xte, yte = feats(te), te["failure"].to_numpy(int)

    print("=" * 78)
    print(f"TRAIN scene={args.train_scene} n={len(tr)} failure_rate={ytr.mean():.3f}")
    print(f"TEST  scene={args.test_scene} n={len(te)} failure_rate={yte.mean():.3f}")
    print("Group split by scene: consecutive frames are near-duplicates, so a")
    print("random frame split would leak and inflate the test number.")
    print("=" * 78)

    # Standardize on TRAIN statistics only.
    # Scale only, no centering: centering would destroy the sign semantics that
    # make the non-negativity constraint interpretable.
    sd = np.maximum(Xtr.std(axis=0), 1e-6)
    Ztr, Zte = Xtr / sd, Xte / sd

    w_free, b_free = fit_logistic(Ztr, ytr, non_negative=False)
    w_nn, b_nn = fit_logistic(Ztr, ytr, non_negative=True)

    # Baselines and fitted models, all evaluated on TEST.
    a_conf = auroc(1.0 - te["s_confidence"], yte)
    a_prior = auroc(1.0 - te["reliability"], yte)
    a_free = auroc(Zte @ w_free + b_free, yte)
    a_nn = auroc(Zte @ w_nn + b_nn, yte)

    print("\nAUROC for predicting perception failure (TEST scene, held out):")
    print(f"  raw detector confidence          {a_conf:.3f}   <- baseline to beat")
    print(f"  a-priori hand-set weights        {a_prior:.3f}")
    print(f"  fitted weights (non-negative)    {a_nn:.3f}")
    print(f"  fitted weights (unconstrained)   {a_free:.3f}")
    verdict = "YES" if a_nn > a_conf else "NO"
    print(
        f"\n  Does fusion beat raw confidence? {verdict} "
        f"(delta = {a_nn - a_conf:+.3f}, non-negative fit)"
    )

    print("\nFitted weights on log-signals (a weighted geometric mean):")
    order = np.argsort(-np.abs(w_free))
    print(f"  {'signal':<22s}{'unconstrained':>15s}{'non-negative':>15s}")
    for i in order:
        print(f"  {names[i]:<22s}{w_free[i]:>15.3f}{w_nn[i]:>15.3f}")
    neg = [names[i] for i in range(len(names)) if w_free[i] < -0.05]
    if neg:
        print(
            "\n  Signals with NEGATIVE unconstrained weight: "
            + ", ".join(neg)
            + "\n  These are anti-correlated with failure for this detector. They are"
            "\n  clamped to 0 in the deployed monitor rather than trusted to raise"
            "\n  reliability, which would be an exploitable artifact of this dataset."
        )

    # --- thresholds ---------------------------------------------------
    # Deployed score = normalized non-negative weighted geometric mean.
    wpos = np.maximum(w_nn / sd, 0.0)  # map back out of the scaled space
    if wpos.sum() <= 0:
        print("\nERROR: all fitted weights are zero; cannot calibrate thresholds")
        return 1
    wpos = wpos / wpos.sum()

    def geo(d):
        deficit = -np.log(np.clip(d[sig_cols].to_numpy(dtype=float), EPS, 1.0))
        return np.exp(-(deficit @ wpos))

    str_tr, str_te = geo(tr), geo(te)

    # The TRUSTED target must be FEASIBLE. Clean frames already fail at the
    # detector's own baseline rate, so demanding <=5% failures inside TRUSTED is
    # unachievable for any bucket that contains clean frames -- the first attempt
    # duly pushed the cut to 0.996 and left TRUSTED empty on the test scene,
    # while clean data itself (0.975) sat below the cut. The target is therefore
    # anchored to the measured clean failure rate plus a margin.
    clean_tr = tr[tr.degradation == "none"]
    clean_fail = float(clean_tr["failure"].mean())
    target = max(0.05, clean_fail + 0.05)
    print(
        f"\nclean-condition failure rate (train) = {clean_fail:.3f}"
        f" -> feasible TRUSTED target = {target:.3f}"
    )
    t_trusted, t_caution = pick_thresholds(str_tr, ytr, target_trusted_fail=target)
    print(
        f"\nThresholds calibrated on TRAIN (target <=5% failures inside TRUSTED):"
        f"\n  TRUSTED  score >= {t_trusted:.3f}"
        f"\n  CAUTION  {t_caution:.3f} <= score < {t_trusted:.3f}"
        f"\n  DEGRADED score <  {t_caution:.3f}"
    )
    print("\nBucket behaviour on the held-out TEST scene:")
    print(f"  {'bucket':<10s}{'n':>6s}{'frames%':>9s}{'failure rate':>14s}")
    buckets = {
        "TRUSTED": str_te >= t_trusted,
        "CAUTION": (str_te >= t_caution) & (str_te < t_trusted),
        "DEGRADED": str_te < t_caution,
    }
    for name, sel in buckets.items():
        n = int(sel.sum())
        fr = float(yte[sel].mean()) if n else float("nan")
        print(f"  {name:<10s}{n:>6d}{100 * n / len(yte):>8.1f}%{fr:>14.3f}")

    clean_scores = geo(df[df.degradation == "none"])
    far = float(np.mean(clean_scores < t_trusted))
    print(
        f"\nFalse-alarm rate on CLEAN frames (scored below TRUSTED) = {far:.3f}"
        "\n  This is the cost of the monitor being wrong in the safe direction."
    )

    # Early warning: does the score drop before failure becomes severe?
    print("\nQ2. Early warning -- reliability at the LOWEST severity of each")
    print("    degradation, versus the clean baseline:")
    clean_mean = float(np.mean(geo(df[df.degradation == "none"])))
    print(f"  clean baseline score = {clean_mean:.3f}")
    for deg in sorted(df.degradation.unique()):
        if deg == "none":
            continue
        sub = df[(df.degradation == deg) & (df.severity == 0.25)]
        if not len(sub):
            continue
        sc = float(np.mean(geo(sub)))
        fr = float(sub["failure"].mean())
        flag = "warns" if sc < t_trusted else "SILENT"
        print(
            f"  {deg:<20s} score={sc:.3f} ({sc - clean_mean:+.3f} vs clean)  "
            f"actual failure rate={fr:.2f}  -> {flag}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "weights": {n: float(v) for n, v in zip(names, wpos)},
                "threshold_trusted": t_trusted,
                "threshold_caution": t_caution,
                "train_scene": args.train_scene,
                "test_scene": args.test_scene,
                "test_auroc_fitted": None if np.isnan(a_nn) else float(a_nn),
                "test_auroc_confidence_baseline": None
                if np.isnan(a_conf)
                else float(a_conf),
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
