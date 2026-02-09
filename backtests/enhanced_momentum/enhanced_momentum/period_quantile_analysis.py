from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Cannot locate repo root (no .git found in parents).")


def _two_sided_sign_test_p(n: int, pos: int) -> float:
    """Exact two-sided sign test p-value under Binomial(n, 0.5)."""
    if n <= 0:
        return float("nan")
    # Two-sided: 2 * min(P(X<=pos), P(X>=pos))
    tail_lo = sum(math.comb(n, k) for k in range(0, pos + 1)) / (2**n)
    tail_hi = sum(math.comb(n, k) for k in range(pos, n + 1)) / (2**n)
    return float(2.0 * min(tail_lo, tail_hi))


def _bootstrap_ci_mean(x: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(x, size=len(x), replace=True)
        boots.append(sample.mean())
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return lo, hi


def main() -> None:
    repo = _repo_root()
    summary_path = repo / "data" / "results" / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found at {summary_path}")

    df = pd.read_csv(summary_path)

    # We need: period, quantile, sharpe + "keys" that define same config
    required = {"quantile", "sharpe", "period"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"summary.csv missing columns: {missing}")

    # Keep only period-runs
    d = df[df["period"].notna()].copy()
    if len(d) == 0:
        print("No rows with period != NaN. Nothing to analyze.")
        return

    base_q = 0.10

    # Keys that define "same other params"
    candidate_keys = [
        "as_zscore",
        "window_days",
        "exclude_last_days",
        "rebal_freq",
        "weighting_scheme",
        "hedge_freq",
        "mode",
    ]
    keys = [c for c in candidate_keys if c in d.columns and d[c].notna().any()]
    if not keys:
        # fallback: at least compare within period only (weak)
        keys = []

    print(f"Using keys for pairing: {keys if keys else '[NONE]'}")
    print(f"Base quantile: {base_q:.2f}")
    print()

    # Per-period analysis
    periods = sorted(d["period"].dropna().unique().tolist())
    rows: list[dict[str, Any]] = []

    for per in periods:
        dp = d[d["period"] == per].copy()

        # Pivot: each row is one config; columns are quantiles; values are sharpe
        piv = dp.pivot_table(
            index=keys,
            columns="quantile",
            values="sharpe",
            aggfunc="first",
        )

        if base_q not in piv.columns:
            print(f"[{per}] base quantile {base_q} not present -> skip")
            continue

        for cand_q in sorted([q for q in piv.columns if abs(float(q) - base_q) > 1e-12]):
            diffs = (piv[cand_q] - piv[base_q]).dropna().to_numpy(dtype=float)
            n = int(len(diffs))
            pos = int((diffs > 0).sum())

            mean_diff = float(np.mean(diffs)) if n else float("nan")
            med_diff = float(np.median(diffs)) if n else float("nan")
            pos_share = float(pos / n) if n else float("nan")
            pval = _two_sided_sign_test_p(n, pos) if n else float("nan")

            ci_lo, ci_hi = _bootstrap_ci_mean(diffs, n_boot=3000, alpha=0.05, seed=0) if n else (float("nan"), float("nan"))

            rows.append(
                dict(
                    period=per,
                    base_q=base_q,
                    cand_q=float(cand_q),
                    n_pairs=n,
                    mean_diff=mean_diff,
                    median_diff=med_diff,
                    pos_share=pos_share,
                    sign_pval=pval,
                    boot_ci_95=f"[{ci_lo:+.4f}, {ci_hi:+.4f}]",
                )
            )

    out = pd.DataFrame(rows)
    if out.empty:
        print("No paired comparisons could be formed (not enough matched configs).")
        return

    out = out.sort_values(["period", "cand_q"]).reset_index(drop=True)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    print("=== Paired comparisons vs base (within each period) ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
