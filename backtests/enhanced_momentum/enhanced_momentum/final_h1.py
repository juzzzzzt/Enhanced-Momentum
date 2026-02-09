from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


SUMMARY_CSV = Path(r"C:\EnhancedMomentum\data\results\summary.csv")


def sign_p(n: int, pos: int) -> float:
    """Exact two-sided sign test p-value."""
    if n <= 0:
        return float("nan")
    tail_hi = sum(math.comb(n, k) for k in range(pos, n + 1)) / (2**n)
    tail_lo = sum(math.comb(n, k) for k in range(0, pos + 1)) / (2**n)
    return 2 * min(tail_hi, tail_lo)


def main() -> None:
    df = pd.read_csv(SUMMARY_CSV)
    df = df[df["sharpe"].notna()].copy()

    # Какие столбцы образуют "блок" (одинаковые параметры, кроме quantile)
    keys_all = [
        "period",
        "split",
        "mode",
        "as_zscore",
        "window_days",
        "exclude_last_days",
        "rebal_freq",
        "weighting_scheme",
        "hedge_freq",
    ]
    keys = [c for c in keys_all if c in df.columns]

    df["block"] = df[keys].astype(str).agg("|".join, axis=1)

    print("=== H1: Is 10% always best? ===")
    print(f"rows={len(df)} blocks={df['block'].nunique()} keys={keys}")
    print()

    # A) Наивная сводка (для интуиции)
    print("=== Naive Sharpe by quantile ===")
    naive = (
        df.groupby("quantile")["sharpe"]
        .agg(count="count", mean="mean", median="median", max="max")
        .sort_values("mean", ascending=False)
    )
    print(naive.to_string())
    print()

    # B) Fixed-effects: вычитаем среднее внутри блока
    df["sharpe_fe"] = df["sharpe"] - df.groupby("block")["sharpe"].transform("mean")
    print("=== Fixed-effect (within-block demeaned) Sharpe by quantile ===")
    fe = (
        df.groupby("quantile")["sharpe_fe"]
        .agg(count="count", mean="mean", median="median", max="max")
        .sort_values("mean", ascending=False)
    )
    print(fe.to_string())
    print()

    # C) Парные сравнения vs base=0.10
    base = 0.10
    piv = df.pivot_table(index="block", columns="quantile", values="sharpe", aggfunc="first")

    if base not in piv.columns:
        print("Base quantile 0.10 not present -> cannot do paired comparisons.")
        return

    def paired(cand: float) -> None:
        if cand not in piv.columns:
            return
        diffs = (piv[cand] - piv[base]).dropna()
        n = len(diffs)
        if n == 0:
            return
        pos = int((diffs > 0).sum())
        pval = sign_p(n, pos)
        # 95% bootstrap CI for mean diff (quick, deterministic seed)
        rng = np.random.default_rng(42)
        boots = []
        x = diffs.to_numpy()
        for _ in range(5000):
            boots.append(rng.choice(x, size=n, replace=True).mean())
        lo, hi = np.quantile(boots, [0.025, 0.975])

        print(
            f"cand={cand:.2f}  n_pairs={n:3d}  "
            f"mean_diff={diffs.mean():+.4f}  median_diff={diffs.median():+.4f}  "
            f"pos_share={pos/n:.3f}  sign_p={pval:.4g}  "
            f"boot_CI95=[{lo:+.4f}, {hi:+.4f}]"
        )

    print("=== Paired vs base quantile 0.10 (same other params) ===")
    for cand in [0.12, 0.20, 0.30, 0.40, 0.05]:
        paired(cand)
    print()

    # D) Однострочный вывод для отчёта
    # (кто лучший по FE mean)
    best_q = fe["mean"].idxmax()
    print("=== One-line takeaway ===")
    print(
        f"By within-block (fixed-effect) mean Sharpe, best quantile is q={best_q}. "
        f"(This directly answers whether 10% is always best.)"
    )


if __name__ == "__main__":
    main()
