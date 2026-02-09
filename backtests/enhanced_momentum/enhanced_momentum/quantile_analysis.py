from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


def repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Cannot locate repo root (.git not found)")


def binom_two_sided_pvalue(pos: int, n: int) -> float:
    """Exact 2-sided binomial p-value for sign test under p=0.5."""
    if n <= 0:
        return float("nan")

    k = min(pos, n - pos)

    # P(X <= k) where X~Bin(n, 0.5)
    # exact sum_{i=0..k} C(n,i) / 2^n
    denom = 2 ** n
    cum = 0
    for i in range(0, k + 1):
        cum += math.comb(n, i)
    p = 2 * (cum / denom)
    return min(1.0, p)


def paired_comparison(df: pd.DataFrame, base_q: float, cand_q: float, key_cols: list[str]) -> dict:
    # агрегируем (если вдруг дубли) → берем максимум Sharpe
    g = (
        df.groupby(key_cols + ["quantile"], dropna=False)["sharpe"]
        .max()
        .reset_index()
    )
    pivot = g.pivot_table(index=key_cols, columns="quantile", values="sharpe", aggfunc="max")

    if base_q not in pivot.columns or cand_q not in pivot.columns:
        return {
            "base_q": base_q,
            "cand_q": cand_q,
            "n_pairs": 0,
            "mean_diff": float("nan"),
            "median_diff": float("nan"),
            "pos_share": float("nan"),
            "sign_pval": float("nan"),
            "boot_ci_95": (float("nan"), float("nan")),
        }

    d = pivot[cand_q] - pivot[base_q]
    d = d.dropna()

    # убираем точные нули для sign-test
    d_nz = d[d != 0]

    n_pairs = int(d.shape[0])
    pos = int((d_nz > 0).sum())
    neg = int((d_nz < 0).sum())
    n_sign = pos + neg

    # bootstrap CI для среднего диффа
    rng = np.random.default_rng(123)
    B = 5000
    if n_pairs > 0:
        boot = []
        vals = d.to_numpy()
        for _ in range(B):
            sample = rng.choice(vals, size=n_pairs, replace=True)
            boot.append(sample.mean())
        lo, hi = np.quantile(boot, [0.025, 0.975])
    else:
        lo, hi = float("nan"), float("nan")

    return {
        "base_q": base_q,
        "cand_q": cand_q,
        "n_pairs": n_pairs,
        "mean_diff": float(d.mean()) if n_pairs else float("nan"),
        "median_diff": float(d.median()) if n_pairs else float("nan"),
        "pos_share": float(pos / n_sign) if n_sign else float("nan"),
        "sign_pval": float(binom_two_sided_pvalue(pos, n_sign)) if n_sign else float("nan"),
        "boot_ci_95": (float(lo), float(hi)),
    }


def main() -> None:
    root = repo_root()
    summary_path = root / "data" / "results" / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_path}")

    df = pd.read_csv(summary_path)

    # фильтры по желанию (оставил минимально)
    if "strategy" in df.columns:
        df = df[df["strategy"] == "SystematicMomentum"].copy()
    if "mode" in df.columns:
        df = df[df["mode"] == "long_short"].copy()

    # нормализуем quantile (чтобы не ловить 0.3000000004)
    df["quantile"] = pd.to_numeric(df["quantile"], errors="coerce").round(2)
    df["sharpe"] = pd.to_numeric(df["sharpe"], errors="coerce")

    # быстрый репорт: max/mean Sharpe по квантилям
    rep = (
        df.groupby("quantile")["sharpe"]
        .agg(count="count", mean="mean", median="median", max="max")
        .sort_values("max", ascending=False)
    )
    print("\n=== Sharpe by quantile (aggregate over other params) ===")
    print(rep.to_string())

    # лучшие прогоны на квантиль
    cols_show = [c for c in ["run_id", "quantile", "as_zscore", "window_days", "exclude_last_days", "hedge_freq", "rebal_freq", "weighting_scheme", "sharpe", "max_dd", "final_nav"] if c in df.columns]
    best = df.sort_values("sharpe", ascending=False).groupby("quantile").head(1)[cols_show]
    print("\n=== Best run per quantile ===")
    print(best.to_string(index=False))

    # парное сравнение против q=0.10 (контролируем остальные параметры)
    base_q = 0.10
    candidates = [0.12, 0.20, 0.30, 0.40]

    # ключи “прочих равных” — берем только те, что реально есть в summary
    key_candidates = ["as_zscore", "window_days", "exclude_last_days", "hedge_freq", "rebal_freq", "weighting_scheme", "start_date", "end_date"]
    key_cols = [c for c in key_candidates if c in df.columns]

    # чтобы pivot не страдал от NaN в строковых ключах
    for c in key_cols:
        if df[c].dtype == "O":
            df[c] = df[c].fillna("NA")

    print("\n=== Paired comparisons vs base quantile 0.10 (same other params) ===")
    rows = []
    for q in candidates:
        rows.append(paired_comparison(df, base_q=base_q, cand_q=q, key_cols=key_cols))

    out = pd.DataFrame(rows)
    # красиво распечатаем CI
    out["boot_ci_95"] = out["boot_ci_95"].apply(lambda x: f"[{x[0]:+.4f}, {x[1]:+.4f}]")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
