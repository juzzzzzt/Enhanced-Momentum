from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    summary_path = Path(r"C:\EnhancedMomentum\data\results_ladder\summary.csv")
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_path}")

    df = pd.read_csv(summary_path)
    df = df[df["sharpe"].notna()].copy()

    cfg_cols = [
        "quantile",
        "exclude_last_days",
        "as_zscore",
        "window_days",
        "rebal_freq",
        "hedge_freq",
        "weighting_scheme",
        "mode",
        # keep these if present (even if not used by ctor)
        "return_type",
        "volatility_scaling",
        "vol_window_days",
    ]
    cfg_cols = [c for c in cfg_cols if c in df.columns]

    df["config_id"] = df[cfg_cols].astype(str).agg("|".join, axis=1)

    # rank inside each fold
    df["rank_in_split"] = df.groupby("split")["sharpe"].rank(ascending=False, method="min")

    rank_table = df.pivot_table(index="config_id", columns="split", values="rank_in_split", aggfunc="min")

    stats = pd.DataFrame(index=rank_table.index)
    stats["median_rank"] = rank_table.median(axis=1, skipna=True)
    stats["mean_rank"] = rank_table.mean(axis=1, skipna=True)
    stats["worst_rank"] = rank_table.max(axis=1, skipna=True)
    stats["folds_covered"] = rank_table.notna().sum(axis=1)
    stats["top3_hits"] = (rank_table <= 3).sum(axis=1)
    stats["top5_hits"] = (rank_table <= 5).sum(axis=1)
    stats["top10_hits"] = (rank_table <= 10).sum(axis=1)

    leaderboard = stats.sort_values(["median_rank", "mean_rank", "worst_rank"], ascending=[True, True, True])

    out_dir = Path(r"C:\EnhancedMomentum\data\results_ladder")
    out_dir.mkdir(parents=True, exist_ok=True)

    rank_table.to_csv(out_dir / "ranks_table.csv")
    leaderboard.to_csv(out_dir / "leaderboard.csv")

    print("Saved:")
    print("-", out_dir / "ranks_table.csv")
    print("-", out_dir / "leaderboard.csv")
    print("\nTop 15 configs by median_rank:")
    print(leaderboard.head(15).to_string())


if __name__ == "__main__":
    main()