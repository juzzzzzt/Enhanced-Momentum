from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Cannot locate repo root (no .git found).")


def load_params(config_path: Path) -> dict[str, Any]:
    obj = json.loads(config_path.read_text(encoding="utf-8"))
    # our config.json is {"params": ..., "meta": ...}
    if isinstance(obj, dict) and "params" in obj:
        return obj["params"]
    # fallback if ever saved differently
    return obj


def metrics_to_dict(metrics: pd.DataFrame) -> dict[str, Any]:
    # metrics: index='metric', column='value'
    s = metrics["value"]

    def get(name: str) -> float | None:
        return float(s.loc[name]) if name in s.index else None

    out: dict[str, Any] = {
        "final_nav": get("final_nav"),
        "sharpe": get("sharpe"),
        "max_dd": get("max_dd"),
        "geom_avg_total_r": get("geom_avg_total_r"),
        "geom_avg_xs_r": get("geom_avg_xs_r"),
        "std_xs_r": get("std_xs_r"),
        "alpha_buy_hold": get("alpha_buy_hold"),
        "ir_buy_hold": get("ir_buy_hold"),
        "alpha_benchmark": get("alpha_benchmark"),
        "alpha_benchmark_pvalue": get("alpha_benchmark_pvalue"),
        "tracking_error_benchmark": get("tracking_error_benchmark"),
        "ir_benchmark": get("ir_benchmark"),
        "ttest_pval": get("ttest_pval"),
    }

    # factor loadings (they are already flattened in your parquet as factor_loadings_*)
    for k in [
        "factor_loadings_low_risk",
        "factor_loadings_momentum",
        "factor_loadings_size",
        "factor_loadings_quality",
        "factor_loadings_value",
        "factor_loadings_spx-rf",
    ]:
        if k in s.index:
            # column name in summary without weird chars
            safe = k.replace("-", "_")
            out[safe] = float(s.loc[k])

    return out


def main() -> None:
    root = repo_root()
    runs_dir = root / "data" / "results" / "runs"

    if not runs_dir.exists():
        raise RuntimeError(f"Runs directory not found: {runs_dir}")

    rows: list[dict[str, Any]] = []

    for run_folder in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        config_path = run_folder / "config.json"
        metrics_path = run_folder / "metrics.parquet"

        if not config_path.exists() or not metrics_path.exists():
            # skip incomplete folders
            continue

        params = load_params(config_path)
        metrics = pd.read_parquet(metrics_path)

        row: dict[str, Any] = {"run_id": run_folder.name, **params, **metrics_to_dict(metrics)}
        rows.append(row)

    summary = pd.DataFrame(rows)

    # nice ordering (optional)
    preferred_first = [
        "run_id",
        "strategy",
        "mode",
        "rebal_freq",
        "quantile",
        "n_holdings",
        "window_days",
        "exclude_last_days",
        "as_zscore",
        "weighting_scheme",
        "start_date",
        "end_date",
        "final_nav",
        "sharpe",
        "max_dd",
        "geom_avg_total_r",
        "geom_avg_xs_r",
        "alpha_benchmark",
        "alpha_benchmark_pvalue",
        "tracking_error_benchmark",
        "ir_benchmark",
    ]
    cols = [c for c in preferred_first if c in summary.columns] + [c for c in summary.columns if c not in preferred_first]
    summary = summary[cols]

    # sort: higher sharpe better, higher max_dd better (less negative)
    if "sharpe" in summary.columns:
        summary = summary.sort_values(["sharpe", "max_dd"], ascending=[False, False])

    out_csv = root / "data" / "results" / "summary.csv"
    out_parquet = root / "data" / "results" / "summary.parquet"

    summary.to_csv(out_csv, index=False)
    summary.to_parquet(out_parquet)

    print(f"Saved:\n- {out_csv}\n- {out_parquet}\n")
    print("Top 10 runs:")
    print(summary.head(10))


if __name__ == "__main__":
    main()
