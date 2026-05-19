from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

import argparse

def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Cannot locate repo root (no .git found).")


def _load_params(config_path: Path) -> dict[str, Any]:
    obj = json.loads(config_path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and "params" in obj:
        return obj["params"]
    return obj


def _metrics_to_row(metrics: pd.DataFrame) -> dict[str, Any]:
    if "value" in metrics.columns:
        s = metrics["value"]
        out = {}
        for k, v in s.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                out[str(k)] = v
        return out

    if metrics.shape[0] == 1:
        out = {}
        for k, v in metrics.iloc[0].items():
            if pd.notna(v):
                try:
                    out[str(k)] = float(v)
                except Exception:
                    out[str(k)] = v
        return out

    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-subdir", default="data/results_oos")
    args = parser.parse_args()

    root = _repo_root()
    results_dir = root / args.results_subdir
    runs_dir = results_dir / "runs"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not runs_dir.exists():
        raise RuntimeError(f"Runs dir not found: {runs_dir}")

    rows: list[dict[str, Any]] = []

    for run_folder in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        config_path = run_folder / "config.json"
        metrics_path = run_folder / "metrics.parquet"
        error_path = run_folder / "error.txt"

        if error_path.exists():
            continue
        if not config_path.exists() or not metrics_path.exists():
            continue

        params = _load_params(config_path)
        metrics = pd.read_parquet(metrics_path)
        metric_row = _metrics_to_row(metrics)

        rows.append(
            {
                "run_id": run_folder.name,
                **params,
                **metric_row,
            }
        )

    summary = pd.DataFrame(rows)

    preferred_first = [
        "run_id",
        "config_name",
        "config_type",
        "eval_window",
        "test_start",
        "test_end",
        "quantile",
        "exclude_last_days",
        "window_days",
        "as_zscore",
        "return_type",
        "volatility_scaling",
        "vol_window_days",
        "rebal_freq",
        "hedge_freq",
        "final_nav",
        "sharpe",
        "ir_benchmark",
        "alpha_benchmark",
        "alpha_benchmark_pvalue",
        "geom_avg_xs_r",
        "max_dd",
        "factor_loadings_low_risk",
        "factor_loadings_momentum",
        "factor_loadings_size",
        "factor_loadings_quality",
        "factor_loadings_value",
        "factor_loadings_spx_rf",
    ]

    cols = [c for c in preferred_first if c in summary.columns] + [
        c for c in summary.columns if c not in preferred_first
    ]
    summary = summary[cols]

    out_csv = results_dir / "summary.csv"
    out_parquet = results_dir / "summary.parquet"

    summary.to_csv(out_csv, index=False)
    summary.to_parquet(out_parquet)

    print(f"Saved:\n- {out_csv}\n- {out_parquet}\n")

    if "ir_benchmark" in summary.columns:
        print("Top by full-OOS IR:")
        full = summary[summary["eval_window"] == "full_oos_2020_2023"].copy()
        print(
            full.sort_values("ir_benchmark", ascending=False)
            .head(10)
            .to_string(index=False)
        )
    else:
        print("[warn] No ir_benchmark column found.")
        print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()