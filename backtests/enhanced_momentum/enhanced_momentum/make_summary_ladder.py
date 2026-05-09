from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Cannot locate repo root (no .git found).")


def _load_params(config_path: Path) -> dict[str, Any]:
    obj = json.loads(config_path.read_text(encoding="utf-8"))
    return obj["params"] if isinstance(obj, dict) and "params" in obj else obj


def _metrics_to_row(metrics: pd.DataFrame) -> dict[str, Any]:
    # ожидаем формат: index=metric, column=value
    if "value" in metrics.columns:
        s = metrics["value"]
        return {str(k): float(v) for k, v in s.items()}

    # fallback: если вдруг metrics уже “одной строкой”
    if metrics.shape[0] == 1:
        return {str(k): float(v) for k, v in metrics.iloc[0].items() if pd.notna(v)}

    return {}


def main() -> None:
    root = _repo_root()
    runs_dir = root / "data" / "results_ladder" / "runs"
    out_dir = root / "data" / "results_ladder"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    if not runs_dir.exists():
        raise RuntimeError(f"Runs dir not found: {runs_dir}")

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

        rows.append({"run_id": run_folder.name, **params, **metric_row})

    summary = pd.DataFrame(rows)

    out_csv = out_dir / "summary.csv"
    out_parquet = out_dir / "summary.parquet"
    summary.to_csv(out_csv, index=False)
    summary.to_parquet(out_parquet)

    print(f"Saved:\n- {out_csv}\n- {out_parquet}\n")

    if "sharpe" in summary.columns:
        print("Top 10 by sharpe:")
        print(summary.sort_values("sharpe", ascending=False).head(10).to_string(index=False))
    else:
        print("[warn] No 'sharpe' column in summary. Check metrics.parquet content.")
        print(summary.head(5).to_string(index=False))


if __name__ == "__main__":
    main()