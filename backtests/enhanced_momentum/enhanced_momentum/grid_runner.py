from __future__ import annotations

import itertools
import json
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from enhanced_momentum.config.project_experiment_config import ProjectExperimentConfig
from enhanced_momentum.run import _repo_root, _run_id, run_backtest
from enhanced_momentum.strategies.systematic_momentum import SystematicMomentum


def _dump_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main() -> None:
    # ---- Global fixed params (edit here if needed) ----
    base_params: dict[str, Any] = {
        "strategy": "SystematicMomentum",
        "mode": "long_short",
        "rebal_freq": "ME",          # monthly rebalance
        "start_date": "2022-01-01",
        "end_date": None,
        "weighting_scheme": "equally_weighted",
        # key change: hedge freq (NOT daily) to avoid huge memory/time
        "hedge_freq": "ME",
    }

    # ---- Grid (edit here) ----
    grid = {
        "quantile": [0.10, 0.12, 0.20, 0.30],
        "as_zscore": [False, True],
        "window_days": [126, 252, 504],
        "exclude_last_days": [0, 21, 63],
    }

    # Expand grid
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = list(itertools.product(*values))
    print(f"Total grid runs: {len(combos)}")

    repo_root = _repo_root()
    runs_root = repo_root / "data" / "results" / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    for i, combo in enumerate(combos, start=1):
        params = dict(base_params)
        params.update({k: v for k, v in zip(keys, combo)})

        run_id = _run_id(params)
        out_dir = runs_root / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        config_path = out_dir / "config.json"
        metrics_path = out_dir / "metrics.parquet"
        error_path = out_dir / "error.txt"

        # Always store config for reproducibility
        if not config_path.exists():
            _dump_json(config_path, params)

        # Cache hit
        if metrics_path.exists():
            print(f"[{i:02d}/{len(combos)}] [cache] {run_id} q={params['quantile']} z={params['as_zscore']} "
                  f"w={params['window_days']} skip={params['exclude_last_days']} hedge={params['hedge_freq']}")
            continue

        print(f"[{i:02d}/{len(combos)}] [run]  {run_id} q={params['quantile']} z={params['as_zscore']} "
              f"w={params['window_days']} skip={params['exclude_last_days']} hedge={params['hedge_freq']}")

        # Build experiment config (override hedge frequency here)
        exp_cfg = ProjectExperimentConfig()
        exp_cfg.HEDGE_FREQ = params["hedge_freq"]

        # Build strategy
        strat = SystematicMomentum(
            mode=params["mode"],
            quantile=params["quantile"],
            window_days=params["window_days"],
            exclude_last_days=params["exclude_last_days"],
            as_zscore=params["as_zscore"],
            weighting_scheme=params["weighting_scheme"],
        )

        try:
            metrics = run_backtest(
                strategy=strat,
                rebal_freq=params["rebal_freq"],
                start_date=pd.Timestamp(params["start_date"]),
                end_date=pd.Timestamp(params["end_date"]) if params["end_date"] else None,
                experiment_cfg=exp_cfg,
                make_plots=False,
            )
            # metrics is a DataFrame (metric x value)
            metrics.to_parquet(metrics_path)

            # If there was an old error file from previous attempts, remove it
            if error_path.exists():
                error_path.unlink()

        except Exception:
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            print(f"[fail] {run_id} -> saved error.txt")
            continue

    print("Grid finished.")


if __name__ == "__main__":
    main()
