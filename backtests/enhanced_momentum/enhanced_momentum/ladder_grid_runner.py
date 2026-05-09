from __future__ import annotations

import argparse
import gc
import inspect
import json
import platform
import subprocess
import traceback
from datetime import datetime
from itertools import product
from multiprocessing import get_context
from pathlib import Path
from typing import Any


FOLDS = [
    {"split": "fold1_test2015", "test_start": "2015-01-01", "test_end": "2015-12-31"},
    {"split": "fold2_test2016", "test_start": "2016-01-01", "test_end": "2016-12-31"},
    {"split": "fold3_test2017", "test_start": "2017-01-01", "test_end": "2017-12-31"},
    {"split": "fold4_test2018", "test_start": "2018-01-01", "test_end": "2018-12-31"},
    {"split": "fold5_test2019", "test_start": "2019-01-01", "test_end": "2019-12-31"},
]


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Cannot locate repo root (no .git found in parents).")


def _run_id(params: dict[str, Any]) -> str:
    import hashlib

    payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:12]


def _git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return None


def _filter_kwargs_for_ctor(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs supported by cls.__init__."""
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed}


def _worker_run_one(params: dict[str, Any], results_subdir: str) -> None:
    import pandas as pd

    from enhanced_momentum.config.project_experiment_config import ProjectExperimentConfig
    from enhanced_momentum.run import run_backtest
    from enhanced_momentum.strategies.systematic_momentum import SystematicMomentum

    repo_root = _repo_root()
    run_id = _run_id(params)

    out_dir = repo_root / results_subdir / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "config.json"
    metrics_path = out_dir / "metrics.parquet"
    error_path = out_dir / "error.txt"

    meta = {
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "git_commit": _git_commit(repo_root),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    config_path.write_text(
        json.dumps({"params": params, "meta": meta}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    if metrics_path.exists():
        return

    try:
        # Build kwargs; pass only those supported by your SystematicMomentum ctor
        strat_kwargs = dict(
            mode=params["mode"],
            quantile=params["quantile"],
            window_days=params["window_days"],
            exclude_last_days=params["exclude_last_days"],
            as_zscore=params["as_zscore"],
            weighting_scheme=params["weighting_scheme"],
            # Optional knobs (stored in config anyway); will be ignored if ctor doesn't support them
            return_type=params.get("return_type"),
            volatility_scaling=params.get("volatility_scaling"),
            vol_window_days=params.get("vol_window_days"),
        )
        strat_kwargs = _filter_kwargs_for_ctor(SystematicMomentum, strat_kwargs)
        strat = SystematicMomentum(**strat_kwargs)

        exp_cfg = ProjectExperimentConfig()
        if "hedge_freq" in params and params["hedge_freq"] is not None:
            exp_cfg.HEDGE_FREQ = params["hedge_freq"]

        # Call run_backtest safely (plot/make_plots signature may differ)
        rb_sig = inspect.signature(run_backtest)
        rb_kwargs = dict(
            strategy=strat,
            rebal_freq=params["rebal_freq"],
            experiment_cfg=exp_cfg,
            start_date=pd.Timestamp(params["test_start"]),
            end_date=pd.Timestamp(params["test_end"]),
            plot=False,
            make_plots=False,
        )
        rb_kwargs = {k: v for k, v in rb_kwargs.items() if k in rb_sig.parameters}
        metrics_df = run_backtest(**rb_kwargs)

        metrics_df.to_parquet(metrics_path)
        if error_path.exists():
            error_path.unlink(missing_ok=True)

    except MemoryError:
        error_path.write_text("MemoryError\n", encoding="utf-8")
    except Exception:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        gc.collect()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-subdir", default="data/results_ladder")
    ap.add_argument("--folds", default="1,2,3,4,5")

    ap.add_argument("--mode", default="long_short")
    ap.add_argument("--rebal-freq", default="ME")
    ap.add_argument("--hedge-freq", default="ME")
    ap.add_argument("--weighting-scheme", default="equally_weighted")

    # Default: your previous small grid (fast sanity). For 128-per-fold, override via CLI.
    ap.add_argument("--quantiles", default="0.10,0.12,0.20,0.30")
    ap.add_argument("--zscore", default="false,true")
    ap.add_argument("--exclude-last-days", default="21,63")
    ap.add_argument("--window-days-list", default="252")

    # Optional knobs (may be unsupported by ctor; kept for provenance anyway)
    ap.add_argument("--return-type", default="simple")
    ap.add_argument("--vol-scaling", default="true")
    ap.add_argument("--vol-window-days", type=int, default=21)

    args = ap.parse_args()

    folds_to_run = {int(x.strip()) for x in args.folds.split(",") if x.strip()}
    chosen_folds = [f for i, f in enumerate(FOLDS, start=1) if i in folds_to_run]

    q_list = [float(x.strip()) for x in args.quantiles.split(",") if x.strip()]
    z_list = [x.strip().lower() in ("1", "true", "yes", "y") for x in args.zscore.split(",") if x.strip()]
    ex_list = [int(x.strip()) for x in args.exclude_last_days.split(",") if x.strip()]
    win_list = [int(x.strip()) for x in args.window_days_list.split(",") if x.strip()]

    vol_scaling = args.vol_scaling.strip().lower() in ("1", "true", "yes", "y")

    repo_root = _repo_root()
    runs_dir = repo_root / args.results_subdir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    for fold in chosen_folds:
        for q, z, ex, win in product(q_list, z_list, ex_list, win_list):
            jobs.append(
                {
                    "strategy": "SystematicMomentum",
                    "split": fold["split"],
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                    "mode": args.mode,
                    "rebal_freq": args.rebal_freq,
                    "hedge_freq": args.hedge_freq,
                    "quantile": q,
                    "as_zscore": z,
                    "exclude_last_days": ex,
                    "window_days": win,
                    "weighting_scheme": args.weighting_scheme,
                    "return_type": args.return_type,
                    "volatility_scaling": vol_scaling,
                    "vol_window_days": args.vol_window_days,
                }
            )

    total = len(jobs)
    print(f"Total ladder jobs: {total} (folds={sorted(folds_to_run)})")
    print(f"Results: {repo_root / args.results_subdir}")

    ctx = get_context("spawn")

    for i, params in enumerate(jobs, start=1):
        run_id = _run_id(params)
        out_dir = runs_dir / run_id
        metrics_path = out_dir / "metrics.parquet"
        error_path = out_dir / "error.txt"

        if metrics_path.exists():
            print(f"[{i:03d}/{total}] [cache] {run_id} {params['split']} q={params['quantile']} z={params['as_zscore']} ex={params['exclude_last_days']} win={params['window_days']}")
            continue

        print(f"[{i:03d}/{total}] [run]   {run_id} {params['split']} q={params['quantile']} z={params['as_zscore']} ex={params['exclude_last_days']} win={params['window_days']}")
        p = ctx.Process(target=_worker_run_one, args=(params, args.results_subdir))
        p.start()
        p.join()

        if p.exitcode != 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            error_path.write_text(f"Child exit code: {p.exitcode}\n", encoding="utf-8")
            print(f"  -> [fail] exitcode={p.exitcode}")

    print("Ladder grid finished.")


if __name__ == "__main__":
    main()