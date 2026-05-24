from __future__ import annotations

import argparse
import gc
import inspect
import json
import platform
import subprocess
import traceback
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any


CONFIGS: list[dict[str, Any]] = [
    {
        "config_name": "finalist_1_q020_ex84_win126",
        "config_type": "finalist",
        "quantile": 0.20,
        "exclude_last_days": 84,
        "window_days": 126,
    },
    {
        "config_name": "finalist_2_q030_ex84_win126",
        "config_type": "finalist",
        "quantile": 0.30,
        "exclude_last_days": 84,
        "window_days": 126,
    },
    {
        "config_name": "finalist_3_q030_ex63_win126",
        "config_type": "finalist",
        "quantile": 0.30,
        "exclude_last_days": 63,
        "window_days": 126,
    },
    {
        "config_name": "finalist_4_q020_ex84_win504",
        "config_type": "finalist",
        "quantile": 0.20,
        "exclude_last_days": 84,
        "window_days": 504,
    },
    {
        "config_name": "finalist_5_q012_ex84_win504",
        "config_type": "finalist",
        "quantile": 0.12,
        "exclude_last_days": 84,
        "window_days": 504,
    },
    {
        "config_name": "baseline_1_academic_jt_q010_ex21_win252",
        "config_type": "baseline_academic",
        "quantile": 0.10,
        "exclude_last_days": 21,
        "window_days": 252,
    },
    {
        "config_name": "baseline_2_median_grid_q020_ex63_win756",
        "config_type": "baseline_median_grid",
        "quantile": 0.20,
        "exclude_last_days": 63,
        "window_days": 756,
    },
]


EVAL_WINDOWS: list[dict[str, str]] = [
    {
        "eval_window": "2020",
        "test_start": "2020-01-01",
        "test_end": "2020-12-31",
    },
    {
        "eval_window": "2021",
        "test_start": "2021-01-01",
        "test_end": "2021-12-31",
    },
    {
        "eval_window": "2022",
        "test_start": "2022-01-01",
        "test_end": "2022-12-31",
    },
    {
        "eval_window": "2023",
        "test_start": "2023-01-01",
        "test_end": "2023-12-31",
    },
    {
        "eval_window": "full_oos_2020_2023",
        "test_start": "2020-01-01",
        "test_end": "2023-12-31",
    },
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
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed}


def _save_aligned_runner_series(
    *,
    runner_obj: Any,
    source_column: str,
    output_name: str,
    output_path: Path,
    strategy_index: Any,
) -> None:
    import pandas as pd

    if not hasattr(runner_obj, "data"):
        raise RuntimeError("runner_obj has no data attribute")

    if source_column not in runner_obj.data.columns:
        raise RuntimeError(f"Cannot find column runner_obj.data[{source_column!r}]")

    full = runner_obj.data[source_column].copy()
    full.index = pd.to_datetime(full.index)

    strategy_index = pd.to_datetime(strategy_index)

    filtered = full.loc[strategy_index[0]:strategy_index[-1]]
    filtered = pd.to_numeric(filtered, errors="coerce").dropna()

    if filtered.empty:
        raise RuntimeError(
            f"Empty {output_name} after filtering from "
            f"{strategy_index[0]} to {strategy_index[-1]}"
        )

    missing_dates = strategy_index.difference(filtered.index)

    if len(missing_dates) > 0:
        raise RuntimeError(
            f"{output_name} is not aligned with strategy_total_r. "
            f"Missing dates: {len(missing_dates)}. "
            f"First missing: {missing_dates[:5].tolist()}"
        )

    filtered = filtered.loc[strategy_index]
    filtered.name = output_name
    filtered.to_frame().to_parquet(output_path)


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
    total_path = out_dir / "strategy_total_r.parquet"
    excess_path = out_dir / "strategy_excess_r.parquet"
    market_path = out_dir / "market_total_r.parquet"
    momentum_path = out_dir / "momentum_factor_r.parquet"
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

    if (
        metrics_path.exists()
        and total_path.exists()
        and excess_path.exists()
        and market_path.exists()
        and momentum_path.exists()
    ):
        return

    try:
        strat_kwargs = dict(
            mode=params["mode"],
            quantile=params["quantile"],
            window_days=params["window_days"],
            exclude_last_days=params["exclude_last_days"],
            as_zscore=params["as_zscore"],
            weighting_scheme=params["weighting_scheme"],
            return_type=params["return_type"],
            volatility_scaling=params["volatility_scaling"],
            vol_window_days=params["vol_window_days"],
        )
        strat_kwargs = _filter_kwargs_for_ctor(SystematicMomentum, strat_kwargs)
        strategy = SystematicMomentum(**strat_kwargs)

        exp_cfg = ProjectExperimentConfig()
        exp_cfg.HEDGE_FREQ = params["hedge_freq"]

        rb_sig = inspect.signature(run_backtest)
        rb_kwargs = dict(
            strategy=strategy,
            rebal_freq=params["rebal_freq"],
            experiment_cfg=exp_cfg,
            start_date=pd.Timestamp(params["test_start"]),
            end_date=pd.Timestamp(params["test_end"]),
            plot=False,
            make_plots=False,
            return_runner=True,
        )
        rb_kwargs = {k: v for k, v in rb_kwargs.items() if k in rb_sig.parameters}

        result = run_backtest(**rb_kwargs)

        if isinstance(result, tuple):
            metrics_df, runner_obj = result
        else:
            metrics_df = result
            runner_obj = None

        metrics_df.to_parquet(metrics_path)

        if runner_obj is None:
            raise RuntimeError("run_backtest did not return runner_obj. Check return_runner=True support.")

        strategy_total_r = getattr(runner_obj, "strategy_total_r", None)
        strategy_excess_r = getattr(runner_obj, "strategy_excess_r", None)

        if strategy_total_r is None:
            raise RuntimeError("runner_obj.strategy_total_r is None")

        if strategy_excess_r is None:
            raise RuntimeError("runner_obj.strategy_excess_r is None")

        strategy_total_r.index = pd.to_datetime(strategy_total_r.index)
        strategy_excess_r.index = pd.to_datetime(strategy_excess_r.index)

        strategy_total_r.to_parquet(total_path)
        strategy_excess_r.to_parquet(excess_path)

        strategy_index = pd.to_datetime(strategy_total_r.index)

        _save_aligned_runner_series(
            runner_obj=runner_obj,
            source_column="spx",
            output_name="market_total_r",
            output_path=market_path,
            strategy_index=strategy_index,
        )

        _save_aligned_runner_series(
            runner_obj=runner_obj,
            source_column="momentum",
            output_name="momentum_factor_r",
            output_path=momentum_path,
            strategy_index=strategy_index,
        )

        if error_path.exists():
            error_path.unlink(missing_ok=True)

    except MemoryError:
        error_path.write_text("MemoryError\n", encoding="utf-8")
    except Exception:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--results-subdir", default="data/results_oos")
    parser.add_argument("--windows", default="2020,2021,2022,2023,full_oos_2020_2023")

    parser.add_argument("--mode", default="long_short")
    parser.add_argument("--rebal-freq", default="ME")
    parser.add_argument("--hedge-freq", default="ME")
    parser.add_argument("--weighting-scheme", default="equally_weighted")

    parser.add_argument("--as-zscore", default="false")
    parser.add_argument("--return-type", default="simple")
    parser.add_argument("--vol-scaling", default="true")
    parser.add_argument("--vol-window-days", type=int, default=21)

    args = parser.parse_args()

    selected_windows = {x.strip() for x in args.windows.split(",") if x.strip()}
    eval_windows = [w for w in EVAL_WINDOWS if w["eval_window"] in selected_windows]

    as_zscore = args.as_zscore.strip().lower() in ("1", "true", "yes", "y")
    volatility_scaling = args.vol_scaling.strip().lower() in ("1", "true", "yes", "y")

    jobs: list[dict[str, Any]] = []

    for window in eval_windows:
        for cfg in CONFIGS:
            jobs.append(
                {
                    "strategy": "SystematicMomentum",
                    "experiment_group": "oos_finalists_plus_baselines",
                    "config_name": cfg["config_name"],
                    "config_type": cfg["config_type"],
                    "eval_window": window["eval_window"],
                    "test_start": window["test_start"],
                    "test_end": window["test_end"],
                    "mode": args.mode,
                    "rebal_freq": args.rebal_freq,
                    "hedge_freq": args.hedge_freq,
                    "quantile": cfg["quantile"],
                    "exclude_last_days": cfg["exclude_last_days"],
                    "window_days": cfg["window_days"],
                    "as_zscore": as_zscore,
                    "weighting_scheme": args.weighting_scheme,
                    "return_type": args.return_type,
                    "volatility_scaling": volatility_scaling,
                    "vol_window_days": args.vol_window_days,
                }
            )

    repo_root = _repo_root()
    runs_dir = repo_root / args.results_subdir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    total = len(jobs)
    print(f"Total OOS jobs: {total}")
    print(f"Results: {repo_root / args.results_subdir}")

    ctx = get_context("spawn")

    for i, params in enumerate(jobs, start=1):
        run_id = _run_id(params)
        out_dir = runs_dir / run_id
        metrics_path = out_dir / "metrics.parquet"
        total_path = out_dir / "strategy_total_r.parquet"
        excess_path = out_dir / "strategy_excess_r.parquet"
        market_path = out_dir / "market_total_r.parquet"
        momentum_path = out_dir / "momentum_factor_r.parquet"
        error_path = out_dir / "error.txt"

        msg = (
            f"[{i:03d}/{total}] "
            f"{params['eval_window']} | {params['config_name']} | "
            f"q={params['quantile']} ex={params['exclude_last_days']} win={params['window_days']}"
        )

        if (
            metrics_path.exists()
            and total_path.exists()
            and excess_path.exists()
            and market_path.exists()
            and momentum_path.exists()
        ):
            print(f"{msg} [cache]")
            continue

        print(f"{msg} [run]")
        proc = ctx.Process(target=_worker_run_one, args=(params, args.results_subdir))
        proc.start()
        proc.join()

        if proc.exitcode != 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            error_path.write_text(f"Child exit code: {proc.exitcode}\n", encoding="utf-8")
            print(f"  -> [fail] exitcode={proc.exitcode}")

    print("OOS runner finished.")


if __name__ == "__main__":
    main()