from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import platform
import subprocess
import sys
import traceback
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any


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


def _set_cfg_field(cfg: Any, field_candidates: list[str], value: Any) -> Any:
    """
    Best-effort: set hedge_freq / similar in ProjectTradingConfig across possible field names.
    Works for dataclasses + plain objects. If nothing fits, returns cfg unchanged.
    """
    for name in field_candidates:
        if hasattr(cfg, name):
            # try dataclasses.replace (for frozen dataclasses)
            try:
                return dataclasses.replace(cfg, **{name: value})
            except Exception:
                try:
                    setattr(cfg, name, value)
                    return cfg
                except Exception:
                    pass
    return cfg


def _worker_run_one(params: dict[str, Any]) -> None:
    """
    Run exactly one backtest in a fresh process.
    Writes:
      - config.json
      - metrics.parquet   (if success)
      - error.txt         (if failure)
    """
    # heavy imports inside child process
    import pandas as pd

    from enhanced_momentum.config.project_trading_config import ProjectTradingConfig
    from enhanced_momentum.run import run_backtest  # uses your existing runner wrapper
    from enhanced_momentum.strategies.systematic_momentum import SystematicMomentum

    repo_root = _repo_root()
    run_id = _run_id(params)
    out_dir = repo_root / "data" / "results" / "runs" / run_id
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
        # already computed
        return

    try:
        sys_mom = SystematicMomentum(
            mode=params["mode"],
            quantile=params["quantile"],
            window_days=params["window_days"],
            exclude_last_days=params["exclude_last_days"],
            as_zscore=params["as_zscore"],
            weighting_scheme=params["weighting_scheme"],
        )

        trading_cfg = ProjectTradingConfig()
        # try to enforce hedge frequency if user provided it
        if "hedge_freq" in params and params["hedge_freq"] is not None:
            trading_cfg = _set_cfg_field(
                trading_cfg,
                field_candidates=["hedge_freq", "hedge_rebal_freq", "HEDGE_FREQ", "HEDGE_REBAL_FREQ"],
                value=params["hedge_freq"],
            )

        metrics_df = run_backtest(
            strategy=sys_mom,
            rebal_freq=params["rebal_freq"],
            trading_cfg=trading_cfg,
            start_date=pd.Timestamp(params["start_date"]),
            end_date=pd.Timestamp(params["end_date"]) if params["end_date"] else None,
            plot=False,  # IMPORTANT: no plots in grid
        )

        metrics_df.to_parquet(metrics_path)

    except MemoryError:
        error_path.write_text("MemoryError\n", encoding="utf-8")
    except Exception:
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2010-02-01")
    parser.add_argument("--end-date", default="2021-12-31")
    parser.add_argument("--rebal-freq", default="ME")
    parser.add_argument("--hedge-freq", default="ME")  # set "D" if you really want daily hedging
    parser.add_argument("--window-days", type=int, default=252)
    parser.add_argument("--exclude-last-days", default="21,63")
    parser.add_argument("--quantiles", default="0.05,0.10,0.12,0.20,0.30,0.40")
    parser.add_argument("--zscore", default="false,true")
    parser.add_argument("--mode", default="long_short")
    parser.add_argument("--weighting-scheme", default="equally_weighted")
    parser.add_argument("--from-idx", type=int, default=0)
    parser.add_argument("--to-idx", type=int, default=10**9)
    args = parser.parse_args()

    exclude_list = [int(x.strip()) for x in args.exclude_last_days.split(",") if x.strip()]
    q_list = [float(x.strip()) for x in args.quantiles.split(",") if x.strip()]
    z_list = [x.strip().lower() in ("1", "true", "yes", "y") for x in args.zscore.split(",") if x.strip()]

    grid: list[dict[str, Any]] = []
    for ex in exclude_list:
        for q in q_list:
            for z in z_list:
                grid.append(
                    {
                        "strategy": "SystematicMomentum",
                        "mode": args.mode,
                        "quantile": q,
                        "window_days": args.window_days,
                        "exclude_last_days": ex,
                        "as_zscore": z,
                        "weighting_scheme": args.weighting_scheme,
                        "rebal_freq": args.rebal_freq,
                        "hedge_freq": args.hedge_freq,
                        "start_date": args.start_date,
                        "end_date": args.end_date,
                    }
                )

    grid = grid[args.from_idx : min(args.to_idx, len(grid))]
    total = len(grid)
    print(f"Total grid runs in this batch: {total}")

    repo_root = _repo_root()
    runs_dir = repo_root / "data" / "results" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    ctx = get_context("spawn")

    for i, params in enumerate(grid, start=1):
        run_id = _run_id(params)
        out_dir = runs_dir / run_id
        metrics_path = out_dir / "metrics.parquet"
        error_path = out_dir / "error.txt"

        # cached?
        if metrics_path.exists():
            print(f"[{i:03d}/{total}] [cache] {run_id} q={params['quantile']} z={params['as_zscore']} ex={params['exclude_last_days']}")
            continue

        print(f"[{i:03d}/{total}] [run]   {run_id} q={params['quantile']} z={params['as_zscore']} ex={params['exclude_last_days']}")
        p = ctx.Process(target=_worker_run_one, args=(params,))
        p.start()
        p.join()

        if p.exitcode != 0:
            # if child crashed hard, record it
            out_dir.mkdir(parents=True, exist_ok=True)
            error_path.write_text(f"Child process exit code: {p.exitcode}\n", encoding="utf-8")
            print(f"  -> [fail] exitcode={p.exitcode}")

    print("Grid finished.")


if __name__ == "__main__":
    main()
