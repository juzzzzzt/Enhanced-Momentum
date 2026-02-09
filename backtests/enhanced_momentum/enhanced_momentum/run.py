from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from enhanced_momentum.config.project_experiment_config import ProjectExperimentConfig
from enhanced_momentum.config.project_trading_config import ProjectTradingConfig
from quant_pml.hedge.market_futures_hedge import MarketFuturesHedge
from quant_pml.runner import build_backtest

if TYPE_CHECKING:
    from quant_pml.strategies.base_strategy import BaseStrategy


# =========================
# Paths / caching helpers
# =========================
def _repo_root() -> Path:
    """Find repository root by walking up until '.git' folder is found."""
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Cannot locate repo root (no .git found in parents).")


def _run_id(params: dict[str, Any]) -> str:
    """Stable run id based only on strategy/experiment params."""
    payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:12]


def _git_commit(repo_root: Path) -> str | None:
    """Best-effort git commit hash (for provenance)."""
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


# =========================
# Backtest runner
# =========================
def run_backtest(  # noqa: PLR0913
    strategy: BaseStrategy,
    rebal_freq: str = "D",
    experiment_cfg: ProjectExperimentConfig | None = None,
    trading_cfg: ProjectTradingConfig | None = None,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
    plot: bool = True,
) -> pd.DataFrame:
    hedger = MarketFuturesHedge(market_name="spx")

    strategy_name = strategy.__class__.__name__
    cfg = experiment_cfg if experiment_cfg is not None else ProjectExperimentConfig()

    preprocessor, runner = build_backtest(
        experiment_config=cfg,
        trading_config=trading_cfg if trading_cfg is not None else ProjectTradingConfig(),
        rebal_freq=rebal_freq,
        start=start_date,
        end=end_date,
    )

    res = runner(
        feature_processor=preprocessor,
        strategy=strategy,
        hedger=hedger,
    )
    # print("RES TYPE:", type(res))
    # print("RES DIR SAMPLE:", [x for x in dir(res) if
    #                           "nav" in x.lower() or "ret" in x.lower() or "series" in x.lower() or "pandas" in x.lower()][
    #                          :50])

    if plot:
        runner.plot_cumulative(
            strategy_name=strategy_name,
            include_factors=True,
        )
        runner.plot_cumulative(
            strategy_name=strategy_name,
            include_factors=True,
            start_date=cfg.END_DATE - pd.Timedelta(days=365 * 2),
        )

    # StrategyStatistics -> DataFrame (metrics table)
    return res.to_pandas()


# =========================
# Main entrypoint
# =========================
if __name__ == "__main__":
    from enhanced_momentum.strategies.systematic_momentum import SystematicMomentum

    params: dict[str, Any] = {
        "strategy": "SystematicMomentum",
        "mode": "long_short",
        "quantile": 0.12,
        "window_days": 252,
        "exclude_last_days": 21,
        "as_zscore": False,
        "weighting_scheme": "equally_weighted",
        "rebal_freq": "ME",
        "start_date": "2022-01-01",
        "end_date": None,
    }

    repo_root = _repo_root()
    run_id = _run_id(params)

    out_dir = repo_root / "data" / "results" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "config.json"
    metrics_path = out_dir / "metrics.parquet"

    meta = {
        "run_id": run_id,
        "repo_root": str(repo_root),
        "git_commit": _git_commit(repo_root),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    config_path.write_text(
        json.dumps({"params": params, "meta": meta}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    if metrics_path.exists():
        print(f"[cache] {run_id} -> {metrics_path}")
        metrics = pd.read_parquet(metrics_path)
    else:
        print(f"[run] {run_id} computing...")

        sys_mom = SystematicMomentum(
            mode=params["mode"],
            quantile=params["quantile"],
            window_days=params["window_days"],
            exclude_last_days=params["exclude_last_days"],
            as_zscore=params["as_zscore"],
            weighting_scheme=params["weighting_scheme"],
        )

        metrics = run_backtest(
            strategy=sys_mom,
            rebal_freq=params["rebal_freq"],
            start_date=pd.Timestamp(params["start_date"]),
            end_date=pd.Timestamp(params["end_date"]) if params["end_date"] else None,
            plot=True,  # <-- ВАЖНО: не make_plots
        )

        metrics.to_parquet(metrics_path)

    print(metrics)
