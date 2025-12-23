from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from config.project_experiment_config import ProjectExperimentConfig
from config.project_trading_config import ProjectTradingConfig
from quant_pml.hedge.market_futures_hedge import MarketFuturesHedge
from quant_pml.runner import build_backtest

if TYPE_CHECKING:
    from quant_pml.strategies.base_strategy import BaseStrategy


def run_backtest(  # noqa: PLR0913
    strategy: BaseStrategy,
    rebal_freq: str = "D",
    experiment_cfg: ProjectExperimentConfig | None = None,
    trading_cfg: ProjectTradingConfig | None = None,
    start_date: pd.Timestamp | str | None = None,
    end_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    hedger = MarketFuturesHedge(market_name="spx")

    strategy_name = strategy.__class__.__name__
    cfg = experiment_cfg if experiment_cfg is not None else ProjectExperimentConfig()

    preprocessor, runner = build_backtest(
        experiment_config=cfg,
        trading_config=trading_cfg
        if trading_cfg is not None
        else ProjectTradingConfig(),
        rebal_freq=rebal_freq,
        start=start_date,
        end=end_date,
    )

    res = runner(
        feature_processor=preprocessor,
        strategy=strategy,
        hedger=hedger,
    )

    runner.plot_cumulative(
        strategy_name=strategy_name,
        include_factors=True,
    )

    runner.plot_cumulative(
        strategy_name=strategy_name,
        include_factors=True,
        start_date=cfg.END_DATE - pd.Timedelta(days=365 * 2),
    )

    return res.to_pandas()


if __name__ == "__main__":
    from enhanced_momentum.strategies.systematic_momentum import SystematicMomentum

    sys_mom = SystematicMomentum(
        mode="long_short",
        quantile=0.1,
    )

    result = run_backtest(
        strategy=sys_mom,
        rebal_freq="ME",
        start_date=pd.Timestamp("2022-01-01"),
    )
    print(result)  # noqa: T201
