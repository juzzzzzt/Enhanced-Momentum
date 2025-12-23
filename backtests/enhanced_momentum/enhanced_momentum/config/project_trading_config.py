from __future__ import annotations

from dataclasses import dataclass

from quant_pml.config.trading_config import TradingConfig


@dataclass
class ProjectTradingConfig(TradingConfig):
    # Broker Fees
    broker_fee: float = 0.05 / 100

    # Trading Setup
    trading_lag_days: int | None = 1
