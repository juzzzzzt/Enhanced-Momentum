from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant_pml.strategies.optimization_data import TrainingData

import numpy as np
import pandas as pd
from quant_pml.strategies.factors.sorting_strategy import SortingStrategy


class SystematicMomentum(SortingStrategy):
    def __init__(  # noqa: PLR0913
        self,
        mode: str,
        sign: int = 1,
        *,
        as_zscore: bool = False,
        window_days: int = 365,
        exclude_last_days: int = 30,
        quantile: float | None = None,
        n_holdings: int | None = None,
        weighting_scheme: str = "equally_weighted",
        return_type: str = "simple",
        volatility_scaling: bool = False,
        vol_window_days: int = 21,
    ) -> None:
        super().__init__(
            quantile=quantile,
            mode=mode,
            n_holdings=n_holdings,
            weighting_scheme=weighting_scheme,
        )
        self.sign = sign
        self.as_zscore = as_zscore
        self.window_days = window_days
        self.exclude_last_days = exclude_last_days

    def get_scores(self, data: TrainingData) -> pd.Series:  # noqa: ARG002
        """
        Classic cross-sectional momentum score:
          - lookback = window_days
          - skip = exclude_last_days (if 0 -> do NOT slice with :-0)
          - prices: mom = P_end / P_start - 1
          - returns: mom = prod(1+r) - 1
        """
        import numpy as np
        import pandas as pd

        assets = list(self.available_assets)

        # 1) Find a DataFrame inside TrainingData that looks like a panel of prices/returns
        panel = None
        for attr in ("prices", "price", "close", "close_prices", "returns", "rets", "data"):
            x = getattr(data, attr, None)
            if isinstance(x, pd.DataFrame):
                panel = x
                break
        if panel is None:
            raise AttributeError(
                "Cannot find a DataFrame with prices/returns inside TrainingData. "
                "Inspect `dir(data)` and add the correct attribute name."
            )

        # 2) Restrict columns to currently tradable assets
        panel = panel.reindex(columns=assets)

        win = int(self.window_days)
        skip = int(self.exclude_last_days)

        # 3) Correct slicing when skip == 0
        #    if skip>0: take [-win-skip : -skip]
        #    if skip==0: take [-win : None]
        start = -(win + skip) if skip > 0 else -win
        end = -skip if skip > 0 else None
        hist = panel.iloc[start:end]

        # Safety: should not be empty now (but keep guard anyway)
        if hist.shape[0] < 2:
            return pd.Series(index=assets, dtype="float64")

        # Decide if this is returns-like (has negatives and small magnitude) or prices-like
        neg_frac = float((hist < 0).mean().mean())
        med_abs = float(hist.abs().stack().median()) if hist.size else 0.0
        looks_like_returns = (neg_frac > 0.01) and (med_abs < 0.5)

        if looks_like_returns:
            mom = (1.0 + hist).prod(axis=0, skipna=True) - 1.0
        else:
            p_start = hist.iloc[0]
            p_end = hist.iloc[-1]
            mom = (p_end / p_start) - 1.0

        mom = mom.replace([np.inf, -np.inf], np.nan)

        # optional sign flip
        if int(self.sign) == -1:
            mom = -mom

        # optional cross-sectional z-score
        if self.as_zscore:
            m = mom.mean(skipna=True)
            s = mom.std(skipna=True, ddof=0)
            if s and float(s) > 0:
                mom = (mom - m) / s

        return mom.astype("float64")

