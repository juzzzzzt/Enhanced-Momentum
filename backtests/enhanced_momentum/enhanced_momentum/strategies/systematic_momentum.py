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
        """(YOUR CODE HERE)."""
        n_assets = len(self.available_assets)
        np.random.seed(123)  # noqa: NPY002
        scores = np.random.randint(low=0, high=n_assets, size=n_assets)  # noqa: NPY002
        return pd.Series(scores, index=self.available_assets)
