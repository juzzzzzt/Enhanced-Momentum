from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_pml.config.topn_experiment_config import TopNExperimentConfig


class ProjectExperimentConfig(TopNExperimentConfig):
    def __init__(self) -> None:
        super().__init__(topn=3_000)

        self.PATH_OUTPUT = Path(__file__).resolve().parents[4] / "data" / "datasets"

        self.HEDGE_FREQ = "D"
        self.DATA_PROCESSING_START_DATE = pd.Timestamp("2000-01-01")
        self.START_DATE = pd.Timestamp("2010-01-01")
        self.CAUSAL_WINDOW_END_DATE_FIELD = "end_date"
