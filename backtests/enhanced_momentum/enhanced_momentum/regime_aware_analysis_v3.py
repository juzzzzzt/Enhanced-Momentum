from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ENS3_CONFIGS = [
    "finalist_1_q020_ex84_win126",
    "finalist_2_q030_ex84_win126",
    "finalist_3_q030_ex63_win126",
]

IS_WF_WINDOWS = [
    "wf_2015",
    "wf_2016",
    "wf_2017",
    "wf_2018",
    "wf_2019",
]

OOS_WINDOW = "full_oos_2020_2023"

TRADING_DAYS = 252


@dataclass(frozen=True)
class ReturnBundle:
    total_r: pd.Series
    excess_r: pd.Series
    market_total_r: pd.Series


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Cannot locate repo root")


def _read_config(run_dir: Path) -> dict[str, Any]:
    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _read_single_col_parquet(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    if isinstance(df, pd.Series):
        s = df.copy()
    else:
        if df.shape[1] != 1:
            raise ValueError(f"Expected single-column parquet at {path}, got {df.shape[1]} columns")
        s = df.iloc[:, 0].copy()

    s.index = pd.to_datetime(s.index)
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s.sort_index()


def _find_matching_runs(
    results_dir: Path,
    eval_windows: set[str],
    config_names: set[str],
) -> list[tuple[Path, dict[str, Any]]]:
    runs_dir = results_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Missing runs directory: {runs_dir}")

    out: list[tuple[Path, dict[str, Any]]] = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue

        cfg = _read_config(run_dir)
        params = cfg.get("params", {})

        if params.get("eval_window") not in eval_windows:
            continue

        if params.get("config_name") not in config_names:
            continue

        required = [
            run_dir / "strategy_total_r.parquet",
            run_dir / "strategy_excess_r.parquet",
            run_dir / "market_total_r.parquet",
        ]

        missing = [str(x) for x in required if not x.exists()]
        if missing:
            raise FileNotFoundError(f"Run {run_dir.name} is missing required files: {missing}")

        out.append((run_dir, params))

    if not out:
        raise RuntimeError(
            f"No matching runs found in {runs_dir}. "
            f"eval_windows={sorted(eval_windows)}, config_names={sorted(config_names)}"
        )

    return out


def _build_ens3_daily(
    results_dir: Path,
    eval_windows: list[str],
    config_names: list[str],
) -> ReturnBundle:
    runs = _find_matching_runs(
        results_dir=results_dir,
        eval_windows=set(eval_windows),
        config_names=set(config_names),
    )

    expected_count = len(eval_windows) * len(config_names)
    if len(runs) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} runs, found {len(runs)}. "
            f"results_dir={results_dir}"
        )

    total_cols = []
    excess_cols = []
    market_cols = []

    for run_dir, params in runs:
        config_name = str(params["config_name"])
        eval_window = str(params["eval_window"])
        unique_name = f"{config_name}__{eval_window}"

        total = _read_single_col_parquet(run_dir / "strategy_total_r.parquet")
        excess = _read_single_col_parquet(run_dir / "strategy_excess_r.parquet")
        market = _read_single_col_parquet(run_dir / "market_total_r.parquet")

        if not total.index.equals(excess.index):
            raise RuntimeError(f"total/excess index mismatch in {run_dir}")

        if not total.index.equals(market.index):
            raise RuntimeError(f"total/market index mismatch in {run_dir}")

        total.name = unique_name
        excess.name = unique_name
        market.name = unique_name

        total_cols.append(total)
        excess_cols.append(excess)
        market_cols.append(market)

    total_df = pd.concat(total_cols, axis=1, sort=True)
    excess_df = pd.concat(excess_cols, axis=1, sort=True)
    market_df = pd.concat(market_cols, axis=1, sort=True)

    max_total_non_na = int(total_df.notna().sum(axis=1).max())
    max_excess_non_na = int(excess_df.notna().sum(axis=1).max())
    max_market_non_na = int(market_df.notna().sum(axis=1).max())

    if max_total_non_na > len(config_names):
        raise RuntimeError(f"Overlapping total returns across WF folds: max_non_na={max_total_non_na}")

    if max_excess_non_na > len(config_names):
        raise RuntimeError(f"Overlapping excess returns across WF folds: max_non_na={max_excess_non_na}")

    if max_market_non_na > len(config_names):
        raise RuntimeError(f"Overlapping market returns across WF folds: max_non_na={max_market_non_na}")

    ens_total = total_df.mean(axis=1, skipna=True).dropna()
    ens_excess = excess_df.mean(axis=1, skipna=True).dropna()
    ens_market = market_df.mean(axis=1, skipna=True).dropna()

    common_index = ens_total.index.intersection(ens_excess.index).intersection(ens_market.index)

    ens_total = ens_total.loc[common_index]
    ens_excess = ens_excess.loc[common_index]
    ens_market = ens_market.loc[common_index]

    ens_total.name = "ens3_total_r"
    ens_excess.name = "ens3_excess_r"
    ens_market.name = "market_total_r"

    return ReturnBundle(
        total_r=ens_total,
        excess_r=ens_excess,
        market_total_r=ens_market,
    )


def _bundle_to_frame(bundle: ReturnBundle) -> pd.DataFrame:
    df = pd.concat(
        [
            bundle.total_r,
            bundle.excess_r,
            bundle.market_total_r,
        ],
        axis=1,
    ).dropna()

    df.columns = [
        "ens3_total_r",
        "ens3_excess_r",
        "market_total_r",
    ]

    return df


def _realized_vol(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).std(ddof=1) * math.sqrt(TRADING_DAYS)


def _trailing_return(s: pd.Series, window: int) -> pd.Series:
    return (1.0 + s).rolling(window).apply(np.prod, raw=True) - 1.0


def _drawdown(s: pd.Series) -> pd.Series:
    nav = (1.0 + s).cumprod()
    return nav / nav.cummax() - 1.0


def _build_daily_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["internal_realized_vol_21"] = _realized_vol(out["ens3_total_r"], 21)
    out["internal_realized_vol_63"] = _realized_vol(out["ens3_total_r"], 63)

    out["internal_trailing_total_r_21"] = _trailing_return(out["ens3_total_r"], 21)
    out["internal_trailing_total_r_63"] = _trailing_return(out["ens3_total_r"], 63)
    out["internal_trailing_excess_r_21"] = _trailing_return(out["ens3_excess_r"], 21)
    out["internal_trailing_excess_r_63"] = _trailing_return(out["ens3_excess_r"], 63)

    out["internal_drawdown"] = _drawdown(out["ens3_total_r"])
    out["internal_daily_total_r"] = out["ens3_total_r"]

    out["market_realized_vol_21"] = _realized_vol(out["market_total_r"], 21)
    out["market_realized_vol_63"] = _realized_vol(out["market_total_r"], 63)
    out["market_drawdown"] = _drawdown(out["market_total_r"])
    out["market_daily_total_r"] = out["market_total_r"]

    signal_cols = [
        "internal_realized_vol_21",
        "internal_realized_vol_63",
        "internal_trailing_total_r_21",
        "internal_trailing_total_r_63",
        "internal_trailing_excess_r_21",
        "internal_trailing_excess_r_63",
        "internal_drawdown",
        "internal_daily_total_r",
        "market_realized_vol_21",
        "market_realized_vol_63",
        "market_drawdown",
        "market_daily_total_r",
    ]

    for col in signal_cols:
        out[f"prev_{col}"] = out[col].shift(1)

    return out


def _q(s: pd.Series, q: float) -> float:
    return float(s.dropna().quantile(q))


def _calibrate_thresholds_on_is(is_signals: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("internal_median_vol_21", _q(is_signals["prev_internal_realized_vol_21"], 0.50)),
        ("internal_median_vol_63", _q(is_signals["prev_internal_realized_vol_63"], 0.50)),
        ("internal_q75_vol_21", _q(is_signals["prev_internal_realized_vol_21"], 0.75)),
        ("internal_q75_vol_63", _q(is_signals["prev_internal_realized_vol_63"], 0.75)),
        ("internal_q90_vol_21", _q(is_signals["prev_internal_realized_vol_21"], 0.90)),
        ("internal_q90_vol_63", _q(is_signals["prev_internal_realized_vol_63"], 0.90)),
        ("internal_q25_trailing_total_r_21", _q(is_signals["prev_internal_trailing_total_r_21"], 0.25)),
        ("internal_q25_trailing_total_r_63", _q(is_signals["prev_internal_trailing_total_r_63"], 0.25)),
        ("internal_q05_daily_total_r", _q(is_signals["prev_internal_daily_total_r"], 0.05)),
        ("internal_q01_daily_total_r", _q(is_signals["prev_internal_daily_total_r"], 0.01)),
        ("market_q75_vol_21", _q(is_signals["prev_market_realized_vol_21"], 0.75)),
        ("market_q75_vol_63", _q(is_signals["prev_market_realized_vol_63"], 0.75)),
        ("market_q90_vol_21", _q(is_signals["prev_market_realized_vol_21"], 0.90)),
        ("market_q90_vol_63", _q(is_signals["prev_market_realized_vol_63"], 0.90)),
        ("market_q10_drawdown", _q(is_signals["prev_market_drawdown"], 0.10)),
    ]

    return pd.DataFrame(rows, columns=["threshold", "value"])


def _threshold_dict(thresholds: pd.DataFrame) -> dict[str, float]:
    return dict(zip(thresholds["threshold"], thresholds["value"]))


def _cooldown_exposure(trigger: pd.Series, cooldown_days: int) -> pd.Series:
    trigger = trigger.fillna(False).astype(bool)
    exposure = pd.Series(1.0, index=trigger.index)

    cooldown_left = 0
    for dt in trigger.index:
        if bool(trigger.loc[dt]):
            cooldown_left = cooldown_days

        if cooldown_left > 0:
            exposure.loc[dt] = 0.0
            cooldown_left -= 1

    return exposure


def _clip_exposure(s: pd.Series, min_exposure: float = 0.0, max_exposure: float = 1.0) -> pd.Series:
    return s.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(min_exposure, max_exposure)


def _build_exposure_variants(signals: pd.DataFrame, thresholds: dict[str, float]) -> dict[str, dict[str, Any]]:
    idx = signals.index

    internal_vol21 = signals["prev_internal_realized_vol_21"]
    internal_vol63 = signals["prev_internal_realized_vol_63"]
    market_vol21 = signals["prev_market_realized_vol_21"]
    market_vol63 = signals["prev_market_realized_vol_63"]

    internal_vol63_q90 = internal_vol63 > thresholds["internal_q90_vol_63"]
    market_vol63_q90 = market_vol63 > thresholds["market_q90_vol_63"]

    variants: dict[str, dict[str, Any]] = {}

    def add(name: str, exposure: pd.Series, signal_type: str, description: str) -> None:
        exposure = exposure.reindex(idx).astype(float).fillna(1.0).clip(0.0, 1.0)
        variants[name] = {
            "exposure": exposure,
            "signal_type": signal_type,
            "description": description,
        }

    add(
        "baseline_no_protection",
        pd.Series(1.0, index=idx),
        "baseline",
        "No exposure scaling",
    )

    add(
        "internal_inverse_vol21_to_is_median",
        _clip_exposure(thresholds["internal_median_vol_21"] / internal_vol21),
        "internal_portfolio",
        "Continuous inverse-vol scaling using internal 21-day vol to IS median",
    )

    add(
        "internal_inverse_vol63_to_is_median",
        _clip_exposure(thresholds["internal_median_vol_63"] / internal_vol63),
        "internal_portfolio",
        "Continuous inverse-vol scaling using internal 63-day vol to IS median",
    )

    add(
        "internal_binary_vol21_above_is_q75",
        (~(internal_vol21 > thresholds["internal_q75_vol_21"])).astype(float),
        "internal_portfolio",
        "Exit when internal 21-day vol is above IS q75",
    )

    add(
        "internal_binary_vol63_above_is_q75",
        (~(internal_vol63 > thresholds["internal_q75_vol_63"])).astype(float),
        "internal_portfolio",
        "Exit when internal 63-day vol is above IS q75",
    )

    add(
        "internal_binary_vol21_above_is_q90",
        (~(internal_vol21 > thresholds["internal_q90_vol_21"])).astype(float),
        "internal_portfolio",
        "Exit when internal 21-day vol is above IS q90",
    )

    add(
        "internal_binary_vol63_above_is_q90",
        (~internal_vol63_q90).astype(float),
        "internal_portfolio",
        "Exit when internal 63-day vol is above IS q90",
    )

    add(
        "internal_binary_trailing_total21_below_is_q25",
        (~(signals["prev_internal_trailing_total_r_21"] < thresholds["internal_q25_trailing_total_r_21"])).astype(float),
        "internal_portfolio",
        "Exit when internal trailing 21-day total return is below IS q25",
    )

    add(
        "internal_binary_trailing_total63_below_is_q25",
        (~(signals["prev_internal_trailing_total_r_63"] < thresholds["internal_q25_trailing_total_r_63"])).astype(float),
        "internal_portfolio",
        "Exit when internal trailing 63-day total return is below IS q25",
    )

    add(
        "internal_binary_drawdown_below_minus_10pct",
        (~(signals["prev_internal_drawdown"] < -0.10)).astype(float),
        "internal_portfolio",
        "Exit when internal drawdown is below -10%",
    )

    add(
        "internal_cooldown_21_after_q05_daily_loss",
        _cooldown_exposure(
            signals["prev_internal_daily_total_r"] < thresholds["internal_q05_daily_total_r"],
            cooldown_days=21,
        ),
        "internal_portfolio",
        "Exit for 21 trading days after internal daily loss below IS q05",
    )

    add(
        "internal_cooldown_42_after_q01_daily_loss",
        _cooldown_exposure(
            signals["prev_internal_daily_total_r"] < thresholds["internal_q01_daily_total_r"],
            cooldown_days=42,
        ),
        "internal_portfolio",
        "Exit for 42 trading days after internal daily loss below IS q01",
    )

    add(
        "external_binary_market_vol21_above_is_q90",
        (~(market_vol21 > thresholds["market_q90_vol_21"])).astype(float),
        "external_market",
        "Exit when S&P 500 21-day realized vol is above IS q90",
    )

    add(
        "external_binary_market_vol63_above_is_q90",
        (~(market_vol63 > thresholds["market_q90_vol_63"])).astype(float),
        "external_market",
        "Exit when S&P 500 63-day realized vol is above IS q90",
    )

    add(
        "external_binary_market_vol63_above_is_q75",
        (~(market_vol63 > thresholds["market_q75_vol_63"])).astype(float),
        "external_market",
        "Exit when S&P 500 63-day realized vol is above IS q75",
    )

    add(
        "external_binary_market_drawdown_below_is_q10",
        (~(signals["prev_market_drawdown"] < thresholds["market_q10_drawdown"])).astype(float),
        "external_market",
        "Exit when S&P 500 drawdown is below IS q10",
    )

    add(
        "combo_vol63_q90_either",
        (~(internal_vol63_q90 | market_vol63_q90)).astype(float),
        "combo_internal_external",
        "Exit when either internal 63-day vol or S&P 500 63-day vol is above its IS q90",
    )

    add(
        "combo_vol63_q90_both",
        (~(internal_vol63_q90 & market_vol63_q90)).astype(float),
        "combo_internal_external",
        "Exit only when both internal 63-day vol and S&P 500 63-day vol are above their IS q90",
    )

    return variants


def _max_drawdown(r: pd.Series) -> float:
    nav = (1.0 + r).cumprod()
    dd = nav / nav.cummax() - 1.0
    return float(dd.min())


def _geom_ann_return(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan

    nav = float((1.0 + r).prod())
    if nav <= 0:
        return -1.0

    return nav ** (TRADING_DAYS / len(r)) - 1.0


def _ann_vol(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))


def _sharpe(r: pd.Series) -> float:
    r = r.dropna()
    vol = r.std(ddof=1)
    if len(r) < 2 or vol == 0 or pd.isna(vol):
        return np.nan
    return float(r.mean() / vol * math.sqrt(TRADING_DAYS))


def _evaluate_variant(
    base_df: pd.DataFrame,
    exposure: pd.Series,
    variant_name: str,
    signal_type: str,
    description: str,
    tc_bps: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    exposure = exposure.reindex(base_df.index).astype(float).fillna(1.0).clip(0.0, 1.0)

    prev_exposure = exposure.shift(1).fillna(1.0)
    turnover = (exposure - prev_exposure).abs()
    tc = turnover * tc_bps / 10_000.0

    scaled_total_r = exposure * base_df["ens3_total_r"] - tc
    scaled_excess_r = exposure * base_df["ens3_excess_r"] - tc

    switches = int((turnover > 1e-12).sum())
    tc_drag = float(tc.sum())

    row = {
        "variant": variant_name,
        "signal_type": signal_type,
        "description": description,
        "n_days": int(len(base_df)),
        "final_nav": float((1.0 + scaled_total_r).prod()),
        "geom_avg_total_r": _geom_ann_return(scaled_total_r),
        "geom_avg_excess_r": _geom_ann_return(scaled_excess_r),
        "ann_vol_total_r": _ann_vol(scaled_total_r),
        "ann_vol_excess_r": _ann_vol(scaled_excess_r),
        "sharpe": _sharpe(scaled_total_r),
        "ir": _sharpe(scaled_excess_r),
        "max_dd": _max_drawdown(scaled_total_r),
        "mean_exposure": float(exposure.mean()),
        "min_exposure": float(exposure.min()),
        "pct_days_out": float((exposure < 0.5).mean()),
        "switches": switches,
        "tc_drag": tc_drag,
        "tc_bps": float(tc_bps),
    }

    daily = pd.DataFrame(
        {
            "variant": variant_name,
            "signal_type": signal_type,
            "exposure": exposure,
            "turnover": turnover,
            "tc": tc,
            "scaled_total_r": scaled_total_r,
            "scaled_excess_r": scaled_excess_r,
        },
        index=base_df.index,
    )

    return row, daily


def _evaluate_all_variants(
    base_df: pd.DataFrame,
    variants: dict[str, dict[str, Any]],
    tc_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    daily_frames = []

    for name, payload in variants.items():
        row, daily = _evaluate_variant(
            base_df=base_df,
            exposure=payload["exposure"],
            variant_name=name,
            signal_type=payload["signal_type"],
            description=payload["description"],
            tc_bps=tc_bps,
        )
        rows.append(row)
        daily_frames.append(daily)

    results = pd.DataFrame(rows).sort_values(
        ["ir", "max_dd"],
        ascending=[False, False],
    )

    baseline_ir = float(results.loc[results["variant"] == "baseline_no_protection", "ir"].iloc[0])
    baseline_max_dd = float(results.loc[results["variant"] == "baseline_no_protection", "max_dd"].iloc[0])
    baseline_sharpe = float(results.loc[results["variant"] == "baseline_no_protection", "sharpe"].iloc[0])

    results["ir_delta_vs_baseline"] = results["ir"] - baseline_ir
    results["max_dd_delta_vs_baseline"] = results["max_dd"] - baseline_max_dd
    results["sharpe_delta_vs_baseline"] = results["sharpe"] - baseline_sharpe

    daily_all = pd.concat(daily_frames, axis=0)
    daily_all.index.name = "date"

    return results, daily_all


def _yearly_breakdown(daily_all: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (variant, year), g in daily_all.groupby(["variant", daily_all.index.year]):
        signal_type = str(g["signal_type"].iloc[0])
        total_r = g["scaled_total_r"]
        excess_r = g["scaled_excess_r"]

        rows.append(
            {
                "variant": variant,
                "signal_type": signal_type,
                "year": int(year),
                "n_days": int(len(g)),
                "final_nav": float((1.0 + total_r).prod()),
                "geom_avg_total_r": _geom_ann_return(total_r),
                "geom_avg_excess_r": _geom_ann_return(excess_r),
                "sharpe": _sharpe(total_r),
                "ir": _sharpe(excess_r),
                "max_dd": _max_drawdown(total_r),
                "mean_exposure": float(g["exposure"].mean()),
                "switches": int((g["turnover"] > 1e-12).sum()),
                "tc_drag": float(g["tc"].sum()),
            }
        )

    return pd.DataFrame(rows).sort_values(["variant", "year"])


def _monthly_returns(df: pd.DataFrame, total_col: str, excess_col: str, market_col: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.resample("ME").last().index)
    out["ens3_total_r"] = df[total_col].resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
    out["ens3_excess_r"] = df[excess_col].resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
    out["market_total_r"] = df[market_col].resample("ME").apply(lambda x: (1.0 + x).prod() - 1.0)
    return out.dropna(how="all")


def _crash_month_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for date, row in monthly.iterrows():
        rows.append(
            {
                "month": date.strftime("%Y-%m"),
                "ens3_total_r": row["ens3_total_r"],
                "ens3_excess_r": row["ens3_excess_r"],
                "market_total_r": row["market_total_r"],
            }
        )

    df = pd.DataFrame(rows)
    return df.sort_values("ens3_total_r").head(12)


def _first_exit_date(exposure: pd.Series, start: str, end: str) -> pd.Timestamp | pd.NaT:
    window = exposure.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if window.empty:
        return pd.NaT

    out = window[window < 0.5]
    if out.empty:
        return pd.NaT

    return pd.Timestamp(out.index[0])


def _lead_time_analysis(
    is_variants: dict[str, dict[str, Any]],
    oos_variants: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    episodes = [
        {
            "episode": "is_2016_q1",
            "sample": "is_wf",
            "start": "2016-01-01",
            "end": "2016-03-31",
            "variants": is_variants,
        },
        {
            "episode": "oos_2020_crash",
            "sample": "oos",
            "start": "2020-02-15",
            "end": "2020-04-30",
            "variants": oos_variants,
        },
        {
            "episode": "oos_2022_hikes",
            "sample": "oos",
            "start": "2022-01-01",
            "end": "2022-12-31",
            "variants": oos_variants,
        },
    ]

    comparisons = [
        ("internal_binary_vol63_above_is_q90", "external_binary_market_vol63_above_is_q90"),
        ("internal_binary_vol21_above_is_q90", "external_binary_market_vol21_above_is_q90"),
        ("internal_binary_vol63_above_is_q90", "combo_vol63_q90_either"),
        ("internal_binary_vol63_above_is_q90", "combo_vol63_q90_both"),
    ]

    rows = []

    for ep in episodes:
        variants = ep["variants"]

        for internal_name, challenger_name in comparisons:
            internal_date = _first_exit_date(
                variants[internal_name]["exposure"],
                ep["start"],
                ep["end"],
            )
            challenger_date = _first_exit_date(
                variants[challenger_name]["exposure"],
                ep["start"],
                ep["end"],
            )

            if pd.isna(internal_date) or pd.isna(challenger_date):
                lead_time_days = np.nan
            else:
                lead_time_days = int((internal_date - challenger_date).days)

            rows.append(
                {
                    "episode": ep["episode"],
                    "sample": ep["sample"],
                    "start": ep["start"],
                    "end": ep["end"],
                    "internal_variant": internal_name,
                    "challenger_variant": challenger_name,
                    "internal_first_exit_date": None if pd.isna(internal_date) else internal_date.date().isoformat(),
                    "challenger_first_exit_date": None if pd.isna(challenger_date) else challenger_date.date().isoformat(),
                    "lead_time_days_positive_means_challenger_earlier": lead_time_days,
                }
            )

    return pd.DataFrame(rows)


def _save_research_log(
    out_dir: Path,
    results: pd.DataFrame,
    thresholds: pd.DataFrame,
    lead_time: pd.DataFrame,
    tc_bps: float,
) -> None:
    best = results.sort_values(["ir", "max_dd"], ascending=[False, False]).iloc[0]
    internal = results.loc[results["variant"] == "internal_binary_vol63_above_is_q90"].iloc[0]

    external_candidates = results[results["signal_type"] == "external_market"]
    combo_candidates = results[results["signal_type"] == "combo_internal_external"]

    best_external = external_candidates.sort_values(["ir", "max_dd"], ascending=[False, False]).iloc[0]
    best_combo = combo_candidates.sort_values(["ir", "max_dd"], ascending=[False, False]).iloc[0]

    lines = [
        f"Date: {datetime.utcnow().isoformat()}Z",
        "Stage: External Signal Integration / Regime-Aware Protection v3",
        "",
        "Data:",
        "- Internal portfolio signal: ens-3 total/excess returns.",
        "- External market signal: runner.data['spx'], saved as market_total_r.parquet.",
        "- Market returns are aligned to strategy_total_r dates inside each run.",
        "",
        "Methodology:",
        "- Thresholds calibrated only on IS walk-forward test returns 2015-2019.",
        "- Thresholds frozen before OOS 2020-2023 evaluation.",
        "- All signal features are lagged by one day to avoid look-ahead.",
        f"- Transaction cost model: abs(delta exposure) * {tc_bps} bps.",
        "",
        "Variants:",
        "- Internal portfolio vol / trailing return / drawdown / cooldown variants.",
        "- External S&P 500 realized-vol and drawdown variants.",
        "- Combo OR/AND variants based on internal vol63 q90 and market vol63 q90.",
        "",
        "Key thresholds:",
    ]

    for _, row in thresholds.iterrows():
        lines.append(f"- {row['threshold']}: {row['value']:.8f}")

    lines += [
        "",
        "Best overall:",
        f"- variant: {best['variant']}",
        f"- signal_type: {best['signal_type']}",
        f"- IR: {best['ir']:.6f}",
        f"- Sharpe: {best['sharpe']:.6f}",
        f"- max_dd: {best['max_dd']:.6f}",
        f"- final_nav: {best['final_nav']:.6f}",
        "",
        "Internal winner reference:",
        f"- variant: {internal['variant']}",
        f"- IR: {internal['ir']:.6f}",
        f"- Sharpe: {internal['sharpe']:.6f}",
        f"- max_dd: {internal['max_dd']:.6f}",
        f"- final_nav: {internal['final_nav']:.6f}",
        "",
        "Best external:",
        f"- variant: {best_external['variant']}",
        f"- IR: {best_external['ir']:.6f}",
        f"- Sharpe: {best_external['sharpe']:.6f}",
        f"- max_dd: {best_external['max_dd']:.6f}",
        f"- final_nav: {best_external['final_nav']:.6f}",
        "",
        "Best combo:",
        f"- variant: {best_combo['variant']}",
        f"- IR: {best_combo['ir']:.6f}",
        f"- Sharpe: {best_combo['sharpe']:.6f}",
        f"- max_dd: {best_combo['max_dd']:.6f}",
        f"- final_nav: {best_combo['final_nav']:.6f}",
        "",
        "Lead-time summary:",
    ]

    for _, row in lead_time.iterrows():
        lines.append(
            "- "
            f"{row['episode']} | {row['challenger_variant']} vs {row['internal_variant']} | "
            f"internal={row['internal_first_exit_date']} | "
            f"challenger={row['challenger_first_exit_date']} | "
            f"lead_days={row['lead_time_days_positive_means_challenger_earlier']}"
        )

    lines += [
        "",
        "Interpretation guide:",
        "- Positive lead_time_days means challenger exited earlier than internal reference.",
        "- External is useful if it improves IR/maxDD or gives earlier exit without material metric degradation.",
        "- If external/combo are worse than internal, conclusion is still useful: market vol was tested and rejected.",
    ]

    (out_dir / "research_log_regime_v3.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--is-results-subdir", default="data/results_is_wf_with_returns")
    parser.add_argument("--oos-results-subdir", default="data/results_oos_with_returns")
    parser.add_argument("--output-subdir", default="data/results_regime_v3")
    parser.add_argument("--tc-bps", type=float, default=25.0)

    args = parser.parse_args()

    repo_root = _repo_root()

    is_results_dir = repo_root / args.is_results_subdir
    oos_results_dir = repo_root / args.oos_results_subdir
    out_dir = repo_root / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    is_bundle = _build_ens3_daily(
        results_dir=is_results_dir,
        eval_windows=IS_WF_WINDOWS,
        config_names=ENS3_CONFIGS,
    )

    oos_bundle = _build_ens3_daily(
        results_dir=oos_results_dir,
        eval_windows=[OOS_WINDOW],
        config_names=ENS3_CONFIGS,
    )

    is_daily = _bundle_to_frame(is_bundle)
    oos_daily = _bundle_to_frame(oos_bundle)

    is_signals = _build_daily_signals(is_daily)
    oos_signals = _build_daily_signals(oos_daily)

    thresholds = _calibrate_thresholds_on_is(is_signals)
    thresholds_dict = _threshold_dict(thresholds)

    is_variants = _build_exposure_variants(is_signals, thresholds_dict)
    oos_variants = _build_exposure_variants(oos_signals, thresholds_dict)

    oos_results, oos_daily_scaled = _evaluate_all_variants(
        base_df=oos_daily,
        variants=oos_variants,
        tc_bps=args.tc_bps,
    )

    is_results, is_daily_scaled = _evaluate_all_variants(
        base_df=is_daily,
        variants=is_variants,
        tc_bps=args.tc_bps,
    )

    oos_by_year = _yearly_breakdown(oos_daily_scaled)
    is_by_year = _yearly_breakdown(is_daily_scaled)

    is_monthly = _monthly_returns(
        is_daily,
        total_col="ens3_total_r",
        excess_col="ens3_excess_r",
        market_col="market_total_r",
    )

    oos_monthly = _monthly_returns(
        oos_daily,
        total_col="ens3_total_r",
        excess_col="ens3_excess_r",
        market_col="market_total_r",
    )

    is_crash_months = _crash_month_summary(is_monthly)
    oos_crash_months = _crash_month_summary(oos_monthly)

    lead_time = _lead_time_analysis(is_variants, oos_variants)

    is_daily.to_csv(out_dir / "is_wf_ens3_daily_returns_v3.csv")
    oos_daily.to_csv(out_dir / "oos_ens3_daily_returns_v3.csv")

    is_monthly.to_csv(out_dir / "is_wf_ens3_monthly_returns_v3.csv")
    oos_monthly.to_csv(out_dir / "oos_ens3_monthly_returns_v3.csv")

    is_signals.to_csv(out_dir / "is_wf_daily_signals_v3.csv")
    oos_signals.to_csv(out_dir / "oos_daily_signals_v3.csv")

    thresholds.to_csv(out_dir / "is_calibrated_thresholds_v3.csv", index=False)

    is_results.to_csv(out_dir / "is_exposure_scaling_results_v3.csv", index=False)
    oos_results.to_csv(out_dir / "oos_exposure_scaling_results_v3.csv", index=False)

    is_by_year.to_csv(out_dir / "is_exposure_scaling_by_year_v3.csv", index=False)
    oos_by_year.to_csv(out_dir / "oos_exposure_scaling_by_year_v3.csv", index=False)

    is_daily_scaled.to_csv(out_dir / "is_exposure_scaled_daily_returns_v3.csv")
    oos_daily_scaled.to_csv(out_dir / "oos_exposure_scaled_daily_returns_v3.csv")

    is_crash_months.to_csv(out_dir / "is_wf_crash_month_summary_v3.csv", index=False)
    oos_crash_months.to_csv(out_dir / "oos_crash_month_summary_v3.csv", index=False)

    benchmark_daily = pd.DataFrame(
        {
            "market_total_r": oos_daily["market_total_r"],
        }
    )
    benchmark_daily.to_csv(out_dir / "benchmark_daily_returns.csv")

    lead_time.to_csv(out_dir / "signal_lead_time_analysis.csv", index=False)

    _save_research_log(
        out_dir=out_dir,
        results=oos_results,
        thresholds=thresholds,
        lead_time=lead_time,
        tc_bps=args.tc_bps,
    )

    print("Regime-aware v3 analysis finished.")
    print(f"Output directory: {out_dir}")
    print()
    print("Top OOS variants:")
    cols = [
        "variant",
        "signal_type",
        "final_nav",
        "ir",
        "sharpe",
        "max_dd",
        "mean_exposure",
        "switches",
        "tc_drag",
    ]
    print(oos_results[cols].head(15).to_string(index=False))
    print()
    print("Lead-time analysis:")
    print(lead_time.to_string(index=False))


if __name__ == "__main__":
    main()