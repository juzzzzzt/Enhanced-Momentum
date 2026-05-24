from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


ENS3_CONFIGS = [
    "finalist_1_q020_ex84_win126",
    "finalist_2_q030_ex84_win126",
    "finalist_3_q030_ex63_win126",
]

IS_ROOT = Path("data/results_is_wf_with_returns/runs")
OOS_ROOT = Path("data/results_oos_with_returns/runs")
OUT_DIR = Path("data/results_regime_v4")

TC_BPS = 25.0
TRADING_DAYS = 252


def repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Cannot locate repo root")


def read_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["params"]


def read_series(path: Path, name: str) -> pd.Series:
    obj = pd.read_parquet(path)
    if isinstance(obj, pd.DataFrame):
        s = obj.iloc[:, 0]
    else:
        s = obj
    s = pd.to_numeric(s, errors="coerce")
    s.index = pd.to_datetime(s.index)
    s.name = name
    return s.dropna()


def read_runs(
    root: Path,
    config_names: list[str],
    eval_window: str | None = None,
) -> dict[tuple[str, str], pd.DataFrame]:
    out: dict[tuple[str, str], pd.DataFrame] = {}

    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue

        config_path = run_dir / "config.json"
        total_path = run_dir / "strategy_total_r.parquet"
        excess_path = run_dir / "strategy_excess_r.parquet"
        market_path = run_dir / "market_total_r.parquet"
        momentum_path = run_dir / "momentum_factor_r.parquet"

        if not config_path.exists():
            continue

        if not (
            total_path.exists()
            and excess_path.exists()
            and market_path.exists()
            and momentum_path.exists()
        ):
            continue

        cfg = read_config(run_dir)

        if cfg["config_name"] not in config_names:
            continue

        if eval_window is not None and cfg["eval_window"] != eval_window:
            continue

        total = read_series(total_path, "strategy_total_r")
        excess = read_series(excess_path, "strategy_excess_r")
        market = read_series(market_path, "market_total_r")
        momentum = read_series(momentum_path, "momentum_factor_r")

        df = pd.concat([total, excess, market, momentum], axis=1).dropna()

        if df.empty:
            continue

        if not (
            df.index.equals(total.loc[df.index].index)
            and df.index.equals(excess.loc[df.index].index)
            and df.index.equals(market.loc[df.index].index)
            and df.index.equals(momentum.loc[df.index].index)
        ):
            raise RuntimeError(f"Index alignment failed for {run_dir}")

        out[(cfg["eval_window"], cfg["config_name"])] = df

    if not out:
        raise RuntimeError(f"No runs found in {root}")

    return out


def build_ensemble(
    runs: dict[tuple[str, str], pd.DataFrame],
    config_names: list[str],
) -> pd.DataFrame:
    frames = []

    eval_windows = sorted({k[0] for k in runs.keys()})

    for ew in eval_windows:
        per_config = []

        for cfg_name in config_names:
            key = (ew, cfg_name)
            if key not in runs:
                raise RuntimeError(f"Missing run for eval_window={ew}, config={cfg_name}")
            per_config.append(runs[key])

        panel = pd.concat(per_config, axis=1, keys=config_names).dropna()

        out = pd.DataFrame(index=panel.index)
        out["strategy_total_r"] = panel.xs("strategy_total_r", axis=1, level=1).mean(axis=1)
        out["strategy_excess_r"] = panel.xs("strategy_excess_r", axis=1, level=1).mean(axis=1)
        out["market_total_r"] = panel.xs("market_total_r", axis=1, level=1).mean(axis=1)
        out["momentum_factor_r"] = panel.xs("momentum_factor_r", axis=1, level=1).mean(axis=1)
        out["eval_window"] = ew

        frames.append(out)

    result = pd.concat(frames).sort_index()
    return result


def rolling_compound_return(s: pd.Series, window: int) -> pd.Series:
    return (1.0 + s).rolling(window).apply(np.prod, raw=True) - 1.0


def rolling_ann_vol(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).std() * np.sqrt(TRADING_DAYS)


def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["strategy_vol21"] = rolling_ann_vol(out["strategy_total_r"], 21)
    out["strategy_vol63"] = rolling_ann_vol(out["strategy_total_r"], 63)

    out["market_vol21"] = rolling_ann_vol(out["market_total_r"], 21)
    out["market_vol63"] = rolling_ann_vol(out["market_total_r"], 63)

    out["mom_trailing21"] = rolling_compound_return(out["momentum_factor_r"], 21)
    out["mom_trailing63"] = rolling_compound_return(out["momentum_factor_r"], 63)
    out["mom_vol21"] = rolling_ann_vol(out["momentum_factor_r"], 21)
    out["mom_vol63"] = rolling_ann_vol(out["momentum_factor_r"], 63)

    signal_cols = [
        "strategy_vol21",
        "strategy_vol63",
        "market_vol21",
        "market_vol63",
        "mom_trailing21",
        "mom_trailing63",
        "mom_vol21",
        "mom_vol63",
    ]

    for col in signal_cols:
        out[f"prev_{col}"] = out[col].shift(1)

    return out


def calibrate_thresholds(is_df: pd.DataFrame) -> dict[str, float]:
    thresholds = {
        "internal_vol63_q90": float(is_df["prev_strategy_vol63"].quantile(0.90)),
        "internal_vol63_q75": float(is_df["prev_strategy_vol63"].quantile(0.75)),
        "market_vol63_q90": float(is_df["prev_market_vol63"].quantile(0.90)),
        "market_vol63_q75": float(is_df["prev_market_vol63"].quantile(0.75)),
        "mom_trailing21_q10": float(is_df["prev_mom_trailing21"].quantile(0.10)),
        "mom_trailing63_q10": float(is_df["prev_mom_trailing63"].quantile(0.10)),
        "mom_trailing21_q25": float(is_df["prev_mom_trailing21"].quantile(0.25)),
        "mom_trailing63_q25": float(is_df["prev_mom_trailing63"].quantile(0.25)),
        "mom_vol63_q90": float(is_df["prev_mom_vol63"].quantile(0.90)),
    }
    return thresholds


def max_drawdown(r: pd.Series) -> float:
    nav = (1.0 + r).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def annualized_return(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    nav = float((1.0 + r).prod())
    years = len(r) / TRADING_DAYS
    if years <= 0:
        return np.nan
    return nav ** (1.0 / years) - 1.0


def annualized_vol(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def annualized_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    vol = r.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return float(r.mean() / vol * np.sqrt(TRADING_DAYS))


def build_variants(thresholds: dict[str, float]) -> dict[str, dict]:
    def internal_vol63_q90(df: pd.DataFrame) -> pd.Series:
        return df["prev_strategy_vol63"] > thresholds["internal_vol63_q90"]

    def external_market_vol63_q90(df: pd.DataFrame) -> pd.Series:
        return df["prev_market_vol63"] > thresholds["market_vol63_q90"]

    def combo_vol_and(df: pd.DataFrame) -> pd.Series:
        return internal_vol63_q90(df) & external_market_vol63_q90(df)

    def spread21_q10(df: pd.DataFrame) -> pd.Series:
        return df["prev_mom_trailing21"] < thresholds["mom_trailing21_q10"]

    def spread63_q10(df: pd.DataFrame) -> pd.Series:
        return df["prev_mom_trailing63"] < thresholds["mom_trailing63_q10"]

    def spread_vol63_q90(df: pd.DataFrame) -> pd.Series:
        return df["prev_mom_vol63"] > thresholds["mom_vol63_q90"]

    def combo_vol_and_or_spread21(df: pd.DataFrame) -> pd.Series:
        return combo_vol_and(df) | spread21_q10(df)

    def combo_vol_and_or_spread63(df: pd.DataFrame) -> pd.Series:
        return combo_vol_and(df) | spread63_q10(df)

    def combo_triple_and(df: pd.DataFrame) -> pd.Series:
        return combo_vol_and(df) & spread21_q10(df)

    return {
        "baseline_no_protection": {
            "type": "baseline",
            "description": "No exposure scaling",
            "signal_fn": lambda df: pd.Series(False, index=df.index),
        },
        "internal_binary_vol63_above_is_q90": {
            "type": "internal_vol",
            "description": "Exit when portfolio vol63 > IS q90",
            "signal_fn": internal_vol63_q90,
        },
        "external_binary_market_vol63_above_is_q90": {
            "type": "external_market_vol",
            "description": "Exit when market vol63 > IS q90",
            "signal_fn": external_market_vol63_q90,
        },
        "combo_vol63_q90_both": {
            "type": "combo_vol",
            "description": "Exit when internal vol63 q90 AND market vol63 q90",
            "signal_fn": combo_vol_and,
        },
        "spread_binary_trailing21_below_is_q10": {
            "type": "momentum_factor_state",
            "description": "Exit when 21d momentum factor trailing return < IS q10",
            "signal_fn": spread21_q10,
        },
        "spread_binary_trailing63_below_is_q10": {
            "type": "momentum_factor_state",
            "description": "Exit when 63d momentum factor trailing return < IS q10",
            "signal_fn": spread63_q10,
        },
        "spread_binary_vol63_above_is_q90": {
            "type": "momentum_factor_state",
            "description": "Exit when momentum factor vol63 > IS q90",
            "signal_fn": spread_vol63_q90,
        },
        "combo_vol_and_or_spread21_q10": {
            "type": "combo_vol_spread",
            "description": "Exit when vol AND rule fires OR spread21 collapses below IS q10",
            "signal_fn": combo_vol_and_or_spread21,
        },
        "combo_vol_and_or_spread63_q10": {
            "type": "combo_vol_spread",
            "description": "Exit when vol AND rule fires OR spread63 collapses below IS q10",
            "signal_fn": combo_vol_and_or_spread63,
        },
        "combo_triple_and_vol_and_spread21": {
            "type": "combo_vol_spread",
            "description": "Exit only when internal vol, market vol, and spread21 all confirm",
            "signal_fn": combo_triple_and,
        },
    }


def apply_variant(
    df: pd.DataFrame,
    variant_name: str,
    variant: dict,
    tc_bps: float = TC_BPS,
) -> pd.DataFrame:
    out = df.copy()

    risk_off = variant["signal_fn"](out).fillna(False).astype(bool)
    exposure = (~risk_off).astype(float)

    exposure_change = exposure.diff().abs().fillna(0.0)
    tc = exposure_change * (tc_bps / 10_000.0)

    out["variant"] = variant_name
    out["variant_type"] = variant["type"]
    out["risk_off"] = risk_off
    out["exposure"] = exposure
    out["exposure_change"] = exposure_change
    out["tc"] = tc
    out["scaled_total_r"] = out["strategy_total_r"] * exposure - tc
    out["scaled_excess_r"] = out["strategy_excess_r"] * exposure - tc

    return out


def summarize_variant(scaled: pd.DataFrame, variant_name: str, variant: dict, sample: str) -> dict:
    total_r = scaled["scaled_total_r"].dropna()
    excess_r = scaled["scaled_excess_r"].dropna()

    return {
        "sample": sample,
        "variant": variant_name,
        "variant_type": variant["type"],
        "description": variant["description"],
        "n_days": int(len(total_r)),
        "final_nav": float((1.0 + total_r).prod()),
        "ann_return_total": annualized_return(total_r),
        "ann_vol_total": annualized_vol(total_r),
        "sharpe_total": annualized_sharpe(total_r),
        "ann_return_excess": annualized_return(excess_r),
        "ann_vol_excess": annualized_vol(excess_r),
        "ir_excess": annualized_sharpe(excess_r),
        "max_dd": max_drawdown(total_r),
        "mean_exposure": float(scaled["exposure"].mean()),
        "risk_off_share": float(scaled["risk_off"].mean()),
        "switches": int(scaled["exposure_change"].sum()),
        "tc_drag": float(scaled["tc"].sum()),
        "mean_daily_total_r": float(total_r.mean()),
        "std_daily_total_r": float(total_r.std(ddof=1)),
        "min_daily_total_r": float(total_r.min()),
        "max_daily_total_r": float(total_r.max()),
    }


def summarize_by_year(scaled: pd.DataFrame, variant_name: str, variant: dict, sample: str) -> list[dict]:
    rows = []

    for year, g in scaled.groupby(scaled.index.year):
        total_r = g["scaled_total_r"].dropna()
        excess_r = g["scaled_excess_r"].dropna()

        if len(total_r) == 0:
            continue

        rows.append(
            {
                "sample": sample,
                "year": int(year),
                "variant": variant_name,
                "variant_type": variant["type"],
                "final_nav": float((1.0 + total_r).prod()),
                "ann_return_total": annualized_return(total_r),
                "ann_vol_total": annualized_vol(total_r),
                "sharpe_total": annualized_sharpe(total_r),
                "ir_excess": annualized_sharpe(excess_r),
                "max_dd": max_drawdown(total_r),
                "mean_exposure": float(g["exposure"].mean()),
                "risk_off_share": float(g["risk_off"].mean()),
                "switches": int(g["exposure_change"].sum()),
                "tc_drag": float(g["tc"].sum()),
            }
        )

    return rows


def monthly_returns(df: pd.DataFrame, value_col: str) -> pd.Series:
    return (1.0 + df[value_col]).resample("ME").prod() - 1.0


def first_fire_date(df: pd.DataFrame, signal_fn: Callable[[pd.DataFrame], pd.Series]) -> str:
    s = signal_fn(df).fillna(False)
    if not s.any():
        return "did_not_fire"
    return str(s[s].index[0].date())


def lead_time_analysis(
    is_df: pd.DataFrame,
    oos_df: pd.DataFrame,
    variants: dict[str, dict],
) -> pd.DataFrame:
    episodes = {
        "IS_2016_Q1": (is_df, "2016-01-01", "2016-06-30"),
        "OOS_2020_COVID": (oos_df, "2020-02-01", "2020-06-30"),
        "OOS_2022_rates": (oos_df, "2022-01-01", "2022-12-31"),
    }

    selected = [
        "internal_binary_vol63_above_is_q90",
        "external_binary_market_vol63_above_is_q90",
        "combo_vol63_q90_both",
        "spread_binary_trailing21_below_is_q10",
        "spread_binary_trailing63_below_is_q10",
        "spread_binary_vol63_above_is_q90",
        "combo_vol_and_or_spread21_q10",
        "combo_vol_and_or_spread63_q10",
    ]

    rows = []

    for episode_name, (df, start, end) in episodes.items():
        sub = df.loc[start:end].copy()

        fire_dates = {}

        for variant_name in selected:
            fire_dates[variant_name] = first_fire_date(sub, variants[variant_name]["signal_fn"])

        baseline_date = fire_dates["combo_vol63_q90_both"]

        for variant_name, fire_date in fire_dates.items():
            lead_vs_combo_days = np.nan

            if baseline_date != "did_not_fire" and fire_date != "did_not_fire":
                lead_vs_combo_days = (
                    pd.Timestamp(baseline_date) - pd.Timestamp(fire_date)
                ).days

            rows.append(
                {
                    "episode": episode_name,
                    "start": start,
                    "end": end,
                    "variant": variant_name,
                    "first_fire_date": fire_date,
                    "combo_vol_and_first_fire_date": baseline_date,
                    "lead_vs_combo_vol_and_days": lead_vs_combo_days,
                }
            )

    return pd.DataFrame(rows)


def write_research_log(
    path: Path,
    thresholds: dict[str, float],
    oos_results: pd.DataFrame,
    lead_df: pd.DataFrame,
) -> None:
    best = oos_results.sort_values("ir_excess", ascending=False).iloc[0]
    combo = oos_results[oos_results["variant"] == "combo_vol63_q90_both"].iloc[0]
    spread21 = oos_results[oos_results["variant"] == "spread_binary_trailing21_below_is_q10"].iloc[0]
    combo_spread21 = oos_results[oos_results["variant"] == "combo_vol_and_or_spread21_q10"].iloc[0]

    lines = []

    lines.append("Regime-aware analysis v4: Cross-sectional momentum factor state signal")
    lines.append("")
    lines.append("Hypothesis:")
    lines.append(
        "A cross-sectional momentum factor state signal can provide earlier warning "
        "of momentum crashes than volatility-based rules."
    )
    lines.append("")
    lines.append("Data source:")
    lines.append(
        "momentum_factor_r.parquet saved from runner.data['momentum'], aligned to strategy_total_r dates."
    )
    lines.append("")
    lines.append("Thresholds calibrated on IS WF 2015-2019 and frozen before OOS:")
    for k, v in thresholds.items():
        lines.append(f"- {k}: {v:.6f} ({v:.4%})")

    lines.append("")
    lines.append("OOS best by IR:")
    lines.append(
        f"- {best['variant']}: NAV={best['final_nav']:.3f}, "
        f"Sharpe={best['sharpe_total']:.3f}, IR={best['ir_excess']:.3f}, "
        f"MaxDD={best['max_dd']:.3%}, exposure={best['mean_exposure']:.2%}, "
        f"switches={int(best['switches'])}"
    )

    lines.append("")
    lines.append("Reference v3 winner:")
    lines.append(
        f"- combo_vol63_q90_both: NAV={combo['final_nav']:.3f}, "
        f"Sharpe={combo['sharpe_total']:.3f}, IR={combo['ir_excess']:.3f}, "
        f"MaxDD={combo['max_dd']:.3%}, exposure={combo['mean_exposure']:.2%}, "
        f"switches={int(combo['switches'])}"
    )

    lines.append("")
    lines.append("Spread-only 21d q10:")
    lines.append(
        f"- spread_binary_trailing21_below_is_q10: NAV={spread21['final_nav']:.3f}, "
        f"Sharpe={spread21['sharpe_total']:.3f}, IR={spread21['ir_excess']:.3f}, "
        f"MaxDD={spread21['max_dd']:.3%}, exposure={spread21['mean_exposure']:.2%}, "
        f"switches={int(spread21['switches'])}"
    )

    lines.append("")
    lines.append("Vol AND OR spread21:")
    lines.append(
        f"- combo_vol_and_or_spread21_q10: NAV={combo_spread21['final_nav']:.3f}, "
        f"Sharpe={combo_spread21['sharpe_total']:.3f}, IR={combo_spread21['ir_excess']:.3f}, "
        f"MaxDD={combo_spread21['max_dd']:.3%}, exposure={combo_spread21['mean_exposure']:.2%}, "
        f"switches={int(combo_spread21['switches'])}"
    )

    lines.append("")
    lines.append("Lead-time table:")
    lines.append(lead_df.to_string(index=False))

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = repo_root()

    is_root = root / IS_ROOT
    oos_root = root / OOS_ROOT
    out_dir = root / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    is_runs = read_runs(is_root, ENS3_CONFIGS)
    oos_runs = read_runs(oos_root, ENS3_CONFIGS, eval_window="full_oos_2020_2023")

    is_df = build_ensemble(is_runs, ENS3_CONFIGS)
    oos_df = build_ensemble(oos_runs, ENS3_CONFIGS)

    is_df = add_signals(is_df)
    oos_df = add_signals(oos_df)

    thresholds = calibrate_thresholds(is_df)
    variants = build_variants(thresholds)

    pd.DataFrame(
        [{"threshold": k, "value": v} for k, v in thresholds.items()]
    ).to_csv(out_dir / "is_calibrated_thresholds_v4.csv", index=False)

    is_df.to_csv(out_dir / "is_wf_ens3_daily_returns_v4.csv")
    oos_df.to_csv(out_dir / "oos_ens3_daily_returns_v4.csv")

    is_monthly = pd.DataFrame(
        {
            "strategy_total_r": monthly_returns(is_df, "strategy_total_r"),
            "strategy_excess_r": monthly_returns(is_df, "strategy_excess_r"),
            "market_total_r": monthly_returns(is_df, "market_total_r"),
            "momentum_factor_r": monthly_returns(is_df, "momentum_factor_r"),
        }
    )
    oos_monthly = pd.DataFrame(
        {
            "strategy_total_r": monthly_returns(oos_df, "strategy_total_r"),
            "strategy_excess_r": monthly_returns(oos_df, "strategy_excess_r"),
            "market_total_r": monthly_returns(oos_df, "market_total_r"),
            "momentum_factor_r": monthly_returns(oos_df, "momentum_factor_r"),
        }
    )

    is_monthly.to_csv(out_dir / "is_wf_ens3_monthly_returns_v4.csv")
    oos_monthly.to_csv(out_dir / "oos_ens3_monthly_returns_v4.csv")

    is_result_rows = []
    oos_result_rows = []
    is_year_rows = []
    oos_year_rows = []
    scaled_daily_frames = []

    for variant_name, variant in variants.items():
        is_scaled = apply_variant(is_df, variant_name, variant)
        oos_scaled = apply_variant(oos_df, variant_name, variant)

        is_result_rows.append(summarize_variant(is_scaled, variant_name, variant, "is_wf_2015_2019"))
        oos_result_rows.append(summarize_variant(oos_scaled, variant_name, variant, "oos_2020_2023"))

        is_year_rows.extend(summarize_by_year(is_scaled, variant_name, variant, "is_wf_2015_2019"))
        oos_year_rows.extend(summarize_by_year(oos_scaled, variant_name, variant, "oos_2020_2023"))

        tmp = oos_scaled[
            [
                "variant",
                "variant_type",
                "risk_off",
                "exposure",
                "exposure_change",
                "tc",
                "scaled_total_r",
                "scaled_excess_r",
            ]
        ].copy()
        scaled_daily_frames.append(tmp)

    is_results = pd.DataFrame(is_result_rows).sort_values("ir_excess", ascending=False)
    oos_results = pd.DataFrame(oos_result_rows).sort_values("ir_excess", ascending=False)

    is_by_year = pd.DataFrame(is_year_rows)
    oos_by_year = pd.DataFrame(oos_year_rows)

    is_results.to_csv(out_dir / "is_exposure_scaling_results_v4.csv", index=False)
    oos_results.to_csv(out_dir / "oos_exposure_scaling_results_v4.csv", index=False)

    is_by_year.to_csv(out_dir / "is_exposure_scaling_by_year_v4.csv", index=False)
    oos_by_year.to_csv(out_dir / "oos_exposure_scaling_by_year_v4.csv", index=False)

    oos_scaled_daily = pd.concat(scaled_daily_frames).sort_index()
    oos_scaled_daily.to_csv(out_dir / "oos_exposure_scaled_daily_returns_v4.csv")

    benchmark = oos_df[["market_total_r", "momentum_factor_r"]].copy()
    benchmark.to_csv(out_dir / "benchmark_and_momentum_factor_daily_returns_v4.csv")

    lead_df = lead_time_analysis(is_df, oos_df, variants)
    lead_df.to_csv(out_dir / "signal_lead_time_analysis_v4.csv", index=False)

    write_research_log(
        out_dir / "research_log_regime_v4.txt",
        thresholds,
        oos_results,
        lead_df,
    )

    print("Saved results to:", out_dir)
    print()
    print("OOS results:")
    cols = [
        "variant",
        "variant_type",
        "final_nav",
        "sharpe_total",
        "ir_excess",
        "max_dd",
        "mean_exposure",
        "switches",
        "tc_drag",
    ]
    print(oos_results[cols].to_string(index=False))

    print()
    print("Lead-time analysis:")
    print(lead_df.to_string(index=False))


if __name__ == "__main__":
    main()