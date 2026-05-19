from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ENS3_CONFIGS = [
    "finalist_1_q020_ex84_win126",
    "finalist_2_q030_ex84_win126",
    "finalist_3_q030_ex63_win126",
]


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Cannot locate repo root")


def _read_config(run_dir: Path) -> dict[str, Any]:
    obj = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    return obj["params"] if "params" in obj else obj


def _load_runs(results_dir: Path, eval_windows: set[str] | None = None) -> pd.DataFrame:
    runs_dir = results_dir / "runs"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs dir not found: {runs_dir}")

    rows = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        config_path = run_dir / "config.json"
        total_path = run_dir / "strategy_total_r.parquet"
        excess_path = run_dir / "strategy_excess_r.parquet"

        if not config_path.exists() or not total_path.exists() or not excess_path.exists():
            continue

        params = _read_config(run_dir)
        eval_window = params.get("eval_window")

        if eval_windows is not None and eval_window not in eval_windows:
            continue

        rows.append(
            {
                "run_dir": run_dir,
                "run_id": run_dir.name,
                "config_name": params.get("config_name"),
                "config_type": params.get("config_type"),
                "eval_window": eval_window,
                "fold": params.get("fold"),
                "test_start": params.get("test_start"),
                "test_end": params.get("test_end"),
                "quantile": params.get("quantile"),
                "exclude_last_days": params.get("exclude_last_days"),
                "window_days": params.get("window_days"),
                "total_path": total_path,
                "excess_path": excess_path,
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        raise RuntimeError(f"No runs found in {runs_dir}")

    return out


def _read_return_series(path: Path, expected_col: str) -> pd.Series:
    df = pd.read_parquet(path)

    if expected_col in df.columns:
        s = df[expected_col].copy()
    elif df.shape[1] == 1:
        s = df.iloc[:, 0].copy()
    else:
        raise RuntimeError(f"Cannot find {expected_col} in {path}; columns={df.columns.tolist()}")

    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    s.name = expected_col
    return s


def _build_ens3_daily(runs: pd.DataFrame) -> pd.DataFrame:
    ens_runs = runs[runs["config_name"].isin(ENS3_CONFIGS)].copy()

    missing = sorted(set(ENS3_CONFIGS) - set(ens_runs["config_name"]))
    if missing:
        raise RuntimeError(f"Missing ens-3 configs: {missing}")

    total_parts = []
    excess_parts = []

    for _, row in ens_runs.iterrows():
        total = _read_return_series(Path(row["total_path"]), "total_r")
        excess = _read_return_series(Path(row["excess_path"]), "excess_r")

        config_name = str(row["config_name"])
        eval_window = str(row["eval_window"])
        unique_name = f"{config_name}__{eval_window}"

        total.name = unique_name
        excess.name = unique_name

        total_parts.append(total)
        excess_parts.append(excess)

    total_df = pd.concat(total_parts, axis=1, sort=True).sort_index()
    excess_df = pd.concat(excess_parts, axis=1, sort=True).sort_index()

    max_total_non_na = int(total_df.notna().sum(axis=1).max())
    max_excess_non_na = int(excess_df.notna().sum(axis=1).max())

    if max_total_non_na > len(ENS3_CONFIGS) or max_excess_non_na > len(ENS3_CONFIGS):
        raise RuntimeError(
            "Date overlap between folds detected: "
            f"max_total_non_na={max_total_non_na}, "
            f"max_excess_non_na={max_excess_non_na}, "
            f"expected<={len(ENS3_CONFIGS)}"
        )

    out = pd.DataFrame(index=total_df.index.union(excess_df.index).sort_values())
    out["ens3_total_r"] = total_df.mean(axis=1, skipna=True)
    out["ens3_excess_r"] = excess_df.mean(axis=1, skipna=True)

    for col in total_df.columns:
        out[f"total_r__{col}"] = total_df[col]

    for col in excess_df.columns:
        out[f"excess_r__{col}"] = excess_df[col]

    out = out.dropna(subset=["ens3_total_r", "ens3_excess_r"], how="all")
    out.index.name = "date"

    return out


def _compound_returns(s: pd.Series) -> float:
    s = s.dropna()
    if s.empty:
        return np.nan
    return float((1.0 + s).prod() - 1.0)


def _to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    monthly = pd.DataFrame(index=daily.resample("ME").last().index)

    monthly["ens3_total_r"] = daily["ens3_total_r"].resample("ME").apply(_compound_returns)
    monthly["ens3_excess_r"] = daily["ens3_excess_r"].resample("ME").apply(_compound_returns)

    for col in daily.columns:
        if col.startswith("total_r__") or col.startswith("excess_r__"):
            monthly[col] = daily[col].resample("ME").apply(_compound_returns)

    monthly = monthly.dropna(subset=["ens3_total_r", "ens3_excess_r"], how="all")
    monthly.index.name = "date"
    monthly["year"] = monthly.index.year
    monthly["month"] = monthly.index.month

    return monthly


def _max_drawdown(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan

    nav = (1.0 + returns).cumprod()
    dd = nav / nav.cummax() - 1.0

    return float(dd.min())


def _annualized_sharpe(returns: pd.Series, periods: int = 252) -> float:
    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan

    std = returns.std(ddof=1)
    if std == 0 or pd.isna(std):
        return np.nan

    return float(np.sqrt(periods) * returns.mean() / std)


def _annualized_ir(excess_returns: pd.Series, periods: int = 252) -> float:
    return _annualized_sharpe(excess_returns, periods=periods)


def _perf_stats(total_r: pd.Series, excess_r: pd.Series) -> dict[str, float]:
    total_r = total_r.dropna()
    excess_r = excess_r.dropna()

    common_idx = total_r.index.intersection(excess_r.index)
    total_r = total_r.loc[common_idx]
    excess_r = excess_r.loc[common_idx]

    final_nav = float((1.0 + total_r).prod()) if not total_r.empty else np.nan

    return {
        "final_nav": final_nav,
        "total_return": final_nav - 1.0 if pd.notna(final_nav) else np.nan,
        "sharpe_total": _annualized_sharpe(total_r),
        "ir_excess": _annualized_ir(excess_r),
        "max_dd_total": _max_drawdown(total_r),
        "mean_daily_total_r": float(total_r.mean()) if not total_r.empty else np.nan,
        "mean_daily_excess_r": float(excess_r.mean()) if not excess_r.empty else np.nan,
        "vol_daily_total_r": float(total_r.std(ddof=1)) if len(total_r) > 1 else np.nan,
        "vol_daily_excess_r": float(excess_r.std(ddof=1)) if len(excess_r) > 1 else np.nan,
    }


def _build_daily_signals(daily: pd.DataFrame) -> pd.DataFrame:
    out = daily.copy()

    r = out["ens3_total_r"].fillna(0.0)
    xr = out["ens3_excess_r"].fillna(0.0)

    out["nav"] = (1.0 + r).cumprod()
    out["drawdown"] = out["nav"] / out["nav"].cummax() - 1.0

    out["realized_vol_21"] = r.rolling(21).std() * np.sqrt(252)
    out["realized_vol_63"] = r.rolling(63).std() * np.sqrt(252)

    out["trailing_total_r_21"] = (1.0 + r).rolling(21).apply(np.prod, raw=True) - 1.0
    out["trailing_total_r_63"] = (1.0 + r).rolling(63).apply(np.prod, raw=True) - 1.0

    out["trailing_excess_r_21"] = (1.0 + xr).rolling(21).apply(np.prod, raw=True) - 1.0
    out["trailing_excess_r_63"] = (1.0 + xr).rolling(63).apply(np.prod, raw=True) - 1.0

    out["prev_day_total_r"] = out["ens3_total_r"].shift(1)
    out["prev_day_excess_r"] = out["ens3_excess_r"].shift(1)
    out["prev_drawdown"] = out["drawdown"].shift(1)
    out["prev_realized_vol_21"] = out["realized_vol_21"].shift(1)
    out["prev_realized_vol_63"] = out["realized_vol_63"].shift(1)
    out["prev_trailing_total_r_21"] = out["trailing_total_r_21"].shift(1)
    out["prev_trailing_total_r_63"] = out["trailing_total_r_63"].shift(1)
    out["prev_trailing_excess_r_21"] = out["trailing_excess_r_21"].shift(1)
    out["prev_trailing_excess_r_63"] = out["trailing_excess_r_63"].shift(1)

    return out


def _calibrate_thresholds_on_is(is_signals: pd.DataFrame) -> dict[str, float]:
    vol21 = is_signals["prev_realized_vol_21"].dropna()
    vol63 = is_signals["prev_realized_vol_63"].dropna()

    trailing_total_21 = is_signals["prev_trailing_total_r_21"].dropna()
    trailing_total_63 = is_signals["prev_trailing_total_r_63"].dropna()

    daily_total = is_signals["prev_day_total_r"].dropna()

    thresholds = {
        "median_vol_21": float(vol21.median()),
        "median_vol_63": float(vol63.median()),
        "q75_vol_21": float(vol21.quantile(0.75)),
        "q75_vol_63": float(vol63.quantile(0.75)),
        "q90_vol_21": float(vol21.quantile(0.90)),
        "q90_vol_63": float(vol63.quantile(0.90)),
        "q25_trailing_total_r_21": float(trailing_total_21.quantile(0.25)),
        "q25_trailing_total_r_63": float(trailing_total_63.quantile(0.25)),
        "q05_daily_total_r": float(daily_total.quantile(0.05)),
        "q01_daily_total_r": float(daily_total.quantile(0.01)),
    }

    return thresholds


def _apply_cooldown(signal_on: pd.Series, cooldown_days: int) -> pd.Series:
    signal_on = signal_on.fillna(False).astype(bool)
    exposure = pd.Series(1.0, index=signal_on.index)

    cooldown = 0

    for date in signal_on.index:
        if signal_on.loc[date]:
            cooldown = cooldown_days

        if cooldown > 0:
            exposure.loc[date] = 0.0
            cooldown -= 1
        else:
            exposure.loc[date] = 1.0

    return exposure


def _build_exposure_variants(
    oos_signals: pd.DataFrame,
    thresholds: dict[str, float],
) -> dict[str, pd.Series]:
    idx = oos_signals.index
    exposures: dict[str, pd.Series] = {}

    exposures["baseline_no_protection"] = pd.Series(1.0, index=idx)

    vol21 = oos_signals["prev_realized_vol_21"]
    vol63 = oos_signals["prev_realized_vol_63"]

    exposures["vol_inverse_21_is_median"] = (
        thresholds["median_vol_21"] / vol21
    ).clip(0.0, 1.0).fillna(1.0)

    exposures["vol_inverse_63_is_median"] = (
        thresholds["median_vol_63"] / vol63
    ).clip(0.0, 1.0).fillna(1.0)

    exposures["binary_vol21_above_is_q75"] = pd.Series(
        np.where(vol21 > thresholds["q75_vol_21"], 0.0, 1.0),
        index=idx,
    ).fillna(1.0)

    exposures["binary_vol63_above_is_q75"] = pd.Series(
        np.where(vol63 > thresholds["q75_vol_63"], 0.0, 1.0),
        index=idx,
    ).fillna(1.0)

    exposures["binary_vol21_above_is_q90"] = pd.Series(
        np.where(vol21 > thresholds["q90_vol_21"], 0.0, 1.0),
        index=idx,
    ).fillna(1.0)

    exposures["binary_vol63_above_is_q90"] = pd.Series(
        np.where(vol63 > thresholds["q90_vol_63"], 0.0, 1.0),
        index=idx,
    ).fillna(1.0)

    exposures["momentum_21_below_is_q25"] = pd.Series(
        np.where(
            oos_signals["prev_trailing_total_r_21"] < thresholds["q25_trailing_total_r_21"],
            0.0,
            1.0,
        ),
        index=idx,
    ).fillna(1.0)

    exposures["momentum_63_below_is_q25"] = pd.Series(
        np.where(
            oos_signals["prev_trailing_total_r_63"] < thresholds["q25_trailing_total_r_63"],
            0.0,
            1.0,
        ),
        index=idx,
    ).fillna(1.0)

    exposures["drawdown_below_fixed_10pct"] = pd.Series(
        np.where(oos_signals["prev_drawdown"] < -0.10, 0.0, 1.0),
        index=idx,
    ).fillna(1.0)

    exposures["cooldown_21d_after_daily_loss_is_q05"] = _apply_cooldown(
        oos_signals["prev_day_total_r"] < thresholds["q05_daily_total_r"],
        cooldown_days=21,
    )

    exposures["cooldown_42d_after_daily_loss_is_q05"] = _apply_cooldown(
        oos_signals["prev_day_total_r"] < thresholds["q05_daily_total_r"],
        cooldown_days=42,
    )

    exposures["cooldown_21d_after_daily_loss_is_q01"] = _apply_cooldown(
        oos_signals["prev_day_total_r"] < thresholds["q01_daily_total_r"],
        cooldown_days=21,
    )

    exposures["cooldown_42d_after_daily_loss_is_q01"] = _apply_cooldown(
        oos_signals["prev_day_total_r"] < thresholds["q01_daily_total_r"],
        cooldown_days=42,
    )

    return exposures


def _apply_transaction_costs(
    scaled_total: pd.Series,
    scaled_excess: pd.Series,
    exposure: pd.Series,
    tc_bps: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    exposure = exposure.reindex(scaled_total.index).fillna(1.0)

    exposure_change = exposure.diff().abs().fillna(0.0)
    tc = exposure_change * (tc_bps / 10_000.0)

    total_after_tc = scaled_total - tc
    excess_after_tc = scaled_excess - tc

    return total_after_tc, excess_after_tc, tc


def _evaluate_exposures(
    signals: pd.DataFrame,
    exposures: dict[str, pd.Series],
    tc_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    daily_scaled_parts = []

    base_total = signals["ens3_total_r"].copy()
    base_excess = signals["ens3_excess_r"].copy()

    baseline_stats = _perf_stats(base_total, base_excess)
    baseline_ir = baseline_stats["ir_excess"]

    for name, exposure in exposures.items():
        exposure = exposure.reindex(signals.index).fillna(1.0)

        scaled_total_raw = base_total * exposure
        scaled_excess_raw = base_excess * exposure

        scaled_total, scaled_excess, tc = _apply_transaction_costs(
            scaled_total=scaled_total_raw,
            scaled_excess=scaled_excess_raw,
            exposure=exposure,
            tc_bps=tc_bps,
        )

        stats = _perf_stats(scaled_total, scaled_excess)

        exposure_change = exposure.diff().abs().fillna(0.0)
        switch_count = int((exposure_change > 0).sum())
        total_tc = float(tc.sum())

        avg_exposure = float(exposure.mean())
        days_off = int((exposure == 0.0).sum())
        days_reduced = int((exposure < 1.0).sum())

        ir_loss = np.nan
        if pd.notna(baseline_ir) and baseline_ir != 0:
            ir_loss = float((baseline_ir - stats["ir_excess"]) / abs(baseline_ir))

        rows.append(
            {
                "variant": name,
                **stats,
                "avg_exposure": avg_exposure,
                "days_off": days_off,
                "days_reduced": days_reduced,
                "switch_count": switch_count,
                "tc_bps": tc_bps,
                "total_tc_return_drag": total_tc,
                "ir_loss_vs_baseline_pct": ir_loss * 100 if pd.notna(ir_loss) else np.nan,
                "signal_type": "none" if name == "baseline_no_protection" else "internal_self_referential",
                "tc_model": "abs_delta_exposure_first_order",
            }
        )

        tmp = pd.DataFrame(
            {
                "date": signals.index,
                "variant": name,
                "exposure": exposure.values,
                "tc": tc.values,
                "scaled_total_r": scaled_total.values,
                "scaled_excess_r": scaled_excess.values,
                "scaled_nav": (1.0 + scaled_total.fillna(0.0)).cumprod().values,
            }
        )
        daily_scaled_parts.append(tmp)

    results = pd.DataFrame(rows).sort_values(
        ["max_dd_total", "ir_excess", "final_nav"],
        ascending=[False, False, False],
    )

    daily_scaled = pd.concat(daily_scaled_parts, ignore_index=True)

    return results, daily_scaled


def _evaluate_by_year(daily_scaled: pd.DataFrame) -> pd.DataFrame:
    df = daily_scaled.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year

    rows = []

    for (variant, year), g in df.groupby(["variant", "year"]):
        g = g.sort_values("date")

        total = pd.Series(g["scaled_total_r"].values, index=g["date"])
        excess = pd.Series(g["scaled_excess_r"].values, index=g["date"])

        stats = _perf_stats(total, excess)

        rows.append(
            {
                "variant": variant,
                "year": year,
                **stats,
                "avg_exposure": float(g["exposure"].mean()),
                "switch_count": int((g["exposure"].diff().abs().fillna(0.0) > 0).sum()),
                "tc_return_drag": float(g["tc"].sum()),
            }
        )

    return pd.DataFrame(rows).sort_values(["year", "ir_excess"], ascending=[True, False])


def _make_monthly_crash_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    m = monthly.copy()

    m["crash_month_total_lt_5pct"] = m["ens3_total_r"] < -0.05
    m["crash_month_excess_lt_5pct"] = m["ens3_excess_r"] < -0.05

    rows = []

    for crash_col in ["crash_month_total_lt_5pct", "crash_month_excess_lt_5pct"]:
        g = m[m[crash_col]].copy()

        rows.append(
            {
                "label": crash_col,
                "n_months": int(g.shape[0]),
                "months": ", ".join(d.strftime("%Y-%m") for d in g.index),
                "mean_total_r": float(g["ens3_total_r"].mean()) if not g.empty else np.nan,
                "mean_excess_r": float(g["ens3_excess_r"].mean()) if not g.empty else np.nan,
                "min_total_r": float(g["ens3_total_r"].min()) if not g.empty else np.nan,
                "min_excess_r": float(g["ens3_excess_r"].min()) if not g.empty else np.nan,
            }
        )

    return pd.DataFrame(rows)


def _thresholds_to_df(thresholds: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"threshold_name": k, "threshold_value": v} for k, v in thresholds.items()]
    ).sort_values("threshold_name")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--is-results-subdir", default="data/results_is_wf_with_returns")
    parser.add_argument("--oos-results-subdir", default="data/results_oos_with_returns")
    parser.add_argument("--out-subdir", default="data/results_regime_v2")
    parser.add_argument("--tc-bps", type=float, default=25.0)
    args = parser.parse_args()

    root = _repo_root()

    is_dir = root / args.is_results_subdir
    oos_dir = root / args.oos_results_subdir
    out_dir = root / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    is_windows = {"wf_2015", "wf_2016", "wf_2017", "wf_2018", "wf_2019"}
    oos_windows = {"full_oos_2020_2023"}

    is_runs = _load_runs(is_dir, eval_windows=is_windows)
    oos_runs = _load_runs(oos_dir, eval_windows=oos_windows)

    is_ens3_daily = _build_ens3_daily(is_runs)
    oos_ens3_daily = _build_ens3_daily(oos_runs)

    is_ens3_monthly = _to_monthly(is_ens3_daily)
    oos_ens3_monthly = _to_monthly(oos_ens3_daily)

    is_signals = _build_daily_signals(is_ens3_daily)
    oos_signals = _build_daily_signals(oos_ens3_daily)

    thresholds = _calibrate_thresholds_on_is(is_signals)
    exposures = _build_exposure_variants(oos_signals, thresholds)

    exposure_results, daily_scaled = _evaluate_exposures(
        signals=oos_signals,
        exposures=exposures,
        tc_bps=args.tc_bps,
    )
    yearly_results = _evaluate_by_year(daily_scaled)

    is_crash_summary = _make_monthly_crash_summary(is_ens3_monthly)
    oos_crash_summary = _make_monthly_crash_summary(oos_ens3_monthly)

    thresholds_df = _thresholds_to_df(thresholds)

    is_ens3_daily.to_csv(out_dir / "is_wf_ens3_daily_returns.csv")
    oos_ens3_daily.to_csv(out_dir / "oos_ens3_daily_returns.csv")

    is_ens3_monthly.to_csv(out_dir / "is_wf_ens3_monthly_returns.csv")
    oos_ens3_monthly.to_csv(out_dir / "oos_ens3_monthly_returns.csv")

    is_signals.to_csv(out_dir / "is_wf_daily_signals.csv")
    oos_signals.to_csv(out_dir / "oos_daily_signals.csv")

    thresholds_df.to_csv(out_dir / "is_calibrated_thresholds.csv", index=False)

    exposure_results.to_csv(out_dir / "oos_exposure_scaling_results.csv", index=False)
    daily_scaled.to_csv(out_dir / "oos_exposure_scaled_daily_returns.csv", index=False)
    yearly_results.to_csv(out_dir / "oos_exposure_scaling_by_year.csv", index=False)

    is_crash_summary.to_csv(out_dir / "is_wf_crash_month_summary.csv", index=False)
    oos_crash_summary.to_csv(out_dir / "oos_crash_month_summary.csv", index=False)

    research_log = f"""Regime-Aware Enhanced Momentum — Research Log

Methodology:
- IS calibration data: concatenated walk-forward test returns from 2015–2019.
- Each IS calibration year is a test fold, not a fitted full-sample in-sample return series.
- OOS evaluation data: held-out 2020–2023 full OOS return stream.
- Ensemble: ens-3 implemented as equal-weight allocation across three strategy return streams:
  finalist_1_q020_ex84_win126, finalist_2_q030_ex84_win126, finalist_3_q030_ex63_win126.
- Thresholds are calibrated only on IS WF returns and frozen before OOS evaluation.
- Exposure decisions use lagged signals only: t-1 signal -> t exposure.
- Current signals are internal/self-referential, based on ens-3 return stream.
- Daily-loss cooldown thresholds are calibrated on IS WF returns using q05/q01 daily total return.
- Drawdown threshold -10% is a pre-specified risk-management rule, not an optimized threshold.
- Transaction cost model: abs(delta exposure) * {args.tc_bps} bps.
- Transaction costs are a conservative first-order approximation, especially for continuous scaling variants.
- Cooldown design: if another crash event occurs during cooldown, cooldown counter is reset to full cooldown.
- as_zscore is fixed to False because prior analysis showed it is a no-op in the current strategy implementation.

Caveats:
- Ens-3 is return-stream ensemble, not position-level merged portfolio.
- Internal signals are primarily damage-control, not external early-warning signals.
- Calibration contains only one clear IS crash year (2016), so results are exploratory.
"""
    (out_dir / "research_log_regime_v2.txt").write_text(research_log, encoding="utf-8")

    print("Saved outputs to:", out_dir)

    print("\nThresholds calibrated on IS WF:")
    print(thresholds_df.to_string(index=False))

    print("\nIS crash summary:")
    print(is_crash_summary.to_string(index=False))

    print("\nOOS crash summary:")
    print(oos_crash_summary.to_string(index=False))

    print("\nOOS exposure scaling results:")
    cols = [
        "variant",
        "signal_type",
        "final_nav",
        "total_return",
        "sharpe_total",
        "ir_excess",
        "max_dd_total",
        "avg_exposure",
        "days_off",
        "days_reduced",
        "switch_count",
        "total_tc_return_drag",
        "ir_loss_vs_baseline_pct",
    ]
    cols = [c for c in cols if c in exposure_results.columns]
    print(exposure_results[cols].to_string(index=False))

    print("\nOOS by year:")
    yearly_cols = [
        "variant",
        "year",
        "final_nav",
        "sharpe_total",
        "ir_excess",
        "max_dd_total",
        "avg_exposure",
        "switch_count",
        "tc_return_drag",
    ]
    yearly_cols = [c for c in yearly_cols if c in yearly_results.columns]
    print(yearly_results[yearly_cols].to_string(index=False))


if __name__ == "__main__":
    main()