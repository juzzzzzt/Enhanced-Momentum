from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ENS3_CONFIGS = [
    "finalist_1_q020_ex84_win126",
    "finalist_2_q030_ex84_win126",
    "finalist_3_q030_ex63_win126",
]

IS_ROOT = Path("data/results_is_wf_with_returns/runs")
OOS_ROOT = Path("data/results_oos_with_returns/runs")
OUT_DIR = Path("data/results_structural_breaks")

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


def read_runs(root: Path, eval_window: str | None = None) -> dict[tuple[str, str], pd.DataFrame]:
    out = {}

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

        if cfg["config_name"] not in ENS3_CONFIGS:
            continue

        if eval_window is not None and cfg["eval_window"] != eval_window:
            continue

        total = read_series(total_path, "strategy_total_r")
        excess = read_series(excess_path, "strategy_excess_r")
        market = read_series(market_path, "market_total_r")
        momentum = read_series(momentum_path, "momentum_factor_r")

        df = pd.concat([total, excess, market, momentum], axis=1).dropna()

        if not df.empty:
            out[(cfg["eval_window"], cfg["config_name"])] = df

    if not out:
        raise RuntimeError(f"No runs found in {root}")

    return out


def build_ensemble(runs: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    frames = []
    eval_windows = sorted({k[0] for k in runs.keys()})

    for ew in eval_windows:
        per_config = []

        for cfg_name in ENS3_CONFIGS:
            key = (ew, cfg_name)
            if key not in runs:
                raise RuntimeError(f"Missing run for eval_window={ew}, config={cfg_name}")
            per_config.append(runs[key])

        panel = pd.concat(per_config, axis=1, keys=ENS3_CONFIGS).dropna()

        out = pd.DataFrame(index=panel.index)
        out["strategy_total_r"] = panel.xs("strategy_total_r", axis=1, level=1).mean(axis=1)
        out["strategy_excess_r"] = panel.xs("strategy_excess_r", axis=1, level=1).mean(axis=1)
        out["market_total_r"] = panel.xs("market_total_r", axis=1, level=1).mean(axis=1)
        out["momentum_factor_r"] = panel.xs("momentum_factor_r", axis=1, level=1).mean(axis=1)
        out["eval_window"] = ew

        frames.append(out)

    return pd.concat(frames).sort_index()


def monthly_compound(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "strategy_total_r": (1.0 + df["strategy_total_r"]).resample("ME").prod() - 1.0,
            "strategy_excess_r": (1.0 + df["strategy_excess_r"]).resample("ME").prod() - 1.0,
            "market_total_r": (1.0 + df["market_total_r"]).resample("ME").prod() - 1.0,
            "momentum_factor_r": (1.0 + df["momentum_factor_r"]).resample("ME").prod() - 1.0,
        }
    ).dropna()


def add_monthly_features(m: pd.DataFrame) -> pd.DataFrame:
    out = m.copy()

    out["strategy_vol_3m"] = out["strategy_total_r"].rolling(3).std() * np.sqrt(12)
    out["strategy_vol_6m"] = out["strategy_total_r"].rolling(6).std() * np.sqrt(12)
    out["market_vol_3m"] = out["market_total_r"].rolling(3).std() * np.sqrt(12)
    out["market_vol_6m"] = out["market_total_r"].rolling(6).std() * np.sqrt(12)
    out["momentum_factor_vol_3m"] = out["momentum_factor_r"].rolling(3).std() * np.sqrt(12)
    out["momentum_factor_vol_6m"] = out["momentum_factor_r"].rolling(6).std() * np.sqrt(12)

    out["strategy_trailing_3m"] = (1.0 + out["strategy_total_r"]).rolling(3).apply(np.prod, raw=True) - 1.0
    out["strategy_trailing_6m"] = (1.0 + out["strategy_total_r"]).rolling(6).apply(np.prod, raw=True) - 1.0
    out["momentum_factor_trailing_3m"] = (1.0 + out["momentum_factor_r"]).rolling(3).apply(np.prod, raw=True) - 1.0
    out["momentum_factor_trailing_6m"] = (1.0 + out["momentum_factor_r"]).rolling(6).apply(np.prod, raw=True) - 1.0

    return out.dropna()


def prefix_sums(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    s1 = np.concatenate([[0.0], np.cumsum(y)])
    s2 = np.concatenate([[0.0], np.cumsum(y * y)])
    return s1, s2


def segment_sse(s1: np.ndarray, s2: np.ndarray, start: int, end: int) -> float:
    n = end - start
    if n <= 0:
        return np.inf
    total = s1[end] - s1[start]
    total2 = s2[end] - s2[start]
    return float(total2 - total * total / n)


def estimate_breaks_dp(
    y: pd.Series,
    max_breaks: int,
    min_size: int,
) -> tuple[pd.DataFrame, dict[int, list[int]]]:
    s = y.dropna()
    values = s.values.astype(float)
    n = len(values)

    if n < 2 * min_size:
        raise ValueError(f"Series too short: n={n}, min_size={min_size}")

    s1, s2 = prefix_sums(values)

    cost = np.full((max_breaks + 1, n + 1), np.inf)
    prev = np.full((max_breaks + 1, n + 1), -1, dtype=int)

    for t in range(min_size, n + 1):
        cost[0, t] = segment_sse(s1, s2, 0, t)

    for k in range(1, max_breaks + 1):
        min_t = (k + 1) * min_size
        for t in range(min_t, n + 1):
            best_val = np.inf
            best_j = -1

            j_min = k * min_size
            j_max = t - min_size

            for j in range(j_min, j_max + 1):
                val = cost[k - 1, j] + segment_sse(s1, s2, j, t)
                if val < best_val:
                    best_val = val
                    best_j = j

            cost[k, t] = best_val
            prev[k, t] = best_j

    rows = []
    breakpoints_by_k = {}

    for k in range(0, max_breaks + 1):
        if not np.isfinite(cost[k, n]):
            continue

        breaks = []
        cur_t = n
        cur_k = k

        while cur_k > 0:
            j = int(prev[cur_k, cur_t])
            if j < 0:
                break
            breaks.append(j)
            cur_t = j
            cur_k -= 1

        breaks = sorted(breaks)
        breakpoints_by_k[k] = breaks

        rss = float(cost[k, n])
        num_params = k + 1
        bic = n * np.log(rss / n) + num_params * np.log(n)
        lwz_like = n * np.log(rss / n) + num_params * 0.299 * np.log(n) ** 2.1

        rows.append(
            {
                "n_breaks": k,
                "rss": rss,
                "bic": bic,
                "lwz_like": lwz_like,
                "break_indices": ",".join(map(str, breaks)),
                "break_dates": ",".join(str(s.index[i - 1].date()) for i in breaks),
            }
        )

    return pd.DataFrame(rows), breakpoints_by_k


def segment_summary(y: pd.Series, break_indices: list[int]) -> pd.DataFrame:
    s = y.dropna()
    idx = s.index
    values = s.values

    bounds = [0] + break_indices + [len(s)]
    rows = []

    for seg_id, (a, b) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
        seg = values[a:b]
        rows.append(
            {
                "segment": seg_id,
                "start": str(idx[a].date()),
                "end": str(idx[b - 1].date()),
                "n_months": int(b - a),
                "mean_monthly": float(np.mean(seg)),
                "vol_monthly": float(np.std(seg, ddof=1)) if len(seg) > 1 else np.nan,
                "ann_mean_approx": float(np.mean(seg) * 12),
                "ann_vol_approx": float(np.std(seg, ddof=1) * np.sqrt(12)) if len(seg) > 1 else np.nan,
                "min_monthly": float(np.min(seg)),
                "max_monthly": float(np.max(seg)),
            }
        )

    return pd.DataFrame(rows)


def distance_to_events(break_date: pd.Timestamp) -> dict:
    events = {
        "2016_momentum_crash": pd.Timestamp("2016-03-01"),
        "2020_covid_crash": pd.Timestamp("2020-03-01"),
        "2022_rate_hike_stress": pd.Timestamp("2022-03-01"),
    }

    distances = {
        name: abs((break_date - event_date).days)
        for name, event_date in events.items()
    }

    nearest = min(distances, key=distances.get)

    return {
        "nearest_event": nearest,
        "distance_days": int(distances[nearest]),
    }


def collect_break_table(
    sample: str,
    feature_name: str,
    y: pd.Series,
    model_selection: pd.DataFrame,
    breakpoints_by_k: dict[int, list[int]],
) -> tuple[list[dict], list[pd.DataFrame]]:
    rows = []
    segment_tables = []

    selected_bic = int(model_selection.sort_values("bic").iloc[0]["n_breaks"])
    selected_lwz = int(model_selection.sort_values("lwz_like").iloc[0]["n_breaks"])

    for criterion, k in [("bic", selected_bic), ("lwz_like", selected_lwz)]:
        breaks = breakpoints_by_k[k]
        dates = [y.dropna().index[i - 1] for i in breaks]

        for rank, d in enumerate(dates, start=1):
            event_info = distance_to_events(pd.Timestamp(d))
            rows.append(
                {
                    "sample": sample,
                    "feature": feature_name,
                    "criterion": criterion,
                    "selected_n_breaks": k,
                    "break_rank": rank,
                    "break_date": str(pd.Timestamp(d).date()),
                    **event_info,
                }
            )

        seg = segment_summary(y, breaks)
        seg.insert(0, "sample", sample)
        seg.insert(1, "feature", feature_name)
        seg.insert(2, "criterion", criterion)
        seg.insert(3, "selected_n_breaks", k)
        segment_tables.append(seg)

    return rows, segment_tables


def main() -> None:
    root = repo_root()
    out_dir = root / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    is_df = build_ensemble(read_runs(root / IS_ROOT))
    oos_df = build_ensemble(read_runs(root / OOS_ROOT, eval_window="full_oos_2020_2023"))

    full_daily = pd.concat([is_df, oos_df]).sort_index()
    full_monthly = add_monthly_features(monthly_compound(full_daily))
    is_monthly = full_monthly.loc["2015-01-01":"2019-12-31"]
    oos_monthly = full_monthly.loc["2020-01-01":"2023-12-31"]
    all_monthly = full_monthly.loc["2015-01-01":"2023-12-31"]

    full_monthly.to_csv(out_dir / "structural_break_monthly_features.csv")

    samples = {
        "is_2015_2019": is_monthly,
        "oos_2020_2023": oos_monthly,
        "full_2015_2023": all_monthly,
    }

    features = [
        "strategy_total_r",
        "strategy_vol_3m",
        "strategy_vol_6m",
        "market_vol_3m",
        "market_vol_6m",
        "momentum_factor_r",
        "momentum_factor_trailing_3m",
        "momentum_factor_trailing_6m",
        "momentum_factor_vol_3m",
        "momentum_factor_vol_6m",
    ]

    all_selection_rows = []
    all_break_rows = []
    all_segment_tables = []

    for sample_name, sample_df in samples.items():
        for feature in features:
            y = sample_df[feature].dropna()

            if sample_name == "oos_2020_2023":
                max_breaks = 2
                min_size = 6
            elif sample_name == "is_2015_2019":
                max_breaks = 2
                min_size = 6
            else:
                max_breaks = 4
                min_size = 6

            model_selection, breakpoints_by_k = estimate_breaks_dp(
                y,
                max_breaks=max_breaks,
                min_size=min_size,
            )

            model_selection.insert(0, "sample", sample_name)
            model_selection.insert(1, "feature", feature)
            all_selection_rows.append(model_selection)

            break_rows, segment_tables = collect_break_table(
                sample_name,
                feature,
                y,
                model_selection,
                breakpoints_by_k,
            )
            all_break_rows.extend(break_rows)
            all_segment_tables.extend(segment_tables)

    selection_df = pd.concat(all_selection_rows, ignore_index=True)
    breaks_df = pd.DataFrame(all_break_rows)
    segments_df = pd.concat(all_segment_tables, ignore_index=True)

    selection_df.to_csv(out_dir / "structural_break_model_selection.csv", index=False)
    breaks_df.to_csv(out_dir / "structural_break_dates.csv", index=False)
    segments_df.to_csv(out_dir / "structural_break_segment_stats.csv", index=False)

    focus_features = [
        "strategy_total_r",
        "strategy_vol_3m",
        "strategy_vol_6m",
        "momentum_factor_trailing_3m",
        "momentum_factor_vol_3m",
        "market_vol_3m",
    ]

    focus_breaks = breaks_df[breaks_df["feature"].isin(focus_features)].copy()
    focus_breaks = focus_breaks.sort_values(
        ["sample", "feature", "criterion", "break_rank"]
    )

    log_lines = []
    log_lines.append("Structural break diagnostics")
    log_lines.append("")
    log_lines.append("Method:")
    log_lines.append(
        "Piecewise-constant mean structural break detection using dynamic programming. "
        "Break dates minimize the global residual sum of squares for each fixed number of breaks."
    )
    log_lines.append("")
    log_lines.append("Important caveat:")
    log_lines.append(
        "This is an ex-post diagnostic only, not a trading rule. Full-sample break estimation "
        "would introduce look-ahead bias if used for live exposure decisions."
    )
    log_lines.append("")
    log_lines.append("Focus break dates:")
    log_lines.append(focus_breaks.to_string(index=False))

    (out_dir / "research_log_structural_breaks.txt").write_text(
        "\n".join(log_lines),
        encoding="utf-8",
    )

    print("Saved results to:", out_dir)
    print()
    print("Focus break dates:")
    print(focus_breaks.to_string(index=False))
    print()
    print("Model selection preview:")
    preview = selection_df[
        selection_df["feature"].isin(focus_features)
    ].sort_values(["sample", "feature", "n_breaks"])
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()