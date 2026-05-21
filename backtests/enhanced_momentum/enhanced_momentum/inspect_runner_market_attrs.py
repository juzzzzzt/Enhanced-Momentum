from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KEYWORDS = [
    "benchmark",
    "bench",
    "market",
    "index",
    "spx",
    "russell",
    "buy",
    "hold",
    "return",
    "returns",
    "excess",
    "total",
    "rf",
    "risk",
    "hedge",
]


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p.parent, *p.parents]:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Cannot locate repo root")


def _is_interesting_name(name: str) -> bool:
    low = name.lower()
    return any(k in low for k in KEYWORDS)


def _safe_getattr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name)
    except Exception as e:
        return f"<getattr failed: {type(e).__name__}: {e}>"


def _describe_value(name: str, value: Any, indent: str = "") -> None:
    print(f"{indent}{name}")
    print(f"{indent}  type: {type(value)}")

    if isinstance(value, pd.DataFrame):
        print(f"{indent}  shape: {value.shape}")
        print(f"{indent}  columns: {list(value.columns)[:20]}")
        print(f"{indent}  index: {type(value.index)}")
        print(f"{indent}  head:")
        print(value.head().to_string())
        _try_return_stats(value, indent=indent + "  ")

    elif isinstance(value, pd.Series):
        print(f"{indent}  shape: {value.shape}")
        print(f"{indent}  name: {value.name}")
        print(f"{indent}  index: {type(value.index)}")
        print(f"{indent}  head:")
        print(value.head().to_string())
        _try_return_stats(value, indent=indent + "  ")

    elif isinstance(value, (str, int, float, bool, type(None))):
        print(f"{indent}  value: {value}")

    elif isinstance(value, (list, tuple, set)):
        print(f"{indent}  len: {len(value)}")
        print(f"{indent}  preview: {list(value)[:10]}")

    elif isinstance(value, dict):
        print(f"{indent}  keys: {list(value.keys())[:30]}")

    else:
        attrs = _interesting_attrs(value)
        print(f"{indent}  interesting nested attrs: {attrs[:30]}")


def _try_return_stats(value: pd.Series | pd.DataFrame, indent: str = "") -> None:
    try:
        if isinstance(value, pd.DataFrame):
            numeric = value.select_dtypes(include=[np.number])
            if numeric.empty:
                return
            if numeric.shape[1] == 1:
                s = numeric.iloc[:, 0]
                _print_series_stats(s, indent=indent)
            else:
                for col in numeric.columns[:10]:
                    print(f"{indent}stats for column: {col}")
                    _print_series_stats(numeric[col], indent=indent + "  ")
        else:
            _print_series_stats(value, indent=indent)
    except Exception as e:
        print(f"{indent}stats failed: {type(e).__name__}: {e}")


def _print_series_stats(s: pd.Series, indent: str = "") -> None:
    s = pd.to_numeric(s, errors="coerce").dropna()

    if s.empty:
        print(f"{indent}empty numeric series")
        return

    mean_daily = s.mean()
    vol_daily = s.std(ddof=1)

    ann_mean = mean_daily * 252
    ann_vol = vol_daily * np.sqrt(252)

    print(f"{indent}n: {len(s)}")
    print(f"{indent}mean_daily: {mean_daily:.8f}")
    print(f"{indent}vol_daily: {vol_daily:.8f}")
    print(f"{indent}ann_mean: {ann_mean:.4%}")
    print(f"{indent}ann_vol: {ann_vol:.4%}")
    print(f"{indent}min: {s.min():.4%}")
    print(f"{indent}max: {s.max():.4%}")


def _interesting_attrs(obj: Any) -> list[str]:
    out = []
    for name in dir(obj):
        if name.startswith("__"):
            continue
        if _is_interesting_name(name):
            out.append(name)
    return sorted(out)


def _dump_interesting_object(obj_name: str, obj: Any, max_nested: int = 2) -> None:
    print("\n" + "=" * 100)
    print(f"OBJECT: {obj_name}")
    print("=" * 100)
    print(f"type: {type(obj)}")

    attrs = _interesting_attrs(obj)
    print(f"interesting attrs ({len(attrs)}):")
    for name in attrs:
        print(f"  - {name}")

    print("\nDetailed interesting attrs:")
    for name in attrs:
        value = _safe_getattr(obj, name)
        print("\n" + "-" * 80)
        _describe_value(f"{obj_name}.{name}", value)

        if max_nested > 0 and not isinstance(
            value,
            (pd.DataFrame, pd.Series, str, int, float, bool, type(None), list, tuple, set, dict),
        ):
            nested_attrs = _interesting_attrs(value)
            for nested_name in nested_attrs[:25]:
                nested_value = _safe_getattr(value, nested_name)
                print("\n" + "." * 80)
                _describe_value(f"{obj_name}.{name}.{nested_name}", nested_value, indent="  ")


def _dump_config_all_caps(config_name: str, cfg: Any) -> None:
    print("\n" + "=" * 100)
    print(f"CONFIG: {config_name}")
    print("=" * 100)
    print(f"type: {type(cfg)}")

    names = []
    for name in dir(cfg):
        if name.startswith("__"):
            continue
        if name.isupper() or _is_interesting_name(name):
            names.append(name)

    names = sorted(set(names))

    for name in names:
        value = _safe_getattr(cfg, name)
        if callable(value):
            continue

        if isinstance(value, (str, int, float, bool, type(None), Path)):
            print(f"{name}: {value}")
        elif isinstance(value, (list, tuple, set)):
            print(f"{name}: {type(value)} len={len(value)} preview={list(value)[:10]}")
        elif isinstance(value, dict):
            print(f"{name}: dict keys={list(value.keys())[:20]}")
        else:
            print(f"{name}: {type(value)}")


def _filter_kwargs_for_ctor(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed}


def main() -> None:
    from enhanced_momentum.config.project_experiment_config import ProjectExperimentConfig
    from enhanced_momentum.run import run_backtest
    from enhanced_momentum.strategies.systematic_momentum import SystematicMomentum

    strategy_kwargs = {
        "mode": "long_short",
        "quantile": 0.2,
        "window_days": 126,
        "exclude_last_days": 84,
        "as_zscore": False,
        "weighting_scheme": "equally_weighted",
        "return_type": "simple",
        "volatility_scaling": True,
        "vol_window_days": 21,
    }
    strategy_kwargs = _filter_kwargs_for_ctor(SystematicMomentum, strategy_kwargs)
    strategy = SystematicMomentum(**strategy_kwargs)

    exp_cfg = ProjectExperimentConfig()
    exp_cfg.HEDGE_FREQ = "ME"

    _dump_config_all_caps("ProjectExperimentConfig BEFORE run", exp_cfg)

    rb_sig = inspect.signature(run_backtest)
    rb_kwargs = {
        "strategy": strategy,
        "rebal_freq": "ME",
        "experiment_cfg": exp_cfg,
        "start_date": pd.Timestamp("2023-01-01"),
        "end_date": pd.Timestamp("2023-12-31"),
        "plot": False,
        "make_plots": False,
        "return_runner": True,
    }
    rb_kwargs = {k: v for k, v in rb_kwargs.items() if k in rb_sig.parameters}

    result = run_backtest(**rb_kwargs)

    if not isinstance(result, tuple):
        raise RuntimeError("run_backtest did not return tuple. Check return_runner=True support.")

    metrics_df, runner = result

    print("\n" + "=" * 100)
    print("METRICS DF")
    print("=" * 100)
    print(metrics_df.to_string())

    _dump_config_all_caps("ProjectExperimentConfig AFTER run", exp_cfg)
    _dump_interesting_object("runner", runner, max_nested=2)

    for obj_name in [
        "benchmark",
        "market",
        "assessor",
        "statistics",
        "strategy",
        "experiment_config",
        "trading_config",
        "dataset",
        "dataset_data",
        "data",
    ]:
        if hasattr(runner, obj_name):
            obj = _safe_getattr(runner, obj_name)
            _dump_interesting_object(f"runner.{obj_name}", obj, max_nested=1)

    print("\n" + "=" * 100)
    print("DIRECT SANITY CHECK: total_r - excess_r")
    print("=" * 100)

    total_r = getattr(runner, "strategy_total_r", None)
    excess_r = getattr(runner, "strategy_excess_r", None)

    if isinstance(total_r, pd.DataFrame) and isinstance(excess_r, pd.DataFrame):
        total_s = total_r.iloc[:, 0]
        excess_s = excess_r.iloc[:, 0]
        candidate = total_s - excess_s
        candidate.name = "total_minus_excess"
        _describe_value("runner.strategy_total_r - runner.strategy_excess_r", candidate)
    else:
        print("strategy_total_r / strategy_excess_r not available as DataFrame")


if __name__ == "__main__":
    main()