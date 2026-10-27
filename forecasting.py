from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    ape = np.abs((y_true - y_pred) / denom)
    mape = np.nanmean(ape)
    if np.isnan(mape):
        return None
    return float(mape)


@dataclass(frozen=True)
class ForecastResult:
    metric: str
    date_col: str
    agg: Literal["mean", "sum"]
    periods: int
    model_name: str
    accuracy_r2: Optional[float]
    accuracy_mape: Optional[float]
    history: pd.DataFrame  # columns: date, value
    forecast: pd.DataFrame  # columns: date, prediction


def to_monthly_series(
    df: pd.DataFrame,
    metric: str,
    *,
    date_col: str = "date",
    agg: Literal["mean", "sum"] = "mean",
) -> pd.Series:
    if date_col not in df.columns:
        raise ValueError(f"Missing date column '{date_col}'.")
    if metric not in df.columns:
        raise ValueError(f"Missing metric column '{metric}'.")

    tmp = df[[date_col, metric]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])
    if tmp.empty:
        raise ValueError("No valid dates after parsing.")

    tmp["month"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()
    if agg == "sum":
        series = tmp.groupby("month")[metric].sum()
    else:
        series = tmp.groupby("month")[metric].mean()

    series = series.sort_index()
    return series


def forecast_next_periods(
    monthly_series: pd.Series,
    *,
    metric: str,
    date_col: str = "date",
    agg: Literal["mean", "sum"] = "mean",
    periods: int = 3,
    random_state: int = 42,
) -> ForecastResult:
    if len(monthly_series) < 6:
        raise ValueError("Need at least 6 months of history to forecast.")

    y = monthly_series.astype(float).values
    idx = monthly_series.index
    t = np.arange(len(y), dtype=float)
    month = pd.DatetimeIndex(idx).month.astype(float)
    X = np.column_stack(
        [
            t,
            np.sin(2 * np.pi * month / 12.0),
            np.cos(2 * np.pi * month / 12.0),
        ]
    )

    # Simple holdout: last 3 months (or less if small).
    test_n = min(3, max(1, len(y) // 4))
    split = len(y) - test_n
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        max_depth=6,
        min_samples_leaf=2,
    )
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    r2 = None
    if len(y_test) >= 2:
        r2 = float(r2_score(y_test, y_pred_test))
    mape = _safe_mape(y_test, y_pred_test)

    # Refit on all data.
    model.fit(X, y)

    last_month = pd.Timestamp(idx[-1]).to_period("M").to_timestamp()
    future_months = pd.date_range(last_month + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
    t_future = np.arange(len(y), len(y) + periods, dtype=float)
    month_future = future_months.month.astype(float)
    X_future = np.column_stack(
        [
            t_future,
            np.sin(2 * np.pi * month_future / 12.0),
            np.cos(2 * np.pi * month_future / 12.0),
        ]
    )
    y_future = model.predict(X_future)

    history_df = pd.DataFrame({"date": idx, "value": y})
    forecast_df = pd.DataFrame({"date": future_months, "prediction": y_future})

    return ForecastResult(
        metric=metric,
        date_col=date_col,
        agg=agg,
        periods=periods,
        model_name="RandomForestRegressor(time+sine/cosine)",
        accuracy_r2=r2,
        accuracy_mape=mape,
        history=history_df,
        forecast=forecast_df,
    )

