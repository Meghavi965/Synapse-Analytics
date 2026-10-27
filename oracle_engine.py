from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any, Optional

import pandas as pd

from core.agent import SelfHealingBIAgent
from core.forecasting import ForecastResult, forecast_next_periods, to_monthly_series
from utils.safeguardai import sanitize_user_query


@dataclass(frozen=True)
class OracleAnswer:
    question: str
    asked_at: datetime
    trust_score: float
    diagnosis: str
    intent: dict[str, Any]
    result: Any
    pandas_code: str
    descriptive_insight: str
    forecast: Optional[ForecastResult]
    strategic_next_steps: list[str]
    verification_notes: list[str]


def _normalize_metric(metric: Optional[str]) -> Optional[str]:
    if not metric:
        return None
    m = metric.lower().strip()
    if m in {"sales"}:
        return "revenue"
    if m in {"price", "rate"}:
        return "price_inr"
    if m in {"expense", "expenses"}:
        return "Expenses"
    if m in {"profit"}:
        return "profit"
    if m in {"churn"}:
        return "churn"
    if m in {"revenue"}:
        return "revenue"
    return metric


def _pick_date_col(df: pd.DataFrame) -> Optional[str]:
    for name in ("date", "Date", "timestamp", "Timestamp", "dt"):
        if name in df.columns:
            return name
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    return None


def _compute_descriptive_change(df: pd.DataFrame, metric: str, date_col: str, query: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    tmp = df[[date_col, metric]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col])
    if tmp.empty:
        return f"{metric}: no valid dates found for change analysis.", ["No valid dates after parsing."]

    q = query.lower()
    m = metric.lower()
    is_price = ("price" in m) or ("rate" in m) or (m in {"close", "adj_close", "price_inr"})

    # If user asked for "today/current/latest", prefer last close vs previous close (price-like series).
    if is_price and any(k in q for k in ("today", "current", "latest", "now", "right now")):
        tmp = tmp.sort_values(date_col)
        if len(tmp) < 2:
            last_val = float(tmp[metric].iloc[-1])
            last_day = pd.Timestamp(tmp[date_col].iloc[-1]).strftime("%Y-%m-%d")
            notes.append("Computed latest price (insufficient history for day-over-day change).")
            return (f"Latest ({last_day}) {metric} was {last_val:.4g}.", notes)
        last_val = float(tmp[metric].iloc[-1])
        prev_val = float(tmp[metric].iloc[-2])
        pct = ((last_val - prev_val) / abs(prev_val)) * 100.0 if abs(prev_val) > 1e-9 else 0.0
        last_day = pd.Timestamp(tmp[date_col].iloc[-1]).strftime("%Y-%m-%d")
        notes.append("Computed latest price and day-over-day change.")
        return (f"Latest ({last_day}) {metric} was {last_val:.4g}, {pct:+.2f}% vs previous close ({prev_val:.4g}).", notes)

    tmp["month"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()
    if metric.lower() == "churn" or is_price:
        series = tmp.groupby("month")[metric].mean().sort_index()
    else:
        series = tmp.groupby("month")[metric].sum().sort_index()

    if len(series) < 2:
        return f"{metric}: insufficient history to compute change.", ["Need >= 2 months for change analysis."]

    # Quarter-specific request (e.g., Q3 or Q3 2025).
    q_match = re.search(r"\bq([1-4])\b", query.lower())
    year_match = re.search(r"\b(20\d{2})\b", query)
    if q_match:
        q = int(q_match.group(1))
        year = int(year_match.group(1)) if year_match else int(series.index.max().year)
        period = pd.Period(f"{year}Q{q}", freq="Q")
        q_start = period.start_time.to_period("M").to_timestamp()
        q_end = period.end_time.to_period("M").to_timestamp()
        q_series = series.loc[(series.index >= q_start) & (series.index <= q_end)]
        prev_q = (period - 1)
        prev_start = prev_q.start_time.to_period("M").to_timestamp()
        prev_end = prev_q.end_time.to_period("M").to_timestamp()
        prev_series = series.loc[(series.index >= prev_start) & (series.index <= prev_end)]
        if len(q_series) >= 1 and len(prev_series) >= 1:
            curr = float(q_series.mean())
            prev = float(prev_series.mean())
            pct = ((curr - prev) / abs(prev)) * 100.0 if abs(prev) > 1e-9 else 0.0
            notes.append("Computed quarter-over-quarter change.")
            prev_label = f"{prev_q.year}-Q{prev_q.quarter}"
            return (
                f"In {year}-Q{q}, {metric} averaged {curr:.4g}, {pct:+.1f}% vs {prev_label} ({prev:.4g}).",
                notes,
            )
        notes.append("Quarter requested but not enough data; falling back to last-month change.")

    # Default: last month vs previous month.
    last = float(series.iloc[-1])
    prev = float(series.iloc[-2])
    pct = ((last - prev) / abs(prev)) * 100.0 if abs(prev) > 1e-9 else 0.0
    notes.append("Computed last-month change.")
    last_month = pd.Timestamp(series.index[-1]).strftime("%Y-%m")
    prev_month = pd.Timestamp(series.index[-2]).strftime("%Y-%m")
    return (
        f"Last month ({last_month}) {metric} was {last:.4g}, {pct:+.1f}% vs {prev_month} ({prev:.4g}).",
        notes,
    )


def _strategic_steps(metric: str, df: pd.DataFrame) -> list[str]:
    m = metric.lower()
    steps: list[str] = []
    is_price = ("price" in m) or ("rate" in m) or (m in {"close", "adj_close", "price_inr"})
    if is_price:
        # Compute simple, explainable technical indicators (educational).
        snap = "Market snapshot"
        try:
            date_col = _pick_date_col(df) or "date"
            if date_col in df.columns and metric in df.columns:
                tmp = df[[date_col, metric] + (["asset"] if "asset" in df.columns else []) + (["unit"] if "unit" in df.columns else [])].copy()
                tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
                tmp = tmp.dropna(subset=[date_col]).sort_values(date_col)
                if not tmp.empty:
                    asset = None
                    if "asset" in tmp.columns:
                        vals = tmp["asset"].dropna().astype(str).str.lower().unique().tolist()
                        asset = vals[0] if len(vals) == 1 else None
                    unit = None
                    if "unit" in tmp.columns:
                        u = tmp["unit"].dropna().astype(str).unique().tolist()
                        unit = u[0] if len(u) == 1 else None

                    series = tmp.set_index(date_col)[metric].astype(float)
                    last = float(series.iloc[-1])
                    sma7 = float(series.rolling(7).mean().iloc[-1]) if len(series) >= 7 else float("nan")
                    sma30 = float(series.rolling(30).mean().iloc[-1]) if len(series) >= 30 else float("nan")
                    trend = None
                    if pd.notna(sma7) and pd.notna(sma30):
                        trend = "bullish" if sma7 > sma30 else "bearish" if sma7 < sma30 else "flat"

                    delta = series.diff()
                    up = delta.clip(lower=0)
                    down = (-delta).clip(lower=0)
                    roll_up = up.rolling(14).mean()
                    roll_down = down.rolling(14).mean().replace(0, pd.NA)
                    rs = roll_up / roll_down
                    rsi = 100 - (100 / (1 + rs))
                    rsi_last = float(rsi.iloc[-1]) if len(rsi) and pd.notna(rsi.iloc[-1]) else None

                    ret_std = float(series.pct_change().dropna().tail(30).std()) if len(series) >= 5 else float("nan")
                    vol = f"{ret_std:.1%}/day (30D stdev)" if pd.notna(ret_std) else "N/A"

                    parts = []
                    if asset:
                        parts.append(asset.title())
                    parts.append(f"latest={last:,.0f}" + (f" INR/{unit}" if unit else ""))
                    if trend:
                        parts.append(f"trend={trend} (7D vs 30D avg)")
                    if rsi_last is not None:
                        state = "oversold" if rsi_last < 30 else "overbought" if rsi_last > 70 else "neutral"
                        parts.append(f"RSI14={rsi_last:.0f} ({state})")
                    parts.append(f"volatility={vol}")
                    snap = "Market snapshot: " + "; ".join(parts) + "."
        except Exception:
            snap = "Market snapshot: indicator computation unavailable."

        steps = [
            f"Educational only (not financial advice). {snap}",
            "Timing (rule-based): if RSI is overbought (>70), some strategies wait or split buys; if oversold (<30), some accumulate gradually. Prefer DCA + price alerts near pullbacks instead of trying to time one perfect entry.",
            "Sizing (risk-based): cap allocation to a small % of investable funds, split into 3–5 tranches, and keep emergency cash intact; metals can be volatile and INR conversion + local premiums matter.",
        ]
        return steps[:3]
    if m == "churn":
        # Simple segmentation evidence.
        top_region = None
        if "region" in df.columns and "churn" in df.columns:
            by_region = df.groupby("region")["churn"].mean().sort_values(ascending=False)
            if not by_region.empty:
                top_region = str(by_region.index[0])
        steps = [
            "Prioritize retention in the highest-churn segment (start with the worst region if available)."
            + (f" Current worst region: {top_region}." if top_region else ""),
            "Launch a win-back campaign for at-risk customers (e.g., 10% limited-time discount + outreach sequence).",
            "Run a driver analysis (age, region, spend) and fix top 1–2 friction points in onboarding/support.",
        ]
    elif m in {"revenue", "sales"}:
        steps = [
            "Concentrate budget on the highest-performing region/channel and replicate its playbook.",
            "Increase ARPA via targeted upsell bundles for high-value cohorts.",
            "Protect margin by aligning discounts to churn-risk and tightening approval controls.",
        ]
    elif m in {"expenses", "cost"}:
        steps = [
            "Identify top expense drivers and set weekly guardrails (alerts when variance exceeds threshold).",
            "Negotiate vendor tiers and remove underused subscriptions.",
            "Automate manual ops to reduce recurring overhead while maintaining SLA.",
        ]
    else:
        steps = [
            "Confirm KPI definition, timeframe, and segment before taking action.",
            "Instrument leading indicators and set thresholds for early warning.",
            "Run a short-cycle experiment (2–4 weeks) and measure impact vs baseline.",
        ]

    return steps[:3]


def _chain_of_verification(df: pd.DataFrame, metric: str, descriptive: str) -> list[str]:
    notes: list[str] = []
    if metric not in df.columns:
        notes.append(f"CoV: metric '{metric}' not present in df.columns; verification limited.")
        return notes

    if not pd.api.types.is_numeric_dtype(df[metric]):
        notes.append(f"CoV: metric '{metric}' is non-numeric; verification limited.")
        return notes

    desc = df[[metric]].describe()
    mn = float(desc.loc["min", metric])
    mx = float(desc.loc["max", metric])
    notes.append(f"CoV: verified metric range [{mn:.4g}, {mx:.4g}] via df.describe().")
    if "nan" in descriptive.lower():
        notes.append("CoV: response contained 'NaN' token; consider data cleaning.")
    return notes


class OracleEngine:
    """
    Executive 'Oracle' workflow: Research -> Analyze -> Advise.
    """

    def __init__(self, agent: SelfHealingBIAgent):
        self.agent = agent

    def answer(self, user_query: str, df: pd.DataFrame) -> OracleAnswer:
        asked_at = datetime.now()
        safe_query = sanitize_user_query(user_query)

        # Research: ML health verification + Pandas logic generation.
        trust_score = 0.0
        diagnosis = "Monitor Untrained"
        if getattr(self.agent.health_monitor, "is_trained", False):
            trust_score, diagnosis = self.agent.health_monitor.predict_health_score(df)

        insights = self.agent.query_insights(safe_query, df)
        pandas_code = str(insights.get("pandas_code") or "")
        intent = insights.get("intent") or {}
        result = insights.get("result")
        metric = _normalize_metric((intent.get("metrics") or [None])[0])
        if not metric:
            if any(c in df.columns for c in ("revenue", "Sales_USD", "Revenue_Amt")):
                metric = "revenue"
            elif "price_inr" in df.columns:
                metric = "price_inr"
            elif "churn" in df.columns:
                metric = "churn"
            else:
                metric = (df.select_dtypes("number").columns.tolist() or ["metric"])[0]

        # Apply intent filters for analysis (e.g., asset == 'gold').
        analysis_df = df
        filters = intent.get("filters") or {}
        if "asset" in filters and "asset" in df.columns:
            asset_val = filters["asset"]
            if isinstance(asset_val, (list, tuple, set)):
                analysis_df = df[df["asset"].isin(list(asset_val))].copy()
            else:
                analysis_df = df[df["asset"].astype(str).str.lower() == str(asset_val).lower()].copy()

        # Materialize virtual metrics (e.g., 'revenue') on SmartBIFrame so downstream analysis/forecasting works.
        if metric and metric not in analysis_df.columns:
            try:
                _ = analysis_df[metric]  # may trigger SmartBIFrame self-heal and add the column
            except Exception:
                pass

        # Analyze: descriptive insight (attempt exact change).
        date_col = _pick_date_col(analysis_df)
        descriptive = insights.get("insight") or ""
        analysis_notes: list[str] = []
        if date_col and metric in analysis_df.columns:
            try:
                descriptive, analysis_notes = _compute_descriptive_change(analysis_df, metric, date_col, safe_query)
            except Exception as e:
                analysis_notes = [f"Descriptive change fallback: {e}"]

        # Predict: forecast next 3 months (if possible).
        forecast: Optional[ForecastResult] = None
        if date_col and metric in analysis_df.columns and pd.api.types.is_numeric_dtype(analysis_df[metric]):
            m = metric.lower()
            is_price = ("price" in m) or ("rate" in m) or (m in {"close", "adj_close", "price_inr"})
            agg = "mean" if (m == "churn" or is_price) else "sum"
            try:
                monthly = to_monthly_series(analysis_df, metric, date_col=date_col, agg=agg)
                forecast = forecast_next_periods(monthly, metric=metric, date_col=date_col, agg=agg, periods=3)
            except Exception as e:
                analysis_notes.append(f"Forecast unavailable: {e}")

        # Advise: actionable next steps.
        steps = _strategic_steps(metric, analysis_df)

        verification = _chain_of_verification(analysis_df, metric, descriptive)
        verification = analysis_notes + verification

        return OracleAnswer(
            question=safe_query,
            asked_at=asked_at,
            trust_score=trust_score,
            diagnosis=diagnosis,
            intent=intent,
            result=result,
            pandas_code=pandas_code,
            descriptive_insight=descriptive,
            forecast=forecast,
            strategic_next_steps=steps,
            verification_notes=verification,
        )
