from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.agent import SelfHealingBIAgent
from core.oracle_engine import OracleAnswer, OracleEngine
from core.self_healing_df import SmartBIFrame
from ui_theme import executive_card, inject_executive_theme
from utils.executive_report import generate_executive_report
from utils.live_market_data import LiveMarketDataError, MetalsDataConfig, fetch_metals_india_prices
from utils.market_research import fetch_market_context, MarketResearchResult

logger = logging.getLogger(__name__)

TRUST_THRESHOLD = 0.7
LIVE_METALS_TTL_SECONDS = 300


@st.cache_data(ttl=LIVE_METALS_TTL_SECONDS, show_spinner=False)
def _cached_metals_prices(
    history_range: str,
    interval: str,
    gold_unit: str,
    silver_unit: str,
    nonce: int,
) -> pd.DataFrame:
    _ = nonce  # cache-buster
    cfg = MetalsDataConfig(
        history_range=history_range,
        interval=interval,
        gold_unit=gold_unit,
        silver_unit=silver_unit,
    )
    return fetch_metals_india_prices(cfg)


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _memory_path() -> Path:
    root = _workspace_root()
    out_dir = root / "output" / "memory"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "oracle_memory.json"


def _load_memory() -> list[dict[str, Any]]:
    path = _memory_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_memory(entries: list[dict[str, Any]]) -> None:
    path = _memory_path()
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def _toast(message: str) -> None:
    toast_fn = getattr(st, "toast", None)
    if callable(toast_fn):
        toast_fn(message)
    else:
        st.success(message)


def _load_demo_data(n: int = 3500, seed: int = 42) -> SmartBIFrame:
    """
    Executive demo dataset with a realistic time signal to support forecasting.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range(
        end=pd.Timestamp.today().to_period("M").to_timestamp(),
        periods=24,
        freq="MS",
    )
    month_idx = rng.integers(0, len(months), size=n)
    dates = months[month_idx] + pd.to_timedelta(rng.integers(0, 28, size=n), unit="D")

    regions = np.array(["North", "South", "East", "West"])
    region = rng.choice(regions, size=n)

    trend = month_idx / max(len(months) - 1, 1)
    region_uplift = np.select(
        [region == "North", region == "South", region == "East", region == "West"],
        [1.15, 0.95, 1.05, 1.00],
        default=1.0,
    )

    sales_mean = (9000 + 3500 * trend) * region_uplift
    sales_usd = rng.normal(sales_mean, 1500).clip(min=0)
    expenses = rng.normal((sales_usd * 0.68), 900).clip(min=0)

    churn_prob = (0.08 + 0.09 * trend) * np.select(
        [region == "South", region == "West"],
        [1.10, 1.05],
        default=1.0,
    )
    churn_prob = churn_prob.clip(0.02, 0.35)
    churn = rng.binomial(1, churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "date": dates,
            "region": region,
            "Sales_USD": sales_usd,
            "Expenses": expenses,
            "churn": churn,
            "customer_age": rng.integers(18, 80, size=n),
        }
    )
    return SmartBIFrame(df, context_map={"revenue": "Sales_USD or Revenue_Amt"})


def _get_agent() -> SelfHealingBIAgent:
    if "agent" not in st.session_state:
        st.session_state["agent"] = SelfHealingBIAgent()
    return st.session_state["agent"]


def _get_df() -> SmartBIFrame:
    source = st.session_state.get("data_source") or "Executive Demo (synthetic)"

    if source.startswith("Live Metals"):
        # Live metals config (approximate India INR via USD conversion).
        metals_range = st.session_state.get("metals_range") or "2y"
        metals_interval = st.session_state.get("metals_interval") or "1d"
        gold_unit = st.session_state.get("gold_unit") or "10g"
        silver_unit = st.session_state.get("silver_unit") or "1kg"
        refresh_nonce = int(st.session_state.get("metals_refresh_nonce") or 0)

        try:
            metals_df = _cached_metals_prices(metals_range, metals_interval, gold_unit, silver_unit, refresh_nonce)
            df_live = SmartBIFrame(
                metals_df,
                context_map={
                    "price": "price_inr",
                    "rate": "price_inr",
                },
            )
            st.session_state["df_raw"] = df_live
            st.session_state["df_raw_source"] = source
        except LiveMarketDataError as e:
            st.error(f"Live metals fetch failed: {e}")
            if "df_raw" not in st.session_state:
                st.session_state["df_raw"] = _load_demo_data()
                st.session_state["df_raw_source"] = "Executive Demo (synthetic)"

    if "df_raw" not in st.session_state:
        st.session_state["df_raw"] = _load_demo_data()
        st.session_state["df_raw_source"] = "Executive Demo (synthetic)"
    df = st.session_state["df_raw"]
    if not isinstance(df, SmartBIFrame):
        df = SmartBIFrame(df, context_map={"revenue": "Sales_USD or Revenue_Amt"})
        st.session_state["df_raw"] = df
    return df


def _train_baseline(agent: SelfHealingBIAgent, df: SmartBIFrame) -> None:
    healthy = [df]
    problematic_df = df.copy()

    # Prefer domain-specific drift injection for demo schema; otherwise fallback to generic drift.
    if "Sales_USD" in problematic_df.columns:
        problematic_df.loc[::10, "Sales_USD"] = np.nan
    elif "price_inr" in problematic_df.columns:
        problematic_df.loc[::10, "price_inr"] = np.nan
    else:
        numeric = problematic_df.select_dtypes(include="number").columns.tolist()
        if numeric:
            problematic_df.loc[::10, numeric[0]] = np.nan

    if "customer_age" in problematic_df.columns:
        problematic_df = problematic_df.drop("customer_age", axis=1)
    else:
        # Generic schema drift: drop the last column (if safe).
        if len(problematic_df.columns) >= 2:
            problematic_df = problematic_df.drop(problematic_df.columns[-1], axis=1)

    agent.register_baseline(healthy, [problematic_df])


def _predict_health(agent: SelfHealingBIAgent, df: pd.DataFrame) -> tuple[float, str, dict[str, float]]:
    score = 0.0
    diagnosis = "Monitor Untrained"
    feats = {"null_pct": 0.0, "outliers_pct": 0.0, "schema_change": 0.0}
    try:
        f = agent.health_monitor.extract_features(df)[0]
        feats = {"null_pct": float(f[0]), "outliers_pct": float(f[3]), "schema_change": float(f[4])}
    except Exception:
        pass

    if getattr(agent.health_monitor, "is_trained", False):
        try:
            score, diagnosis = agent.health_monitor.predict_health_score(df)
        except Exception as e:
            logger.error("Health prediction failed: %s", e)
            score, diagnosis = 0.0, "Prediction Error"
    return score, diagnosis, feats


def _render_filters(df: SmartBIFrame) -> tuple[SmartBIFrame, int]:
    regions = sorted(df["region"].dropna().unique().tolist()) if "region" in df.columns else []
    assets = sorted(df["asset"].dropna().unique().tolist()) if "asset" in df.columns else []
    st.markdown("### Filters")
    sel_regions = st.multiselect("Region", regions, default=regions) if regions else []
    sel_assets = st.multiselect("Asset", assets, default=assets) if assets else []

    date_min = pd.to_datetime(df["date"]).min().date() if "date" in df.columns else None
    date_max = pd.to_datetime(df["date"]).max().date() if "date" in df.columns else None
    date_range = None
    if date_min and date_max:
        date_range = st.date_input("Date range", (date_min, date_max))

    preview_rows = st.slider("Preview rows", 25, 300, 120)

    mask = pd.Series(True, index=df.index)
    if sel_regions and "region" in df.columns:
        mask &= df["region"].isin(sel_regions)
    if sel_assets and "asset" in df.columns:
        mask &= df["asset"].isin(sel_assets)
    if date_range and "date" in df.columns:
        start, end = date_range
        dt = pd.to_datetime(df["date"]).dt.date
        mask &= (dt >= start) & (dt <= end)

    filtered = SmartBIFrame(df.loc[mask].copy(), context_map=getattr(df, "_context_map", None))
    return filtered, int(preview_rows)


def _proof_chart_from_answer(answer: OracleAnswer) -> Optional[Any]:
    """
    Build a "Graphical Proof" figure from the Oracle result.
    """
    result = answer.result
    metric = (answer.intent.get("metrics") or [None])[0]
    metric = metric or "metric"

    if result is None:
        return None
    if isinstance(result, pd.Series):
        tmp = result.reset_index()
        tmp.columns = ["group", "value"]
        if pd.api.types.is_datetime64_any_dtype(tmp["group"]):
            return px.line(tmp, x="group", y="value", title=f"{metric} over time (proof)")
        return px.bar(tmp, x="group", y="value", title=f"{metric} by group (proof)")
    if isinstance(result, pd.DataFrame):
        tmp = result.reset_index()
        numeric_cols = tmp.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            return None
        y = "mean" if "mean" in numeric_cols else numeric_cols[0]
        x = [c for c in tmp.columns if c != y][0]
        return px.bar(tmp, x=x, y=y, title=f"{metric} (proof)")
    return None


def _forecast_chart(answer: OracleAnswer):
    if not answer.forecast:
        return None
    hist = answer.forecast.history.copy()
    fc = answer.forecast.forecast.copy()
    hist["kind"] = "history"
    hist = hist.rename(columns={"value": "y", "date": "date"})
    fc["kind"] = "forecast"
    fc = fc.rename(columns={"prediction": "y", "date": "date"})
    combined = pd.concat([hist[["date", "y", "kind"]], fc[["date", "y", "kind"]]], ignore_index=True)
    return px.line(combined, x="date", y="y", color="kind", title=f"{answer.forecast.metric}: next 3 months forecast")


def _internal_kpi_context(df: pd.DataFrame, answer: OracleAnswer) -> str:
    metric_raw = (answer.intent.get("metrics") or [None])[0]
    metric = (metric_raw or "").lower()
    metric_col = "churn" if metric in {"", "churn"} else metric_raw
    if metric in {"sales", "revenue"}:
        metric_col = "revenue"
    if metric in {"expenses", "expense"}:
        metric_col = "Expenses"

    if "date" not in df.columns:
        return ""
    try:
        if metric_col == "revenue" and hasattr(df, "revenue"):
            _ = getattr(df, "revenue")
    except Exception:
        pass

    if metric_col not in df.columns:
        return ""

    tmp = df[["date", metric_col]].copy()
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    tmp = tmp.dropna(subset=["date"])
    if tmp.empty:
        return ""

    tmp["month"] = tmp["date"].dt.to_period("M").dt.to_timestamp()
    if metric_col.lower() == "churn":
        series = tmp.groupby("month")[metric_col].mean().sort_index()
        val = float(series.tail(3).mean()) if len(series) >= 1 else float("nan")
        return f"**Internal churn (last 3 months avg):** {val:.1%}"

    series = tmp.groupby("month")[metric_col].sum().sort_index()
    val = float(series.tail(3).mean()) if len(series) >= 1 else float("nan")
    return f"**Internal {metric_col} (last 3 months avg):** {val:,.0f}"


def run_executive_suite() -> None:
    st.set_page_config(page_title="Executive Strategy Suite", layout="wide")
    logging.basicConfig(level=logging.INFO)
    inject_executive_theme("dark_executive")

    if "data_source" not in st.session_state:
        st.session_state["data_source"] = "Executive Demo (synthetic)"

    agent = _get_agent()
    df_raw = _get_df()

    # Ensure semantic layer and train monitor for the current dataset.
    try:
        _ = df_raw.revenue
    except Exception:
        pass
    train_sig = f"{st.session_state.get('data_source','')}:{'|'.join([str(c) for c in df_raw.columns])}"
    if st.session_state.get("monitor_train_sig") != train_sig or not getattr(agent.health_monitor, "is_trained", False):
        _train_baseline(agent, df_raw)
        st.session_state["monitor_train_sig"] = train_sig

    # Memory
    if "memory_thread" not in st.session_state:
        st.session_state["memory_thread"] = _load_memory()
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    # Top bar placeholders (updated after filters resolve).
    top = st.container()
    with top:
        c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.0, 1.2], gap="large")
        trust_ph = c1.empty()
        health_ph = c2.empty()
        rows_ph = c3.empty()
        integ_ph = c4.empty()

    left, center, right = st.columns([1.15, 2.35, 1.25], gap="large")

    with left:
        st.markdown("### Data Source")
        source = st.selectbox(
            "Choose dataset",
            ["Executive Demo (synthetic)", "Live Metals (India: Gold/Silver)"],
            key="data_source",
        )
        if source.startswith("Live Metals"):
            with st.expander("Live stream settings", expanded=False):
                st.caption(
                    "Live prices are an approximation: global futures (GC=F/SI=F) converted via USD/INR. "
                    "Local premiums/taxes are not included. Educational use only."
                )
                u1, u2 = st.columns(2)
                u1.selectbox("Gold unit", ["10g", "1g", "oz"], key="gold_unit")
                u2.selectbox("Silver unit", ["1kg", "1g", "oz"], key="silver_unit")
                st.selectbox("History range", ["6mo", "1y", "2y", "5y", "max"], index=2, key="metals_range")
                st.selectbox("Interval", ["1d"], key="metals_interval")

                c1, c2 = st.columns([1, 1])
                if c1.button("Refresh prices now"):
                    st.session_state["metals_refresh_nonce"] = int(st.session_state.get("metals_refresh_nonce") or 0) + 1
                    st.rerun()
                auto = c2.checkbox("Auto-refresh", value=False, key="metals_autorefresh")
                if auto:
                    secs = int(st.slider("Refresh every (sec)", 30, 600, 180, key="metals_autorefresh_sec"))
                    st.markdown(f"<meta http-equiv='refresh' content='{secs}'>", unsafe_allow_html=True)

        # Reload the dataframe if the data source widget changed.
        if st.session_state.get("df_raw_source") != st.session_state.get("data_source"):
            df_raw = _get_df()
            st.session_state["chat_messages"] = []
            st.session_state["last_oracle_answer"] = None
            st.session_state["last_market_context"] = None
            st.rerun()

        df, preview_rows = _render_filters(df_raw)
        executive_card("Data Preview", "<span class='muted'>Slice under current filters.</span>")
        st.dataframe(df.head(preview_rows), use_container_width=True)

    score, diagnosis, feats = _predict_health(agent, df)
    schema_sig = "|".join([str(c) for c in df_raw.columns])
    is_bi_schema = "region" in df_raw.columns and ("churn" in df_raw.columns or "Sales_USD" in df_raw.columns or "revenue" in df_raw.columns)
    if is_bi_schema and score < TRUST_THRESHOLD and st.session_state.get("auto_heal_schema_sig") != schema_sig:
        st.session_state["auto_heal_schema_sig"] = schema_sig
        try:
            repaired = agent.repair_pipeline(
                df_raw,
                target_schema=["date", "region", "revenue", "Expenses", "churn", "customer_age"],
                context_dict={"revenue": "Sales_USD or Revenue_Amt or sales_amount"},
                threshold=0.5,
            )
        except Exception as e:
            logger.error("Auto-heal failed: %s", e)
            repaired = None

        if repaired is not None:
            st.session_state["df_raw"] = repaired
            st.session_state["auto_heal_schema_sig"] = "|".join([str(c) for c in repaired.columns])
            mapped_to = agent.last_mappings.get("revenue")
            if mapped_to:
                _toast(f"✅ Healing Complete: Mapped 'revenue' to '{mapped_to}' using semantic context.")
            else:
                _toast("✅ Healing Complete: Repair pipeline executed.")
            st.rerun()
    trust_ph.metric("Data Trust Score", f"{score:.0%}")
    health_ph.metric("System Health", "OK" if score >= TRUST_THRESHOLD else "ATTENTION", diagnosis)
    rows_ph.metric("Rows", f"{len(df):,}")
    integ_ph.metric("Integrity", f"Null={feats['null_pct']:.0%}", f"SchemaΔ={feats['schema_change']:.0%}")

    with center:
        st.markdown("## Insight Canvas")

        last_answer: Optional[OracleAnswer] = st.session_state.get("last_oracle_answer")

        if last_answer:
            executive_card("Descriptive Insight", last_answer.descriptive_insight)

            proof_fig = _proof_chart_from_answer(last_answer)
            if proof_fig is not None:
                st.plotly_chart(proof_fig, use_container_width=True)
            else:
                st.info("No proof chart available for the last answer.")

            fc_fig = _forecast_chart(last_answer)
            if fc_fig is not None:
                st.plotly_chart(fc_fig, use_container_width=True)
                if last_answer.forecast:
                    st.dataframe(last_answer.forecast.forecast, use_container_width=True)
            else:
                st.info("Forecast unavailable for the last answer (need metric + date history).")

            executive_card(
                "Strategic Next Steps",
                "<br/>".join([f"• {s}" for s in last_answer.strategic_next_steps]),
            )

            with st.expander("Chain-of-Verification (details)"):
                for n in last_answer.verification_notes:
                    st.write(f"- {n}")

            col_r1, col_r2 = st.columns([1.0, 1.0])
            with col_r1:
                gen_report = st.button("Generate Executive Report")
            with col_r2:
                include_market = st.toggle("Include Market Context", value=False)

            if gen_report:
                market = None
                if include_market:
                    market = fetch_market_context(last_answer.question)
                    internal_md = _internal_kpi_context(df, last_answer)
                    if market and internal_md:
                        market = MarketResearchResult(
                            enabled=market.enabled,
                            summary_markdown=market.summary_markdown + "\n\n" + internal_md,
                            sources=market.sources,
                            benchmark_candidates=market.benchmark_candidates,
                        )
                    st.session_state["last_market_context"] = market
                artifacts = generate_executive_report(
                    last_answer,
                    df_snapshot=df.copy(),
                    proof_fig=proof_fig,
                    forecast_fig=fc_fig,
                    market=market,
                )
                _toast(f"Report generated: {artifacts.markdown_path.name}" + (f" + {artifacts.pdf_path.name}" if artifacts.pdf_path else " (PDF unavailable)"))

                md_bytes = artifacts.markdown_path.read_bytes()
                st.download_button("Download Markdown", md_bytes, file_name=artifacts.markdown_path.name)
                if artifacts.pdf_path and artifacts.pdf_path.exists():
                    st.download_button("Download PDF", artifacts.pdf_path.read_bytes(), file_name=artifacts.pdf_path.name)
                else:
                    st.warning("PDF export requires `reportlab`, and chart embedding requires `kaleido`.")

        else:
            st.info("Ask the Oracle a question to populate the Insight Canvas.")

    with right:
        st.markdown("## Strategy Chatbot (Oracle)")

        with st.expander("Memory Thread", expanded=True):
            st.caption("Persistent across reruns; can be saved to disk.")
            if st.button("Save Memory Thread"):
                _save_memory(st.session_state["memory_thread"])
                _toast("Memory saved.")
            if st.button("Clear Memory Thread"):
                st.session_state["memory_thread"] = []
                _save_memory([])
                _toast("Memory cleared.")
                st.rerun()

            for item in st.session_state["memory_thread"][-8:]:
                st.write(f"- **{item.get('asked_at','')}** — {item.get('question','')}")

        for msg in st.session_state["chat_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("code"):
                    st.code(msg["code"], language="python")

        prompt = st.chat_input("Ask for insights, forecasts, or 'Market Context'...")
        if prompt:
            st.session_state["chat_messages"].append({"role": "user", "content": prompt})

            oracle = OracleEngine(agent)
            try:
                answer = oracle.answer(prompt, df)
            except Exception as e:
                st.session_state["chat_messages"].append({"role": "assistant", "content": f"Oracle error: {e}"})
                st.rerun()

            st.session_state["last_oracle_answer"] = answer

            # Optional market context fetch if explicitly requested.
            market = fetch_market_context(answer.question)
            internal_md = _internal_kpi_context(df, answer)
            if market.summary_markdown and internal_md:
                market = MarketResearchResult(
                    enabled=market.enabled,
                    summary_markdown=market.summary_markdown + "\n\n" + internal_md,
                    sources=market.sources,
                    benchmark_candidates=market.benchmark_candidates,
                )
            st.session_state["last_market_context"] = market

            accuracy = "N/A"
            if answer.forecast:
                if answer.forecast.accuracy_r2 is not None:
                    accuracy = f"R^2={answer.forecast.accuracy_r2:.2f}"
                elif answer.forecast.accuracy_mape is not None:
                    accuracy = f"MAPE={answer.forecast.accuracy_mape:.1%}"

            forecast_lines = ""
            if answer.forecast is not None:
                try:
                    rows = answer.forecast.forecast.copy()
                    rows["date"] = pd.to_datetime(rows["date"]).dt.strftime("%Y-%m")
                    forecast_lines = "\n".join([f"  - {r.date}: {r.prediction:.4g}" for r in rows.itertuples()])
                except Exception:
                    forecast_lines = ""

            assistant_md = (
                f"**Research**\n"
                f"- Data Trust Score: {answer.trust_score:.0%} ({answer.diagnosis})\n"
                f"- Generated Pandas logic (see below)\n\n"
                f"**Analyze**\n"
                f"- {answer.descriptive_insight}\n\n"
                f"**Advise**\n"
                f"- Next 3 months forecast: {('available' if answer.forecast else 'unavailable')} (Accuracy: {accuracy})\n"
            )
            if forecast_lines:
                assistant_md += forecast_lines + "\n"
            assistant_md += (
                f"- Strategic next steps:\n"
                + "\n".join([f"  - {s}" for s in answer.strategic_next_steps])
            )
            if market.summary_markdown:
                assistant_md += "\n\n" + market.summary_markdown

            st.session_state["chat_messages"].append(
                {"role": "assistant", "content": assistant_md, "code": answer.pandas_code}
            )

            # Memory thread: store compact, verified summary.
            st.session_state["memory_thread"].append(
                {
                    "asked_at": answer.asked_at.strftime("%Y-%m-%d %H:%M"),
                    "question": answer.question,
                    "descriptive": answer.descriptive_insight,
                    "trust_score": round(answer.trust_score, 4),
                    "diagnosis": answer.diagnosis,
                    "next_steps": answer.strategic_next_steps,
                    "verification": answer.verification_notes[:3],
                }
            )
            _save_memory(st.session_state["memory_thread"])
            st.rerun()
