import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.agent import SelfHealingBIAgent
from core.self_healing_df import SmartBIFrame
from utils.safeguardai import sanitize_user_query

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DashboardConfig:
    trust_threshold: float = 0.7
    target_schema: tuple[str, ...] = ("region", "revenue", "churn", "customer_age")
    context_map: Optional[dict[str, str]] = None  # set in __post_init__

    def __post_init__(self):
        if self.context_map is None:
            object.__setattr__(
                self,
                "context_map",
                {"revenue": "sales amount or Sales_USD or Revenue_Amt"},
            )


CONFIG = DashboardConfig()


def _toast(message: str) -> None:
    toast_fn = getattr(st, "toast", None)
    if callable(toast_fn):
        toast_fn(message)
    else:
        st.success(message)


def _load_demo_data(n: int = 1000, seed: int = 42) -> SmartBIFrame:
    np.random.seed(seed)
    data = {
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "Sales_USD": np.random.normal(10000, 2000, n),
        "Expenses": np.random.normal(7000, 1500, n),
        "churn": np.random.choice([0, 1], n, p=[0.85, 0.15]),
        "customer_age": np.random.randint(18, 80, n),
    }
    return SmartBIFrame(pd.DataFrame(data), context_map=CONFIG.context_map)


def _get_agent() -> SelfHealingBIAgent:
    if "agent" not in st.session_state:
        st.session_state["agent"] = SelfHealingBIAgent()
    return st.session_state["agent"]


def _get_df() -> SmartBIFrame:
    if "df" not in st.session_state:
        st.session_state["df"] = _load_demo_data()
        st.session_state["healing_attempted"] = False
    df = st.session_state["df"]
    if not isinstance(df, SmartBIFrame):
        df = SmartBIFrame(df, context_map=CONFIG.context_map)
        st.session_state["df"] = df
    return df


def _train_baseline(agent: SelfHealingBIAgent, df: SmartBIFrame) -> None:
    healthy = [df]
    problematic_df = df.copy()
    if "Sales_USD" in problematic_df.columns:
        problematic_df.loc[::10, "Sales_USD"] = np.nan
    if "customer_age" in problematic_df.columns:
        problematic_df = problematic_df.drop("customer_age", axis=1)
    problematic = [problematic_df]
    agent.register_baseline(healthy, problematic)


def _predict_health(agent: SelfHealingBIAgent, df: SmartBIFrame) -> tuple[float, str]:
    score = 0.0
    diagnosis = "Monitor Untrained"
    if getattr(agent.health_monitor, "is_trained", False):
        try:
            score, diagnosis = agent.health_monitor.predict_health_score(df)
        except Exception as e:
            logger.error("Health prediction failed: %s", e)
            score = 0.0
            diagnosis = "Prediction Error"
    return score, diagnosis


def _predictive_advice(agent: SelfHealingBIAgent, df: SmartBIFrame, score: float, diagnosis: str) -> list[str]:
    advice: list[str] = []
    try:
        features = agent.health_monitor.extract_features(df)[0]
        null_pct, _, _, outliers_pct, schema_change = (float(x) for x in features)
    except Exception:
        null_pct, outliers_pct, schema_change = 0.0, 0.0, 0.0

    if score >= CONFIG.trust_threshold and diagnosis == "Healthy":
        advice.append("Data looks healthy; proceed with analysis.")
        return advice

    if "Null" in diagnosis or null_pct > 0.2:
        advice.append("Null spike detected; validate source refresh and consider imputations.")
    if "Outlier" in diagnosis or outliers_pct > 0.1:
        advice.append("Outlier spike detected; review distribution caps/winsorization.")
    if "Schema Drift" in diagnosis or schema_change > 0.0:
        advice.append("Schema drift suspected; semantic mapping can reconcile renamed fields.")
    if not advice:
        advice.append("Run healing pipeline and re-check the trust score.")
    return advice


def _auto_heal(agent: SelfHealingBIAgent, df: SmartBIFrame, score: float) -> tuple[SmartBIFrame, Optional[str]]:
    if score >= CONFIG.trust_threshold:
        return df, None

    repaired_df = None
    try:
        repaired_df = agent.repair_pipeline(
            df,
            list(CONFIG.target_schema),
            CONFIG.context_map,
            threshold=CONFIG.trust_threshold,
        )
    except Exception as e:
        logger.error("Repair failed: %s", e)
        repaired_df = None

    if repaired_df is None:
        return df, None

    mapped_source = agent.last_mappings.get("revenue")
    if mapped_source:
        msg = f"✅ Healing Complete: Mapped 'revenue' to '{mapped_source}' using semantic context."
    else:
        msg = "✅ Healing Complete: Semantic repair applied."
    return SmartBIFrame(repaired_df, context_map=CONFIG.context_map), msg


def _is_describe_series(series: pd.Series) -> bool:
    keys = {str(k).strip().lower() for k in series.index}
    expected = {"count", "mean", "std", "min", "25%", "50%", "75%", "max"}
    return expected.issubset(keys)


def _build_graphical_proof(insights: dict[str, Any], df: SmartBIFrame):
    result = insights.get("result")
    intent = insights.get("intent") or {}
    metric = (intent.get("metrics") or [None])[0]

    if result is None:
        return None

    if isinstance(result, pd.Series):
        if _is_describe_series(result) and metric and metric in df.columns:
            return px.histogram(df, x=metric, nbins=30, title=f"{metric} distribution")

        tmp = result.reset_index()
        tmp.columns = ["group", "value"]
        if pd.api.types.is_datetime64_any_dtype(tmp["group"]):
            return px.line(tmp, x="group", y="value", title=f"{metric or 'Metric'} over time")
        return px.bar(tmp, x="group", y="value", title=f"{metric or 'Metric'} by group")

    if isinstance(result, pd.DataFrame):
        tmp = result.reset_index()
        numeric_cols = tmp.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            return None
        y = "mean" if "mean" in numeric_cols else numeric_cols[0]
        x = [c for c in tmp.columns if c != y][0]
        return px.bar(tmp, x=x, y=y, title=f"{metric or y} (graphical proof)")

    if metric and metric in df.columns:
        return px.histogram(df, x=metric, nbins=30, title=f"{metric} distribution")

    return None


def run_dashboard() -> None:
    st.set_page_config(page_title="Self-Healing Agentic Dashboard", layout="wide")
    logging.basicConfig(level=logging.INFO)

    agent = _get_agent()
    df = _get_df()
    try:
        _ = df.revenue
    except Exception:
        pass

    # Auto-train once for the demo so health checks always work.
    if not getattr(agent.health_monitor, "is_trained", False) and not st.session_state.get("auto_trained"):
        _train_baseline(agent, df)
        st.session_state["auto_trained"] = True
        _toast("✅ Baseline trained (demo).")

    # Global health status (used for the predictive panel + auto-heal trigger).
    score, diagnosis = _predict_health(agent, df)

    # Auto-heal at most once per df-state unless user forces it.
    if score < CONFIG.trust_threshold and not st.session_state.get("healing_attempted", False):
        healed_df, heal_msg = _auto_heal(agent, df, score)
        st.session_state["healing_attempted"] = True
        if heal_msg:
            _toast(heal_msg)
        st.session_state["df"] = healed_df
        df = healed_df
        score, diagnosis = _predict_health(agent, df)

    st.title("Self-Healing Agentic Dashboard")
    st.caption("SmartBIFrame + ML health monitor + SafeGuardAI + NLQ → Pandas → Graphical proof")

    left, right = st.columns([2, 1], gap="large")

    with left:
        metric_col, diag_col = st.columns([1, 2])
        metric_col.metric("Data Trust Score", f"{score:.0%}")
        diag_col.write(f"**Diagnosis:** {diagnosis}")

        with st.expander("Predictive Advice", expanded=True):
            for line in _predictive_advice(agent, df, score, diagnosis):
                st.write(f"- {line}")

        st.subheader("Core Charts")
        try:
            _ = df.revenue  # attribute-based self-heal path
        except Exception:
            pass
        heal_event = getattr(df, "last_heal_event", None)
        if isinstance(heal_event, dict) and st.session_state.get("last_heal_toast") != heal_event:
            requested = heal_event.get("requested")
            mapped_to = heal_event.get("mapped_to")
            if requested and mapped_to:
                _toast(f"✅ Healing Complete: Mapped '{requested}' to '{mapped_to}' using semantic context.")
                st.session_state["last_heal_toast"] = heal_event

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            churn_series = df.groupby("region")["churn"].mean()
            st.plotly_chart(px.bar(churn_series, title="Regional churn"), use_container_width=True)
        with chart_col2:
            if "revenue" in df.columns:
                st.plotly_chart(px.histogram(df, x="revenue", title="Revenue distribution"), use_container_width=True)
            else:
                st.info("Revenue not available yet; ask a question or run healing.")

        st.subheader("Graphical Proof")
        last = st.session_state.get("last_insights")
        if isinstance(last, dict):
            fig = _build_graphical_proof(last, df)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No chart available for the last response.")
        else:
            st.info("Ask a question to generate a proof chart.")

        st.subheader("Data Preview")
        st.dataframe(df.head(200), use_container_width=True)

    with right:
        st.subheader("Conversational Sidebar")

        with st.expander("Agent Controls", expanded=True):
            if st.button("Retrain baseline model"):
                _train_baseline(agent, df)
                _toast("✅ Baseline retrained.")
            if st.button("Simulate schema drift"):
                drifted = df.rename(columns={"Sales_USD": "Revenue_Amt"})
                st.session_state["df"] = SmartBIFrame(drifted, context_map=CONFIG.context_map)
                st.session_state["healing_attempted"] = False
                _toast("⚠️ Drift injected: Sales_USD → Revenue_Amt")
                st.rerun()
            if st.button("Force heal now"):
                st.session_state["healing_attempted"] = False
                st.rerun()
            if st.button("Reset data"):
                st.session_state["df"] = _load_demo_data()
                st.session_state["healing_attempted"] = False
                st.session_state.pop("last_insights", None)
                _toast("🔄 Data reset.")
                st.rerun()
            if st.button("Clear chat"):
                st.session_state["chat_messages"] = []
                st.session_state.pop("last_insights", None)
                st.rerun()

        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = []

        for msg in st.session_state["chat_messages"]:
            role = msg.get("role", "assistant")
            with st.chat_message(role):
                st.markdown(msg.get("content", ""))
                if role == "assistant" and msg.get("pandas_code"):
                    st.code(msg["pandas_code"], language="python")

        prompt = st.chat_input("Ask a BI question (e.g., 'Show me regional churn')")
        if prompt:
            st.session_state["chat_messages"].append({"role": "user", "content": prompt})

            try:
                safe_prompt = sanitize_user_query(prompt)
            except ValueError as e:
                st.session_state["chat_messages"].append(
                    {"role": "assistant", "content": f"Blocked by SafeGuardAI: {e}"}
                )
                st.rerun()

            # Per-question health verification (explicit requirement).
            score_q, diagnosis_q = _predict_health(agent, df)
            if score_q < CONFIG.trust_threshold:
                healed_df, heal_msg = _auto_heal(agent, df, score_q)
                if heal_msg:
                    _toast(heal_msg)
                st.session_state["df"] = healed_df
                df = healed_df
                score_q, diagnosis_q = _predict_health(agent, df)

            insights = agent.query_insights(safe_prompt, df)
            st.session_state["last_insights"] = insights
            insight_text = insights.get("insight")
            errors = insights.get("errors") or []

            assistant_md = f"**Health Check:** {score_q:.0%} ({diagnosis_q})\n\n"
            if insight_text:
                assistant_md += f"**Insight:** {insight_text}\n\n"
            assistant_md += "**Pandas Logic:**\n\n**Graphical Proof:** Updated in the left panel."
            if errors:
                assistant_md += f"\n\n**Healing Notes:** {len(errors)} retry/repair signals captured."
            st.session_state["chat_messages"].append(
                {
                    "role": "assistant",
                    "content": assistant_md,
                    "pandas_code": insights.get("pandas_code", ""),
                }
            )
            st.rerun()
