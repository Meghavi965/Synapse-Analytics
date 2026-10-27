import streamlit as st

from executive_suite_app import run_executive_suite

run_executive_suite()
st.stop()
import pandas as pd
import numpy as np
import plotly.express as px
import logging
import time
from core.agent import SelfHealingBIAgent
from core.data_health import DataHealthMonitor
from utils.nlq_parser import SafeNLQParser
from core.self_healing_df import SmartBIFrame
from core.decorators import SchemaGuard

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Self-Healing BI Agent", layout="wide")

class PredictiveDashboard:
    def __init__(self):
        self.agent = SelfHealingBIAgent()
        self.nlq_parser = None
        self.sample_data = SmartBIFrame(self._load_sample_data())
    
    def _load_sample_data(self):
        """Demo dataset with sales/churn."""
        np.random.seed(42)
        n = 1000
        data = {
            'region': np.random.choice(['North', 'South', 'East', 'West'], n),
            'Sales_USD': np.random.normal(10000, 2000, n),
            'Expenses': np.random.normal(7000, 1500, n),
            'churn': np.random.choice([0, 1], n, p=[0.85, 0.15]),
            'customer_age': np.random.randint(18, 80, n)
        }
        # Sample data ready for self-healing tests
        return pd.DataFrame(data)
    
    def train_baseline(self):
        """Train on sample healthy/problematic data."""
        try:
            healthy = [self.sample_data]
            problematic_df = self.sample_data.copy()
            problematic_df.loc[::10, 'Sales_USD'] = np.nan  # nulls
            problematic_df = problematic_df.drop('customer_age', axis=1)  # schema drift
            problematic = [problematic_df]
            self.agent.register_baseline(healthy, problematic)
            st.success("✅ Baseline trained!")
        except Exception as e:
            st.error(f"Training failed: {e}")
    
    def run(self):
        st.title("🛠️ Self-Healing BI Agent")
        st.markdown("**SAP-inspired dashboard with autonomous data repair & NLQ**")
        
        # Header: Data Trust Score
        col1, col2 = st.columns([3, 1])
        with col1:
            score_placeholder = st.empty()
        with col2:
            st.metric("Data Trust Score", "92%", "↗️ +2%")
        
        # Healing status toast
        healing_status = st.empty()
        
        # Sidebar
        with st.sidebar:
            st.header("Agent Controls")
            if st.button("🚀 Train Baseline Model"):
                self.train_baseline()
            
            monitor_interval = st.slider("Monitor Interval (s)", 5, 60, 30)
            st.info("👉 Use NLQ below for insights!")
        
        # Simulate monitoring & healing
        df = SmartBIFrame(self.sample_data.copy())
        if st.button("Simulate Schema Drift"):
            # Drift: rename column (assign back)
            df = df.rename(columns={'Sales_USD': 'Revenue_Amt'})
            st.rerun()
        
        # Ensure Expenses always available for derivations
        if 'Expenses' not in df.columns:
            df['Expenses'] = np.random.normal(7000, 1500, len(df))
        
# Health check & repair
        # Initialize with safe defaults
        score = 0.0
        diagnosis = "Monitor Untrained"

        # Health monitoring (only if trained)
        if hasattr(self.agent.health_monitor, 'is_trained') and self.agent.health_monitor.is_trained:
            try:
                score, diagnosis = self.agent.health_monitor.predict_health_score(df)
            except Exception as e:
                logger.error(f"Health prediction failed: {e}")
                score = 0.0
                diagnosis = "Prediction Error"

        # Update display
        score_placeholder.metric("Data Trust Score", f"{score:.1%}", delta="1.2%" if score > 0 else None)

        # Repair pipeline (independent of health monitor)
        target_schema = ['region', 'revenue', 'churn', 'customer_age']
        context = {'revenue': 'sales amount or Sales_USD or Revenue_Amt'}

        try:
            repaired_df = self.agent.repair_pipeline(df, target_schema, context, threshold=0.5)
        except Exception as e:
            logger.error(f"Repair failed: {e}")
            repaired_df = None

        if repaired_df is not None:
            healing_status.success("✅ Healing Complete!")
            df = SmartBIFrame(repaired_df)
        elif score < 0.7:
            healing_status.error(f"⚠️ Healing Needed: {diagnosis}")
        
        # Init NLQ
        if self.nlq_parser is None and len(df.columns) > 0:
            self.nlq_parser = SafeNLQParser(list(df.columns))
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs(["📊 Data", "🔍 NLQ Insights", "⚙️ Agent Logs"])
        
        with tab1:
            st.dataframe(df, use_container_width=True)
            
            # Pre-heal columns to trigger self-healing before plotly validation
            _ = df['revenue']  # Triggers mapping if needed
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(df.groupby('region')['churn'].mean(), title="Regional Churn")
                st.plotly_chart(fig)
            with col2:
                fig2 = px.histogram(df, x='revenue', title="Revenue (Virtual Layer Test)")
                st.plotly_chart(fig2)
        
        with tab2:
            if self.nlq_parser:
                nl_query = st.text_input("Ask in natural language...", "Show me regional churn")
                if st.button("Query Data"):
                    insights = self.nlq_parser.get_insights(nl_query, df)
                    st.code(insights['pandas_code'])
                    st.dataframe(insights['result'])
                    st.markdown("### 🧠 Insight")
                    st.markdown(insights['insight'])
                    if insights['errors']:
                        st.warning(f"Healing log: {insights['errors']}")
                    st.json({"intent": insights['intent']})
            else:
                st.info("👈 Train baseline first to enable NLQ.")
        
        with tab3:
            st.text("Agent ready for production monitoring.")

if __name__ == "__main__":
    dashboard = PredictiveDashboard()
    dashboard.run()
