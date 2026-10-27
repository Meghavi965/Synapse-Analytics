import time
import logging
import pandas as pd
from typing import Callable, Any, Dict, List
from .data_health import DataHealthMonitor
from .auto_mapper import AutoMapper
from .self_healing_df import SmartBIFrame
from .insight_engine import InsightEngine

logger = logging.getLogger(__name__)

class SelfHealingBIAgent:
    """
    Agentic logic for autonomous ETL repair and insights.
    """
    
    def __init__(self):
        self.health_monitor = DataHealthMonitor()
        self.mapper = AutoMapper()
        self.insight_engine = InsightEngine([])
        self.is_monitoring = False
        self.last_mappings: Dict[str, str] = {}
    
    def register_baseline(self, healthy_dfs, problematic_dfs):
        """Train health model."""
        baseline_dfs = healthy_dfs + problematic_dfs
        labels = [1] * len(healthy_dfs) + [0] * len(problematic_dfs)
        self.health_monitor.train_baseline(baseline_dfs, labels)
    
    def repair_pipeline(self, source_df, target_schema: list, context_dict: dict = None, threshold: float = 0.5) -> Any:
        """Autonomous repair: map + validate."""
        logger.info("Healing pipeline...")
        
        source_df = SmartBIFrame(source_df)
        
        # Semantic mapping
        mappings = self.mapper.map_columns(source_df, target_schema, context_dict)
        self.last_mappings = mappings.copy()
        repaired_df = self.mapper.apply_mapping(source_df, mappings)
        
        # Health check (lower threshold)
        score = 0.0
        diagnosis = "Monitor Untrained"
        if getattr(self.health_monitor, 'is_trained', False):
            score, diagnosis = self.health_monitor.predict_health_score(repaired_df)
            if score < threshold:
                logger.warning(f"Repair incomplete: {diagnosis} (score: {score:.2f})")
                return None
        else:
            logger.warning("Health monitor is untrained; skipping post-repair score gate.")
        
        logger.info("Pipeline repaired successfully.")
        return SmartBIFrame(repaired_df)
    
    def start_monitoring(self, data_fetcher: Callable[[], pd.DataFrame], interval: int = 300):
        """Periodic monitoring loop."""
        self.is_monitoring = True
        while self.is_monitoring:
            try:
                df = data_fetcher()
                if not self.health_monitor.check_integrity(df):
                    logger.warning("Data integrity issue detected. Triggering repair...")
                    repaired = self.repair_pipeline(df, expected_schema=list(df.columns))
                    if repaired is not None:
                        logger.info("Repair complete.")
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def stop_monitoring(self):
        self.is_monitoring = False

    def query_insights(self, nl_query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        InsightEngine query via agent: NL -> Pandas + heal + insight.
        """
        # Preserve any existing semantic context map when wrapping.
        if isinstance(df, SmartBIFrame):
            df = df
        else:
            ctx = getattr(df, "_context_map", None)
            df = SmartBIFrame(df, context_map=ctx)
        self.insight_engine.schema = list(df.columns)  # Update schema
        return self.insight_engine.query(nl_query, df)
