import logging
import pandas as pd
from typing import Dict, List, Any
from core.self_healing_df import SmartBIFrame
from core.insight_engine import InsightEngine
from core.decorators import SchemaGuard
from utils.safeguardai import sanitize_user_query

logger = logging.getLogger(__name__)

class SafeNLQParser:
    """
    Natural Language Query with SQL injection safety.
    LangChain-style wrapper sanitized for BI dashboard.
    """
    
    def __init__(self, df_schema: List[str]):
        self.schema = df_schema
        self.insight_engine = InsightEngine(df_schema)
        # self.sanitizer = self._build_sanitizer()  # Removed unused
    
    def _is_safe(self, query: str) -> bool:
        """Safety layer: Check for injection patterns."""
        try:
            sanitize_user_query(query)
            return True
        except ValueError:
            return False
    
    def parse_to_sql(self, nl_query: str, df: pd.DataFrame) -> str:
        """
        Convert NLQ to safe Pandas code via InsightEngine.
        """
        nl_query = sanitize_user_query(nl_query)
        insight_result = self.insight_engine.query(nl_query, df)
        return insight_result['pandas_code']
    
    def get_insights(self, nl_query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Full InsightEngine query: code + result + insight.
        """
        nl_query = sanitize_user_query(nl_query)
        return self.insight_engine.query(nl_query, df)
    
    @SchemaGuard
    def execute_query(self, code: str, df: pd.DataFrame) -> any:
        """Exec code via InsightEngine healing."""
        df = SmartBIFrame(df)
        result, errors = InsightEngine(self.schema).execute_with_healing(code, df)
        if errors:
            logger.warning(f"Healing attempts: {errors}")
        return result
