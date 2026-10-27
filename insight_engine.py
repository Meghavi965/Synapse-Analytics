import re
import logging
import pandas as pd
from typing import Dict, Any, List, Tuple
from .self_healing_df import SmartBIFrame
from .auto_mapper import AutoMapper

logger = logging.getLogger(__name__)

class InsightEngine:
    """
    Translates NL → Pandas with validation, self-healing execution, explanatory insights.
    VALIDATION: df.columns + SmartBIFrame virtuals ('revenue'→'Sales_USD').
    ACCURACY: parse→code→try/exec→rewrite on error.
    INSIGHTS: Results + \"Why\" correlations.
    """
    
    def __init__(self, df_schema: List[str]):
        self.schema = df_schema
        self.mapper = AutoMapper(similarity_threshold=0.75)
    
    def parse_intent(self, nl_query: str) -> Dict[str, Any]:
        """Step 1: Parse intent (Trend/Comparison/Summary)."""
        query_lower = nl_query.lower()

        # Intent classification (keyword rules, extensible)
        timeframe = None
        if any(phrase in query_lower for phrase in ["last month", "previous month", "past month"]):
            timeframe = "last_month"
        elif any(phrase in query_lower for phrase in ["last quarter", "previous quarter", "past quarter"]):
            timeframe = "last_quarter"
        elif re.search(r"\blast\s+3\s+months\b", query_lower) or "last three months" in query_lower:
            timeframe = "last_3_months"
        elif any(phrase in query_lower for phrase in ["last year", "past year", "previous year", "last 12 months"]):
            timeframe = "last_year"

        if timeframe is not None:
            intent_type = "trend"
        elif any(word in query_lower for word in ['trend', 'change', 'over time', 'growth']):
            intent_type = 'trend'
        elif any(word in query_lower for word in ['compare', 'vs', 'between', 'difference']):
            intent_type = 'comparison'
        else:
            intent_type = 'summary'
        
        # Extract entities (basic regex/keywords)
        metrics = re.findall(r'(revenue|sales|profit|churn|expenses?|price|rate)[s ]?', query_lower)
        metric_aliases = {
            "sales": "revenue",
            "rate": "price",
        }
        metrics = [metric_aliases.get(m, m) for m in metrics]
        groups = re.findall(r'(region|area|group|category|date|asset)[s ]?', query_lower)
        filters: Dict[str, Any] = {}  # e.g., 'region': 'North' via NER later

        # Asset detection for simple market datasets (gold/silver).
        asset_hits: list[str] = []
        if "gold" in query_lower:
            asset_hits.append("gold")
        if "silver" in query_lower:
            asset_hits.append("silver")
        if asset_hits:
            # If user mentions one asset, treat as a filter; if both, treat as a grouping hint.
            if len(asset_hits) == 1:
                filters["asset"] = asset_hits[0]
            else:
                if "asset" not in groups:
                    groups.append("asset")
            # If no metric was explicitly mentioned, default to price for market questions.
            if not metrics:
                metrics = ["price"]

        # If the user specified grouping (e.g., "by region"), treat it as an aggregation request.
        if intent_type == 'summary' and groups:
            intent_type = 'comparison'
        
        # Virtual column validation (SmartBIFrame whitelisting)
        if 'revenue' in metrics and 'revenue' not in self.schema:
            metrics = ['revenue']  # Triggers healing
        
        return {
            'type': intent_type,
            'metrics': list(set(metrics)),
            'groups': list(set(groups)),
            'filters': filters,
            'timeframe': timeframe,
        }
    
    def generate_pandas_code(self, intent: Dict[str, Any], df: SmartBIFrame) -> str:
        """Step 2: Generate validated Pandas code."""
        metrics = intent['metrics']
        # Default grouping depends on the dataset (region for BI, asset for markets).
        if intent["groups"]:
            groups = intent["groups"]
        elif "region" in df.columns:
            groups = ["region"]
        elif "asset" in df.columns:
            groups = ["asset"]
        else:
            groups = []
        valid_groups = [g for g in groups if g in df.columns]
        timeframe = intent.get("timeframe")
        
        # Schema validation + healing suggestions
        valid_metrics = []
        for m in metrics:
            if m in df.columns:
                valid_metrics.append(m)
            elif hasattr(df, "_context_map") and m in getattr(df, "_context_map", {}):
                # Allow virtual semantic metrics (e.g., 'revenue') and let SmartBIFrame heal at execution time.
                valid_metrics.append(m)
            else:
                # Suggest heal
                match, score = df.find_closest_attribute(m)
                if match:
                    valid_metrics.append(match)  # Use real col
                    logger.info(f"Healed '{m}' → '{match}' ({score:.2f})")
        
        metric = valid_metrics[0] if valid_metrics else None
        if not metric:
            return "df.describe()"

        # Apply lightweight filters (e.g., asset == 'gold').
        base_df = "df"
        filters = intent.get("filters") or {}
        if "asset" in filters and "asset" in df.columns:
            asset_val = filters["asset"]
            if isinstance(asset_val, (list, tuple, set)):
                items = ", ".join([repr(str(x)) for x in asset_val])
                base_df = f"df[df['asset'].isin([{items}])]"
            else:
                base_df = f"df[df['asset'] == {repr(str(asset_val))}]"

        if intent['type'] == 'trend':
            # Prefer time-series if a date column exists.
            if "date" in df.columns:
                m = str(metric).lower()
                is_price = ("price" in m) or ("rate" in m) or (m in {"close", "adj_close", "price_inr"})
                agg = "mean" if (m == "churn" or is_price) else "sum"
                code = (
                    f"{base_df}.assign(__month=pd.to_datetime({base_df}['date'], errors='coerce')"
                    ".dt.to_period('M').dt.to_timestamp())"
                    ".dropna(subset=['__month'])"
                    f".groupby('__month')['{metric}']." f"{agg}().sort_index()"
                )
                if timeframe in {"last_month", "last_3_months", "last_quarter"}:
                    # Keep a small window for executive readability + chart proof.
                    code = f"({code}).tail(6)"
                elif timeframe in {"last_year"}:
                    code = f"({code}).tail(14)"
            elif valid_groups:
                code = f"{base_df}.groupby({valid_groups})['{metric}'].mean().sort_values(ascending=False)"
            else:
                code = f"{base_df}['{metric}'].describe()"
        elif intent['type'] == 'comparison':
            if valid_groups:
                code = f"{base_df}.groupby({valid_groups})['{metric}'].agg(['mean', 'sum']).round(2)"
            else:
                code = f"{base_df}['{metric}'].describe()"
        else:  # summary
            code = f"{base_df}['{metric}'].describe()"

        return code
    
    def execute_with_healing(self, code: str, df: SmartBIFrame, max_retries: int = 3) -> Tuple[Any, List[str]]:
        """Step 3: Exec + self-heal on error."""
        safe_ns = {'df': df, 'pd': pd}
        errors = []
        
        for attempt in range(max_retries):
            # Pre-heal virtual/semantic columns referenced in the code (e.g., groupby()[\"revenue\"]).
            try:
                if hasattr(df, "_context_map"):
                    referenced = set(re.findall(r"\[['\"]([^'\"]+)['\"]\]", code))
                    for col in referenced:
                        if col not in df.columns and col in getattr(df, "_context_map", {}):
                            try:
                                _ = df[col]
                            except Exception:
                                pass
            except Exception:
                pass

            try:
                result = eval(code, {"__builtins__": {}}, safe_ns)
                return result, errors
            except Exception as e:
                error_str = str(e)
                errors.append(error_str)
                logger.warning(f"Attempt {attempt+1} failed: {error_str}")
                
                # Rewrite: Handle KeyError (col missing)
                if 'KeyError' in str(type(e)) or "'" in error_str:
                    col = error_str.split("'")[1] if "'" in error_str else None
                    if col and col in code:
                        match, score = df.find_closest_attribute(col)
                        if match:
                            healed_code = code.replace(f"'{col}'", f"'{match}'").replace(f'"{col}"', f'"{match}"')
                            code = healed_code  # Retry with heal
                            continue
                
                # Other errors: Simplify
                if 'NameError' in str(type(e)):
                    code = code.replace('pct_change()', 'diff() / df.shift(1)').replace('rolling', 'mean()')
        
        return None, errors
    
    def generate_insight(self, result: Any, df: SmartBIFrame, intent: Dict[str, Any]) -> str:
        """Step 4: Explain with 'Why'."""
        if result is None:
            return "Query failed after healing attempts."
        
        metrics = intent['metrics'][0] if intent['metrics'] else 'metric'
        groups = intent['groups'][0] if intent['groups'] else 'group'
        
        if isinstance(result, pd.Series):
            top = result.nlargest(1).index[0]
            delta = (result.max() - result.mean()) / result.mean() * 100 if result.mean() > 0 else 0
            insight = f"**Top {metrics} in {top}: +{delta:.1f}%**. "
        elif isinstance(result, pd.DataFrame):
            if 'mean' in result.columns:
                top_change = result['mean'].nlargest(1)
                insight = f"**{metrics} avg: {top_change.values[0]:.2f}** (top: {top_change.index[0]})."
            else:
                insight = f"**Summary**: {result.to_dict()}"
        else:
            insight = f"**Result**: {result}"
        
        # Why: Simple correlation
        try:
            corr = df.corr(numeric_only=True)
            if 'churn' in corr.columns and metrics.lower() in corr.columns:
                anti_corr = corr[metrics.lower()]['churn']
                insight += f" Why? Low churn correlates with high {metrics} (r={anti_corr:.2f})."
        except:
            pass
        
        return insight
    
    def query(self, nl_query: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Full pipeline: NL → Pandas → Heal → Insight."""
        # Preserve semantic context map if already wrapped.
        if isinstance(df, SmartBIFrame):
            df = df
        else:
            ctx = getattr(df, "_context_map", None)
            df = SmartBIFrame(df, context_map=ctx)
        intent = self.parse_intent(nl_query)
        code = self.generate_pandas_code(intent, df)
        result, errors = self.execute_with_healing(code, df)
        insight = self.generate_insight(result, df, intent)
        
        return {
            'pandas_code': code,
            'result': result,
            'insight': insight,
            'errors': errors,
            'intent': intent
        }

