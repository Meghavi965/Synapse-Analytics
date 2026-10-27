import pandas as pd
import logging
from rapidfuzz import fuzz, process
from typing import Optional, Dict, Any, List
import re
from .auto_mapper import AutoMapper

logger = logging.getLogger(__name__)

# Reuse SQL injection patterns for column name sanitation
SQL_INJECTION_PATTERNS = [
    r'select.*from', r'drop table', r'update.*set', r'delete.*from',
    r'exec|execute|script|alert|onerror', r'union.*select'
]

# Context map: semantic hints used for fuzzy reconciliation.
# Values are free-form descriptions; AutoMapper uses them in scoring.
DEFAULT_CONTEXT_MAP: Dict[str, str] = {
    'revenue': 'sales amount, Sales_USD, Revenue_Amt, sales_amount, total_sales',
    'expenses': 'operating expenses, Expenses, Cost, operating_expenses',
    'profit': 'profit, net profit, gross profit, Net_Profit, Gross_Profit',
}

# Whitelist of approved formulas to prevent bloat/hallucinations
WHITELIST_FORMULAS = {
    'revenue': "df['Sales_USD'] if 'Sales_USD' in df.columns else df['Revenue_Amt'] if 'Revenue_Amt' in df.columns else None",
    'Total_Profit': "df['revenue'] - df['Expenses']",
    'Total_Revenue': "df['revenue']",
    'Margin': "(df['revenue'] - df['Expenses']) / df['revenue']",
    'Profit_Margin': "(df['revenue'] - df['Expenses']) / df['revenue'] * 100",
    'Net_Profit': "df['Gross_Profit'] - df['Taxes']",
    # Add more domain-specific as needed
}

class SelfHealingDataFrame(pd.DataFrame):
    """
    Self-healing Pandas DataFrame with semantic column reconciliation.
    Automatically remaps missing attributes using Levenshtein similarity.
    """

    _metadata = ['heal_log', '_context_map', 'last_heal_event']

    def __init__(self, data=None, *args, context_map: Optional[Dict[str, str]] = None, **kwargs):
        if context_map is None and 'context_map' in kwargs:
            context_map = kwargs.pop('context_map')
        else:
            kwargs.pop('context_map', None)

        super().__init__(data=data, *args, **kwargs)
        self.heal_log = []
        self.last_heal_event = None
        merged_context = DEFAULT_CONTEXT_MAP.copy()
        merged_context.update(context_map or {})
        self._context_map = merged_context
    
    @property
    def _constructor(self):
        return SelfHealingDataFrame
    
    def __getitem__(self, key):
        """
        Override to trigger self-healing on KeyError.
        """
        try:
            return super().__getitem__(key)
        except (KeyError, AttributeError):
            return self._handle_missing_column(key)
    
    def _handle_missing_column(self, key):
        missing_key = str(key)
        logger.info(f"Missing column '{missing_key}', attempting smart heal...")
        
        # 1. Try fuzzy alias via AutoMapper
        mapper = AutoMapper(similarity_threshold=0.75)
        mappings = mapper.map_columns(self, [missing_key], context_dict=self._context_map)
        if mappings:
            match = mappings[missing_key]
            self[missing_key] = super().__getitem__(match)
            self.last_heal_event = {
                "requested": missing_key,
                "mapped_to": match,
                "method": "semantic_context",
            }
            self.heal_log.append(f"Alias heal: '{match}' → '{missing_key}'")
            logger.info(self.heal_log[-1])
            return super().__getitem__(missing_key)
        
        # 2. Try derivation from whitelist
        formula = WHITELIST_FORMULAS.get(missing_key)
        if formula:
            try:
                derived_series = self._execute_safe_formula(formula)
            except Exception:
                derived_series = None

            if derived_series is not None:
                self[missing_key] = derived_series
                self.heal_log.append(f"Derived: '{missing_key}' from whitelist formula")
                logger.info(self.heal_log[-1])
                return super().__getitem__(missing_key)
        
        # 3. Clarification fallback
        schema_md = self._get_schema_metadata()
        raise KeyError(
            f"Clarification Request: Column '{missing_key}' not found/derivable. "
            f"Available: {list(self.columns)}. Closest alias: None. "
            f"Schema: {schema_md}"
        )
    
    def find_closest_attribute(self, requested: str, threshold: float = 0.85) -> tuple[Optional[str], float]:
        """
        Find closest column using Levenshtein distance (fuzz.ratio).
        Returns (match, score) or (None, 0.0).
        """
        if not self._is_safe_column_name(requested):
            raise ValueError(f"Unsafe column name '{requested}' blocked.")
        
        if requested in self.columns:
            return requested, 1.0
        
        best = process.extractOne(
            requested,
            self.columns.tolist(),
            scorer=fuzz.ratio,
        )
        if not best:
            return None, 0.0

        best_match = best[0]
        score = float(best[1]) / 100.0
        return best_match if score >= threshold else None, score
    
    def _is_safe_column_name(self, col_name: str) -> bool:
        """Sanitation: Block malicious column names."""
        col_lower = col_name.lower()
        return not any(re.search(pattern, col_lower) for pattern in SQL_INJECTION_PATTERNS)
    
    def _get_schema_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Graceful fallback: DB-like schema describe."""
        schema = {}
        for col in self.columns:
            dtype = str(self[col].dtype)
            sample = self[col].dropna().iloc[0] if len(self[col].dropna()) > 0 else None
            schema[col] = {
                'dtype': dtype,
                'sample_value': sample,
                'null_pct': self[col].isnull().mean(),
                'unique_count': self[col].nunique()
            }
        return schema
    
    def get_heal_log(self) -> list[str]:
        return self.heal_log.copy()

    def _can_derive(self, formula: str) -> bool:
        """Check if all required columns exist in formula."""
        cols_in_formula = re.findall(r"df\['([^']+)'\]", formula)
        return all(col in self.columns for col in cols_in_formula)

    def _execute_safe_formula(self, formula: str) -> pd.Series:
        """Safely exec formula in restricted namespace."""
        safe_namespace = {'df': self, 'pd': pd}
        try:
            result = eval(formula, {"__builtins__": {}}, safe_namespace)
            if result is None:
                raise ValueError("Formula returned None")
            if isinstance(result, pd.Series):
                return result.reindex(self.index)
            return pd.Series(result, index=self.index)
        except Exception as e:
            logger.error(f"Formula execution failed: {e}")
            raise ValueError(f"Cannot derive column: {e}")

class SmartBIFrame(SelfHealingDataFrame):
    """
    Virtual Semantic Layer for BI: Smart column resolution with aliasing, derivations, whitelisting.
    Wraps raw_df; overrides __getitem__ to prevent ValueErrors in visualizations.
    """
    
    def __init__(self, data=None, *args, context_map: Optional[Dict[str, str]] = None, **kwargs):
        if context_map is None and 'context_map' in kwargs:
            context_map = kwargs.pop('context_map')
        else:
            kwargs.pop('context_map', None)

        super().__init__(data=data, *args, context_map=context_map, **kwargs)
    
    @property
    def _constructor(self):
        return SmartBIFrame
    
    def __getitem__(self, key):
        """Enhanced for BI: alias → derive → clarify."""
        if isinstance(key, list):
            # For multiple columns, create virtual view
            return self.get_virtual_df(self, key)
        return super().__getitem__(key)

    def __getattr__(self, name: str):
        """
        Attribute access override:
        If a column is missing (e.g., df.revenue), attempt fuzzy/context healing.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name.startswith('_') or not self._is_safe_column_name(name):
                raise
            try:
                return self[name]
            except KeyError as e:
                raise AttributeError(str(e)) from e
    
    def get_virtual_df(self, raw_df: pd.DataFrame, requested_columns: List[str]) -> pd.DataFrame:
        """
        Virtual wrapper: alias missing cols, derive from whitelist if possible.
        """
        virtual_df = raw_df.copy()
        missing = [col for col in requested_columns if col not in virtual_df.columns]
        
        if missing:
            mapper = AutoMapper(similarity_threshold=0.75)
            mappings = mapper.map_columns(virtual_df, missing, context_dict=self._context_map)
            
            # Apply aliases
            for target, source in mappings.items():
                if target not in virtual_df.columns and source in virtual_df.columns:
                    virtual_df[target] = virtual_df[source]
            
            # Derive remaining
            for col in missing:
                if col not in virtual_df.columns:
                    formula = WHITELIST_FORMULAS.get(col)
                    if formula:
                        try:
                            virtual_df[col] = virtual_df._execute_safe_formula(formula)
                            continue
                        except Exception:
                            pass

                    schema_md = virtual_df._get_schema_metadata()
                    raise ValueError(
                        f"Clarification Request for '{col}': No alias/derivation. "
                        f"Available: {list(virtual_df.columns)}. Schema: {schema_md}"
                    )
        
        selected_columns = [col for col in requested_columns if col in virtual_df.columns]
        # Bypass SmartBIFrame.__getitem__ to avoid recursion when selecting a list of columns.
        return pd.DataFrame.__getitem__(virtual_df, selected_columns)
    
    def _can_derive(self, formula: str) -> bool:
        return super()._can_derive(formula)
    
    def _execute_safe_formula(self, formula: str) -> pd.Series:
        return super()._execute_safe_formula(formula)
