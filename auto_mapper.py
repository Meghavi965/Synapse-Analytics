import pandas as pd
from rapidfuzz import fuzz
from typing import Dict, List, Optional
import logging
import re

logger = logging.getLogger(__name__)

class AutoMapper:
    """
    NLP-based Semantic Column Mapper for self-healing data pipelines.
    Uses string similarity (Levenshtein via rapidfuzz) + context to auto-remap drifted columns.
    """
    
    def __init__(self, similarity_threshold: float = 0.75, context_weight: float = 0.3):
        self.similarity_threshold = similarity_threshold
        self.context_weight = context_weight
        self.known_mappings: Dict[str, List[str]] = {
            'revenue': ['Sales_USD', 'Revenue_Amt', 'sales_amount', 'total_sales'],
            'expenses': ['Expenses', 'Cost', 'operating_expenses'],
            'profit': ['Profit', 'Net_Profit', 'Gross_Profit']
        }
    
    def compute_similarity(self, source_col: str, target_col: str, context: str = "") -> float:
        """Compute weighted similarity score."""
        def norm(text: str) -> str:
            return re.sub(r"[^a-z0-9]+", "", str(text).lower())

        source_norm = norm(source_col)
        target_norm = norm(target_col)
        string_sim = fuzz.ratio(source_norm, target_norm) / 100.0
        
        # Context similarity (e.g., description or sample data)
        context_sim = 0.0
        if context:
            ctx_norm = norm(context)
            context_sim = fuzz.partial_ratio(source_norm, ctx_norm) / 100.0

            # Extra boost: compare against context "alias" tokens (e.g., Sales_USD, Revenue_Amt).
            # This materially improves robustness against formatting drift like SalesUSD vs Sales_USD.
            tokens = [t.strip() for t in re.split(r"[,\n/|]+", str(context)) if t.strip()]
            alias_sim = 0.0
            for tok in tokens:
                tok_norm = norm(tok)
                if not tok_norm:
                    continue
                alias_sim = max(alias_sim, fuzz.ratio(source_norm, tok_norm) / 100.0)
            context_sim = max(context_sim, alias_sim)
        
        return (1 - self.context_weight) * string_sim + self.context_weight * context_sim
    
    def map_columns(self, source_df: pd.DataFrame, target_schema: List[str], context_dict: Dict[str, str] = None) -> Dict[str, str]:
        """
        Auto-map source columns to target schema.
        
        Example:
        source: "Sales_USD" -> target: "revenue" (high similarity)
        """
        mappings = {}
        context_dict = context_dict or {}
        
        for target_col in target_schema:
            best_match: Optional[str] = None
            score: float = 0.0

            # Check known mappings first
            if target_col in self.known_mappings:
                for alias in self.known_mappings[target_col]:
                    # Exact match
                    if alias in source_df.columns:
                        mappings[target_col] = alias
                        logger.info(f"Known mapping: '{alias}' -> '{target_col}'")
                        best_match = alias
                        score = 1.0
                        break

                    # Normalized match (handles SalesUSD vs Sales_USD, sales-usd vs Sales_USD, etc.)
                    alias_norm = re.sub(r"[^a-z0-9]+", "", str(alias).lower())
                    for col in source_df.columns:
                        col_norm = re.sub(r"[^a-z0-9]+", "", str(col).lower())
                        if col_norm and alias_norm and col_norm == alias_norm:
                            mappings[target_col] = str(col)
                            logger.info(f"Known mapping (normalized): '{col}' -> '{target_col}'")
                            best_match = str(col)
                            score = 1.0
                            break
                    if best_match is not None:
                        break

            if best_match is not None:
                continue
            
            # Fallback to fuzzy matching
            if best_match is None:
                context = context_dict.get(target_col, "")
                best_match, score = None, 0.0
                for candidate in list(source_df.columns):
                    cand_score = self.compute_similarity(candidate, target_col, context)
                    if cand_score > score:
                        best_match, score = candidate, cand_score
            
            if best_match is not None and score >= self.similarity_threshold:
                mappings[target_col] = best_match
                logger.info(f"Auto-mapped '{best_match}' -> '{target_col}' (score: {score:.2f})")
            else:
                logger.warning(f"No mapping found for '{target_col}' (best score: {score:.2f})")
        
        for target, source in mappings.items():
            alias_list = self.known_mappings.setdefault(target, [])
            if source not in alias_list:
                alias_list.append(source)
        return mappings
    
    def apply_mapping(self, df: pd.DataFrame, mappings: Dict[str, str]) -> pd.DataFrame:
        """Reorder/rename columns based on mappings. Safe if incomplete."""
        if not mappings:
            logger.warning("No mappings provided, returning original df.")
            return df.copy()
        
        # Rename only mapped columns
        rename_dict = {v: k for k, v in mappings.items()}
        mapped_df = df.rename(columns=rename_dict)
        
        # Reorder to target order if all targets mapped, else keep original order
        if len(mappings) == len(df.columns):
            mapped_df = mapped_df[list(mappings.keys())]
        
        logger.info(f"Applied {len(mappings)} mappings.")
        return mapped_df
