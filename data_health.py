import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class DataHealthMonitor:
    """
    Uses Random Forest to classify "Data Health" scores.
    Detects null spikes, schema drift, outliers.
    """
    
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.baseline_stats = {}
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for health scoring: null %, cardinality, skew, etc."""
        features = []
        
        # Null percentage
        null_pct = df.isnull().mean()
        null_pct_mean = null_pct.mean() if len(null_pct) > 0 else 0
        
        # Cardinality ratio (unique / total rows)
        card_ratio = df.nunique() / max(len(df), 1)
        card_ratio_mean = card_ratio.mean()
        
        # Skewness
        numeric_cols = df.select_dtypes(include=[np.number])
        skew_mean = numeric_cols.skew().fillna(0).mean() if not numeric_cols.empty else 0
        
        # Outlier detection (Z-score > 3)
        outliers_pct = []
        for col in numeric_cols.columns:
            col_data = numeric_cols[col].dropna()
            if len(col_data) > 0:
                z_scores = np.abs(stats.zscore(col_data))
                outliers_pct.append((z_scores > 3).mean())
        outliers_mean = np.mean(outliers_pct) if outliers_pct else 0
        
        # Schema stability (column count change from baseline)
        baseline_n_cols = self.baseline_stats.get('n_cols', len(df.columns))
        schema_change = abs(len(df.columns) - baseline_n_cols) / max(len(df.columns), 1)
        
        features = [null_pct_mean, card_ratio_mean, skew_mean, outliers_mean, schema_change]
        return np.array(features).reshape(1, -1)
    
    def train_baseline(self, baseline_dfs: list[pd.DataFrame], labels: list[int]):
        """Train on healthy (1) vs problematic (0) baseline datasets."""
        if not baseline_dfs:
            raise ValueError("No baseline DataFrames provided.")
        
# Aggregate baseline stats from healthy dfs
        healthy_count = sum(1 for label in labels if label == 1)
        healthy_n_cols = [len(df.columns) for df in baseline_dfs[:healthy_count]]
        healthy_n_rows = [len(df) for df in baseline_dfs[:healthy_count]]
        self.baseline_stats = {
            'n_cols': np.mean(healthy_n_cols),
            'n_rows': np.mean(healthy_n_rows)
        }
        
        all_features = []
        for df in baseline_dfs:
            all_features.append(self.extract_features(df))
        
        X = np.vstack(all_features)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, labels)
        self.is_trained = True
        logger.info("DataHealthMonitor trained on baseline.")
    
    def predict_health_score(self, df: pd.DataFrame) -> Tuple[float, str]:
        """Predict health score (0-1) and diagnosis."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_baseline first.")
        
        features = self.extract_features(df)
        features_scaled = self.scaler.transform(features)
        prob = self.model.predict_proba(features_scaled)[0][1]  # Prob of 'healthy'
        
        # Root cause diagnosis using decision tree logic
        diagnosis = self._diagnose_issue(df, features[0])
        
        logger.info(f"Data Trust Score: {prob:.2f} - {diagnosis}")
        return prob, diagnosis
    
    def _diagnose_issue(self, df: pd.DataFrame, features: np.ndarray) -> str:
        """Simple decision tree for root cause."""
        null_pct = features[0]
        outliers = features[3]
        
        if null_pct > 0.2:
            return "High Null Values - Source Connection Issue?"
        elif outliers > 0.1:
            return "Outlier Spike - Data Quality Issue"
        elif abs(len(df.columns) - self.baseline_stats.get('n_cols', 0)) > 1:
            return "Schema Drift Detected"
        else:
            return "Healthy"
    
    def check_integrity(self, df: pd.DataFrame, threshold: float = 0.7) -> bool:
        """Monitoring loop check."""
        score, _ = self.predict_health_score(df)
        return score >= threshold
