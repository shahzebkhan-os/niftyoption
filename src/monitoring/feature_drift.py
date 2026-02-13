import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import logging

logger = logging.getLogger(__name__)

class FeatureDriftMonitor:
    def ks_test(self, historical, live):
        """
        Performs the Kolmogorov-Smirnov test on two distributions.
        Returns the p-value. A small p-value (e.g., < 0.01) indicates 
        that the two samples come from different distributions.
        """
        if len(historical) < 20 or len(live) < 20:
            return 1.0 # Not enough data to conclude drift
            
        stat, p_value = ks_2samp(historical, live)
        return p_value

    def detect_drift(self, historical_df, live_df, columns):
        """
        Detects structural drift across a set of columns.
        """
        drift_report = {}

        for col in columns:
            if col not in historical_df.columns or col not in live_df.columns:
                logger.warning(f"Column '{col}' not found in dataframes for drift detection.")
                continue

            p_value = self.ks_test(
                historical_df[col].dropna(),
                live_df[col].dropna()
            )
            drift_report[col] = p_value
            
            if p_value < 0.01:
                logger.warning(f"DRIFT DETECTED: Feature '{col}' has shifted (p={p_value:.4f})")

        return drift_report
