from sklearn.metrics import brier_score_loss
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class CalibrationMonitor:
    def compute_brier(self, y_true, y_prob):
        """
        Calculates the Brier score loss. 
        Lower is better (0.0 is perfect calibration).
        """
        if len(y_true) == 0:
            return 0.0
        return brier_score_loss(y_true, y_prob)

    def bucket_accuracy(self, y_true, y_prob):
        """
        Calculates the actual winrate across different probability buckets.
        Helps detect if a 70% prediction actually wins 70% of the time.
        """
        # We focus on the prediction side (probs > 0.5)
        buckets = np.linspace(0.5, 1.0, 6)
        results = {}

        y_true = np.array(y_true)
        y_prob = np.array(y_prob)

        for i in range(len(buckets) - 1):
            low = buckets[i]
            high = buckets[i+1]

            mask = (y_prob >= low) & (y_prob < high)

            if mask.sum() > 0:
                # Actual winrate in this probability bucket
                actual_winrate = y_true[mask].mean()
                results[f"{low:.2f}-{high:.2f}"] = {
                    "actual": float(actual_winrate),
                    "expected": float((low + high) / 2),
                    "count": int(mask.sum())
                }

        return results
