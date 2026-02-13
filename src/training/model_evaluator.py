import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def evaluate(self, model, df, feature_columns, target_column):
        """
        Evaluates a model on the provided dataframe using core systematic metrics.
        Returns a dictionary of metrics.
        """
        if df.empty:
            logger.warning("Evaluation dataframe is empty.")
            return {"auc": 0.5, "brier": 1.0, "sharpe_proxy": -1.0}

        X = df[feature_columns]
        y = df[target_column]

        try:
            # Model is expected to have predict_proba (CalibratedClassifierCV)
            # We take the probability of class 1 (up move)
            prob = model.predict_proba(X)[:, 1]

            auc = roc_auc_score(y, prob)
            brier = brier_score_loss(y, prob)

            # Sharpe Proxy: A simple measure of prediction edge vs uncertainty
            # If mean prob is significantly away from 0.5 with low std, edge is high.
            # This is a heuristic for 'Trade Quality'
            excess_prob = np.abs(prob - 0.5).mean()
            prob_vol = prob.std()
            sharpe_proxy = (excess_prob / prob_vol) if prob_vol > 0 else 0.0

            return {
                "auc": float(auc),
                "brier": float(brier),
                "sharpe_proxy": float(sharpe_proxy),
                "sample_size": len(df)
            }
        except Exception as e:
            logger.error(f"Evaluation error: {e}")
            return {"auc": 0.5, "brier": 1.0, "sharpe_proxy": -1.0}
