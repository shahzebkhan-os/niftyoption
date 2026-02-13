import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class RegimeEnsemble:
    def __init__(self, model_registry):
        self.registry = model_registry

    def predict(self, features, regime, regime_confidence=1.0):
        """
        Performs confidence-weighted prediction by selecting the specialized regime model
        and blending its output with a neutral baseline (0.5).
        
        Args:
            features: list or array of feature values.
            regime: str, name of the detected regime.
            regime_confidence: float [0, 1], how certain the classifier is about the regime.
        """
        model = self.registry.get_model(regime)

        if model is None:
            # Fallback for unknown regimes or missing models
            logger.warning(f"No specialized model for regime '{regime}'. System in UNCERTAIN state, returning neutral baseline (0.5).")
            return 0.5 
            
        try:
            # Get probability from specialized model
            # Assumes binary classification, returns prob of class 1
            prob = model.predict_proba([features])[0][1]

            # Weight by regime confidence
            # If confidence is 1.0, we use the model's prob entirely.
            # If confidence is 0.0, we use 0.5 (maximum uncertainty/neutral).
            weighted_prob = prob * regime_confidence + 0.5 * (1 - regime_confidence)

            return float(weighted_prob)
        except Exception as e:
            logger.error(f"Ensemble prediction error for regime {regime}: {e}")
            return 0.5
