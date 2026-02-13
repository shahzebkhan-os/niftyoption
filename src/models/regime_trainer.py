import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class RegimeTrainer:
    def __init__(self, regime_name, model_dir="models"):
        self.regime_name = regime_name
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def train(self, df, feature_columns, target_column):
        """
        Trains and calibrates a model specifically for the given regime.
        """
        # Filter data for regime only
        df_regime = df[df["regime"] == self.regime_name]
        
        if len(df_regime) < 100: # Practical threshold for regime-specific training
            logger.warning(f"Insufficient data for regime {self.regime_name} ({len(df_regime)} rows). Skipping.")
            return None

        logger.info(f"Training specialized model for regime: {self.regime_name} ({len(df_regime)} rows)")

        X = df_regime[feature_columns]
        y = df_regime[target_column]

        # TimeSeriesSplit is critical for financial data to prevent leakage
        tscv = TimeSeriesSplit(n_splits=5)

        base_model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            importance_type='gain',
            verbose=-1
        )

        # Force categorical treatment of 'regime_encoded' if it's in feature_columns
        # but usually it's better to exclude it from specialized models as they already know their regime.
        
        calibrated_model = CalibratedClassifierCV(
            base_model,
            method="isotonic",
            cv=tscv
        )

        calibrated_model.fit(X, y)

        model_path = os.path.join(self.model_dir, f"{self.regime_name}_model.pkl")
        joblib.dump(calibrated_model, model_path)
        logger.info(f"Saved specialized model for {self.regime_name} to {model_path}")

        return calibrated_model
