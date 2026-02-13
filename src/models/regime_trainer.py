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
        Includes a stability audit across folds to detect overfitting.
        """
        # Filter data for regime only
        df_regime = df[df["regime"] == self.regime_name]
        
        if len(df_regime) < 100: 
            logger.warning(f"Insufficient data for regime {self.regime_name}.")
            return None

        X = df_regime[feature_columns]
        y = df_regime[target_column]

        # 1. Stability Audit (Manual Cross-Validation)
        tscv = TimeSeriesSplit(n_splits=5)
        fold_auc = []
        
        logger.info(f"Auditing stability for {self.regime_name}...")
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            fold_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, verbose=-1)
            fold_model.fit(X_tr, y_tr)
            
            from sklearn.metrics import roc_auc_score
            preds = fold_model.predict_proba(X_val)[:, 1]
            fold_auc.append(roc_auc_score(y_val, preds))
        
        auc_std = np.std(fold_auc)
        logger.info(f"Stability Audit: Mean AUC={np.mean(fold_auc):.4f}, Std={auc_std:.4f}")

        # Overfitting Defense: If variance across folds is too high, the model is unstable
        if auc_std > 0.15:
            logger.warning(f"REJECTED: Model stability too low (Std={auc_std:.4f}). High risk of overfitting.")
            return None

        # 2. Final Calibrated Training
        base_model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            importance_type='gain',
            verbose=-1
        )

        calibrated_model = CalibratedClassifierCV(
            base_model,
            method="isotonic",
            cv=tscv
        )

        calibrated_model.fit(X, y)

        model_path = os.path.join(self.model_dir, f"{self.regime_name}_model.pkl")
        joblib.dump(calibrated_model, model_path)
        
        return calibrated_model
