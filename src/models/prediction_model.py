import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import brier_score_loss, roc_auc_score
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_params=None):
        self.params = model_params or {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        }
        self.model = None
        self.calibrated_model = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the LightGBM model.
        """
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(self.params, train_data, num_boost_round=100)
        return self.model

    def train_calibrated(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains model and calibrates probabilities using Isotonic Regression or Platt Scaling.
        """
        # Split for calibration
        # Need a separate set for calibration to avoid overfitting the calibration
        split_idx = int(len(X) * 0.8)
        X_train, X_calib = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_calib = y.iloc[:split_idx], y.iloc[split_idx:]

        # Train Base Model
        base_learner = lgb.LGBMClassifier(**self.params, n_estimators=100)
        base_learner.fit(X_train, y_train)

        # Calibrate
        self.calibrated_model = CalibratedClassifierCV(base_learner, cv='prefit', method='isotonic')
        self.calibrated_model.fit(X_calib, y_calib)
        
        logger.info("Model trained and calibrated.")
        return self.calibrated_model

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns calibrated probabilities.
        """
        if self.calibrated_model:
            return self.calibrated_model.predict_proba(X)[:, 1]
        elif self.model:
             # raw prediction
            return self.model.predict(X)
        else:
            raise ValueError("Model not trained.")

    def walk_forward_validation(self, X: pd.DataFrame, y: pd.Series, n_splits=5):
        """
        Performs walk-forward validation (TimeSeriesSplit).
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            self.train_calibrated(X_train, y_train)
            preds = self.predict_proba(X_test)
            
            score = roc_auc_score(y_test, preds)
            brier = brier_score_loss(y_test, preds)
            scores.append({'auc': score, 'brier': brier})
            
            logger.info(f"Fold Score - AUC: {score:.4f}, Brier: {brier:.4f}")
            
        return scores

    def save_model(self, path):
        joblib.dump(self.calibrated_model, path)
    
    def load_model(self, path):
        self.calibrated_model = joblib.load(path)
