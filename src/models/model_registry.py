import joblib
import os
import logging

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.models = {}
        self.load_models()

    def load_models(self):
        """
        Loads all regime-specific models from the models directory.
        """
        if not os.path.exists(self.model_dir):
            logger.warning(f"Model directory '{self.model_dir}' not found.")
            return

        for file in os.listdir(self.model_dir):
            if file.endswith("_model.pkl"):
                # Extract regime name (e.g., TRENDING_UP from TRENDING_UP_model.pkl)
                regime = file.replace("_model.pkl", "")
                path = os.path.join(self.model_dir, file)
                try:
                    self.models[regime] = joblib.load(path)
                    logger.info(f"Loaded specialized model for regime: {regime}")
                except Exception as e:
                    logger.error(f"Failed to load model {file}: {e}")

    def get_model(self, regime):
        """
        Returns the model for the requested regime, or None if not available.
        """
        return self.models.get(regime)
