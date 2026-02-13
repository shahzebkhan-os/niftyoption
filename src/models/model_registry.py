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
        Loads the latest versioned regime-specific models from the models directory.
        Expects format: {regime}_{version}.pkl or {regime}_model.pkl
        """
        if not os.path.exists(self.model_dir):
            logger.warning(f"Model directory '{self.model_dir}' not found.")
            return

        # 1. Group files by regime
        regime_groups = {}
        for file in os.listdir(self.model_dir):
            if file.endswith(".pkl"):
                # Handle both legacy ({regime}_model.pkl) and versioned ({regime}_{version}.pkl)
                if "_model.pkl" in file:
                    regime = file.replace("_model.pkl", "")
                else:
                    # Split by last underscore to separate regime from version
                    parts = file.replace(".pkl", "").rsplit("_", 1)
                    if len(parts) == 2:
                        regime = parts[0]
                    else:
                        continue
                
                if regime not in regime_groups:
                    regime_groups[regime] = []
                regime_groups[regime].append(file)

        # 2. Pick the latest for each and load
        for regime, files in regime_groups.items():
            latest_file = sorted(files)[-1] # String sort works for YYYY_MM_DD_HHMM
            path = os.path.join(self.model_dir, latest_file)
            try:
                self.models[regime] = joblib.load(path)
                logger.info(f"Loaded latest model for {regime}: {latest_file}")
            except Exception as e:
                logger.error(f"Failed to load model {latest_file}: {e}")

    def get_model(self, regime):
        """
        Returns the model for the requested regime, or None if not available.
        """
        return self.models.get(regime)
