import os
import shutil
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ModelVersionManager:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def generate_version(self):
        """Generates a UTC timestamped version string."""
        return datetime.utcnow().strftime("%Y_%m_%d_%H%M")

    def promote_model(self, regime, temp_model_path):
        """
        Moves a temporary trained model to the production directory 
        with a timestamped version.
        """
        if not os.path.exists(temp_model_path):
            logger.error(f"Temporary model path {temp_model_path} does not exist.")
            return None

        version = self.generate_version()
        new_filename = f"{regime}_{version}.pkl"
        new_path = os.path.join(self.model_dir, new_filename)
        
        try:
            # Copy to preserve the original temp file if needed, or move
            shutil.move(temp_model_path, new_path)
            logger.info(f"Promoted {regime} model to versioned path: {new_path}")
            return new_path
        except Exception as e:
            logger.error(f"Failed to promote model: {e}")
            return None

    def get_latest_model_path(self, regime):
        """
        Returns the path to the most recent version of a model for a specific regime.
        """
        matching_models = [f for f in os.listdir(self.model_dir) if f.startswith(f"{regime}_") and f.endswith(".pkl")]
        if not matching_models:
            return None
        
        # Sort by filename (which includes timestamp) and pick the last one
        latest_filename = sorted(matching_models)[-1]
        return os.path.join(self.model_dir, latest_filename)

    def cleanup_old_versions(self, regime, keep=5):
        """
        Keeps only the 'keep' most recent models for a regime to save space.
        """
        matching_models = [f for f in os.listdir(self.model_dir) if f.startswith(f"{regime}_") and f.endswith(".pkl")]
        if len(matching_models) <= keep:
            return

        # Sort and pick old ones
        old_models = sorted(matching_models)[:-keep]
        for model in old_models:
            os.remove(os.path.join(self.model_dir, model))
            logger.info(f"Cleaned up old model version: {model}")
