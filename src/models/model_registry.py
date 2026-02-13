import joblib
import os
import logging
import json

logger = logging.getLogger(__name__)

class ModelRegistry:
    """
    Production-grade model registry.
    Handles loading of versioned models, metadata parsing, and registry persistence.
    """
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.models = {}
        self.metadata = {}
        self.registry_path = os.path.join(model_dir, "registry.json")
        self.load_models()

    def load_models(self):
        """
        Loads models from structured versioned directories.
        Syncs with registry.json if it exists, otherwise picks latest by timestamp.
        """
        if not os.path.exists(self.model_dir):
            logger.warning(f"Model directory '{self.model_dir}' not found.")
            return

        # 1. Load Registry if available
        active_versions = {}
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, "r") as f:
                    active_versions = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load registry.json: {e}")

        # 2. Scan for all available models
        regime_versions = {}
        for entry in os.scandir(self.model_dir):
            if entry.is_dir():
                # Expected format: {regime}_{timestamp}
                parts = entry.name.rsplit("_", 1)
                if len(parts) == 2:
                    regime = parts[0]
                    if regime not in regime_versions:
                        regime_versions[regime] = []
                    regime_versions[regime].append(entry.name)

        # 3. Resolve and Load
        for regime, versions in regime_versions.items():
            # If registry has a specific version, use it. Otherwise, use latest.
            version_to_load = active_versions.get(regime)
            if not version_to_load or version_to_load not in versions:
                version_to_load = sorted(versions)[-1]
            
            path = os.path.join(self.model_dir, version_to_load)
            try:
                model_file = os.path.join(path, "model.pkl")
                meta_file = os.path.join(path, "metadata.json")
                
                if os.path.exists(model_file):
                    self.models[regime] = joblib.load(model_file)
                    if os.path.exists(meta_file):
                        with open(meta_file, "r") as f:
                            self.metadata[regime] = json.load(f)
                    logger.info(f"Loaded {regime} model version: {version_to_load}")
            except Exception as e:
                logger.error(f"Failed to load model from {path}: {e}")
        
        # 4. Handle legacy models in root for backward compatibility
        for file in os.listdir(self.model_dir):
            if file.endswith("_model.pkl"):
                regime = file.replace("_model.pkl", "")
                if regime not in self.models:
                    try:
                        self.models[regime] = joblib.load(os.path.join(self.model_dir, file))
                        logger.info(f"Loaded legacy model for {regime}")
                    except:
                        pass

    def get_model(self, regime):
        """Returns the model for the requested regime."""
        return self.models.get(regime)

    def get_metadata(self, regime):
        """Returns metadata for the requested regime model."""
        return self.metadata.get(regime)
