import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import joblib
except ImportError:
    joblib = None


def get_default_artifacts_dir() -> Path:
    """Return absolute path to model artifacts directory."""
    env_dir = os.getenv("MODEL_ARTIFACTS_DIR")
    if env_dir:
        return Path(env_dir).resolve()
    return Path(__file__).resolve().parent / "artifacts"


@dataclass
class ModelArtifact:
    """Container for loaded machine learning model and its metadata."""

    model: Any
    version: str
    trained_on: str
    metrics: Dict[str, Any]
    metadata: Dict[str, Any]
    artifact_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert model metadata and version info to dictionary representation."""
        return {
            "version": self.version,
            "trained_on": self.trained_on,
            "metrics": self.metrics,
            "artifact_path": str(self.artifact_path) if self.artifact_path else None,
            "metadata": self.metadata,
        }


class FallbackModelStub:
    """Lightweight stub model used when no binary model artifact is present on disk."""

    def __init__(self, classes: Optional[list] = None) -> None:
        self.classes_ = classes or [
            "Machinery Fault",
            "Glass Breaking",
            "Alarm or Siren",
            "Vehicle Horn",
            "Animal Sound",
            "Gunshot",
            "Panic Scream",
            "Aggression or Violent Conflict",
            "Person Asking for Help",
            "Background Noise",
        ]

    def predict(self, features: Any) -> list:
        """Return fallback default prediction class."""
        return [self.classes_[0]]

    def predict_proba(self, features: Any) -> list:
        """Return fallback uniform class confidence distribution."""
        n_classes = len(self.classes_)
        uniform_prob = 1.0 / n_classes
        return [[uniform_prob] * n_classes]


def load_model_artifact(artifacts_dir: Optional[Union[str, Path]] = None) -> ModelArtifact:
    """Load model artifact and associated metadata.json from specified directory.

    Args:
        artifacts_dir: Path to directory containing metadata.json and model file.

    Returns:
        ModelArtifact object containing model, version, trained_on, and metrics.
    """
    path = Path(artifacts_dir) if artifacts_dir else get_default_artifacts_dir()

    metadata_path = path / "metadata.json"
    metadata: Dict[str, Any] = {}
    if metadata_path.is_file():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    version = metadata.get("version", "1.0.0")
    trained_on = metadata.get("trained_on", "unknown")
    metrics = metadata.get("metrics", {})

    model_obj: Any = None
    loaded_artifact_path: Optional[Path] = None

    # Explicitly prioritize the winning XGBoost model artifact (xgb_model.joblib / *xgb*)
    xgb_candidates = [
        f for f in path.iterdir()
        if f.is_file() and ("xgb" in f.name.lower()) and f.suffix in (".joblib", ".pkl", ".pickle")
    ]
    if xgb_candidates:
        model_file = xgb_candidates[0]
    else:
        candidate_files = list(path.glob("*.joblib")) + list(path.glob("*.pkl")) + list(path.glob("*.pickle"))
        model_file = candidate_files[0] if candidate_files else None

    if model_file is not None:
        loaded_artifact_path = model_file
        if joblib is not None and model_file.suffix == ".joblib":
            model_obj = joblib.load(model_file)
        else:
            with open(model_file, "rb") as f:
                model_obj = pickle.load(f)

    if model_obj is None:
        model_obj = FallbackModelStub(classes=metadata.get("classes"))

    # Training scripts serialise a dict wrapper: {model, label_encoder, classes, ...}
    # Unwrap it so model_obj is the actual estimator.
    artifact_metadata: Dict[str, Any] = {}
    if isinstance(model_obj, dict):
        artifact_metadata = {k: v for k, v in model_obj.items() if k != "model"}
        model_obj = model_obj.get("model", FallbackModelStub(classes=metadata.get("classes")))

    return ModelArtifact(
        model=model_obj,
        version=version,
        trained_on=trained_on,
        metrics=metrics,
        metadata={**metadata, **artifact_metadata},
        artifact_path=loaded_artifact_path,
    )


_MODEL_SINGLETON: Optional[ModelArtifact] = None


def get_model(
    artifacts_dir: Optional[Union[str, Path]] = None,
    force_reload: bool = False,
) -> ModelArtifact:
    """Return the cached module-level singleton model artifact instance.

    Loads the artifact on first invocation and returns the cached instance on subsequent calls.

    Args:
        artifacts_dir: Optional directory override for loading artifacts.
        force_reload: If True, forces reloading artifact from disk.

    Returns:
        ModelArtifact singleton instance.
    """
    global _MODEL_SINGLETON
    if _MODEL_SINGLETON is None or force_reload:
        _MODEL_SINGLETON = load_model_artifact(artifacts_dir=artifacts_dir)
    return _MODEL_SINGLETON


def initialize_model(artifacts_dir: Optional[Union[str, Path]] = None) -> ModelArtifact:
    """Eagerly load and initialize the module-level model singleton at service startup."""
    return get_model(artifacts_dir=artifacts_dir, force_reload=True)
