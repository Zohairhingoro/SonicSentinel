import json
import tempfile
from pathlib import Path
import joblib
import pytest
try:
    from ml_service.model.loader import (
        FallbackModelStub,
        ModelArtifact,
        get_model,
        initialize_model,
        load_model_artifact,
    )
except ModuleNotFoundError:
    from model.loader import (
        FallbackModelStub,
        ModelArtifact,
        get_model,
        initialize_model,
        load_model_artifact,
    )



class DummyClassifier:
    def __init__(self):
        self.classes_ = ["ClassA", "ClassB"]

    def predict(self, X):
        return ["ClassA"]

    def predict_proba(self, X):
        return [[0.8, 0.2]]


def test_load_default_metadata():
    artifact = load_model_artifact()
    assert isinstance(artifact, ModelArtifact)
    assert isinstance(artifact.version, str) and len(artifact.version) > 0
    assert isinstance(artifact.trained_on, str) and len(artifact.trained_on) > 0
    assert "accuracy" in artifact.metrics
    assert artifact.model is not None
    assert artifact.artifact_path is not None and artifact.artifact_path.name == "xgb_model.joblib"


def test_singleton_get_model():
    model_instance1 = get_model(force_reload=True)
    model_instance2 = get_model()
    assert model_instance1 is model_instance2


def test_force_reload_get_model():
    model_instance1 = get_model()
    model_instance2 = get_model(force_reload=True)
    assert isinstance(model_instance1, ModelArtifact)
    assert isinstance(model_instance2, ModelArtifact)


def test_load_from_custom_artifacts_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        meta = {
            "version": "2.1.0",
            "trained_on": "2026-09-25",
            "metrics": {"accuracy": 0.92, "macro_f1": 0.89},
            "classes": ["ClassA", "ClassB"],
        }
        with open(tmp_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

        dummy_model = DummyClassifier()
        joblib.dump(dummy_model, tmp_path / "model.joblib")

        artifact = load_model_artifact(artifacts_dir=tmp_path)
        assert artifact.version == "2.1.0"
        assert artifact.trained_on == "2026-09-25"
        assert artifact.metrics["accuracy"] == 0.92
        assert artifact.artifact_path == tmp_path / "model.joblib"
        assert artifact.model.predict(None) == ["ClassA"]


def test_fallback_model_stub():
    stub = FallbackModelStub(classes=["C1", "C2"])
    assert stub.predict(None) == ["C1"]
    assert stub.predict_proba(None) == [[0.5, 0.5]]


def test_model_artifact_to_dict():
    artifact = load_model_artifact()
    d = artifact.to_dict()
    assert isinstance(d, dict)
    assert "version" in d
    assert "trained_on" in d
    assert "metrics" in d
