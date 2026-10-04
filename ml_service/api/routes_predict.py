"""Audio prediction endpoint route handler for ML inference microservice."""

import base64
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from exceptions import AudioDecodeError, MalformedRequestError, ModelInferenceError
from audio.preprocessing import preprocess_audio
from audio.validation import _decode_audio, validate_audio_file
from audio.visualization import analyze_audio_visualization
from features.extract import extract_features
from model.loader import get_model
from api.schemas import PredictResponse, TopPrediction

router = APIRouter()


def load_categories(config_path: Optional[Union[str, Path]] = None) -> List[str]:
    """Load sound category list from config/classes.json."""
    if config_path:
        path = Path(config_path)
    else:
        env_dir = os.getenv("CONFIG_DIR")
        if env_dir:
            path = Path(env_dir) / "classes.json"
        else:
            path = Path(__file__).resolve().parent.parent.parent / "config" / "classes.json"

    if not path.is_file():
        return [
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

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("categories", [])


def run_predict_pipeline_bytes(contents: bytes, filename: str = "uploaded_audio.wav") -> PredictResponse:
    """Execute validation, preprocessing, feature extraction, and ML inference on raw audio bytes.

    Args:
        contents: Raw audio bytes or base64-decoded byte payload.
        filename: Audio filename or label.

    Returns:
        PredictResponse payload.

    Raises:
        HTTPException: 400 for validation or processing errors; 500 for model failures.
    """
    if not contents or len(contents) == 0:
        raise MalformedRequestError(
            detail="Uploaded audio file is empty (0 bytes).",
        )

    if contents.startswith(b"data:audio/") or (len(contents) > 20 and b";base64," in contents[:100]):
        try:
            base64_data = contents.split(b";base64,")[-1]
            contents = base64.b64decode(base64_data)
        except Exception:
            pass

    val_result = validate_audio_file(contents, filename=filename)
    if not val_result.is_valid:
        if val_result.error_code == "FILE_CORRUPT":
            raise AudioDecodeError(detail=val_result.error_message or "Audio decoding failed due to corrupted file.")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=val_result.error_message or "Audio validation failed.",
        )

    try:
        audio_data, sample_rate = _decode_audio(contents)
    except Exception as exc:
        raise AudioDecodeError(
            detail=f"Audio decoding failed: {str(exc)}",
        ) from exc

    try:
        preprocessed = preprocess_audio(audio_data, sample_rate=sample_rate)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Audio preprocessing failed: {str(exc)}",
        )

    if len(preprocessed.data) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Preprocessed audio signal is empty.",
        )

    try:
        features_res = extract_features(preprocessed.data, sr=preprocessed.sample_rate)
        feat_vec = features_res["features_vector"]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Feature extraction failed: {str(exc)}",
        )

    try:
        model_artifact = get_model()
        model_obj = model_artifact.model
        model_version = model_artifact.version
    except Exception as exc:
        raise ModelInferenceError(
            detail=f"Failed to load ML model artifact: {str(exc)}",
        ) from exc

    all_categories = load_categories()

    target_classes = model_artifact.metadata.get("classes") if (model_artifact.metadata and "classes" in model_artifact.metadata) else all_categories

    feat_vec_2d = feat_vec.reshape(1, -1)

    try:
        if hasattr(model_obj, "predict_proba"):
            raw_probs = model_obj.predict_proba(feat_vec_2d)[0]
            prob_map = {}
            for i, prob in enumerate(raw_probs):
                if hasattr(model_obj, "classes_") and i < len(model_obj.classes_):
                    cls_val = model_obj.classes_[i]
                    if isinstance(cls_val, (int, float, type(0))):
                        idx = int(cls_val)
                        cls_name = target_classes[idx] if idx < len(target_classes) else str(idx)
                    else:
                        cls_name = str(cls_val)
                else:
                    cls_name = target_classes[i] if i < len(target_classes) else f"Class_{i}"
                prob_map[cls_name] = float(prob)
        elif hasattr(model_obj, "predict"):
            pred = model_obj.predict(feat_vec_2d)[0]
            if isinstance(pred, (int, float)):
                pred_cls = target_classes[int(pred)] if int(pred) < len(target_classes) else str(pred)
            else:
                pred_cls = str(pred)
            prob_map = {cls_name: (1.0 if cls_name == pred_cls else 0.0) for cls_name in all_categories}
        else:
            prob_map = {cls_name: 0.1 for cls_name in all_categories}
    except Exception as exc:
        raise ModelInferenceError(
            detail=f"Model inference failed: {str(exc)}",
        ) from exc

    confidence: Dict[str, float] = {}
    for cat in all_categories:
        confidence[cat] = round(float(prob_map.get(cat, 0.0)), 4)

    predicted_class = max(confidence.items(), key=lambda x: x[1])[0]

    sorted_preds = sorted(confidence.items(), key=lambda x: x[1], reverse=True)
    top3 = [
        TopPrediction(category=cat, confidence=score)
        for cat, score in sorted_preds[:3]
    ]

    viz_res = analyze_audio_visualization(preprocessed.data, sr=preprocessed.sample_rate)

    return PredictResponse(
        predicted_class=predicted_class,
        confidence=confidence,
        top3=top3,
        quality=viz_res["quality"],
        waveform=viz_res["waveform"],
        spectrogram=viz_res["spectrogram"],
        model_version=model_version,
    )


async def run_predict_pipeline(file: UploadFile) -> PredictResponse:
    """Execute prediction pipeline on an UploadFile instance."""
    filename = file.filename or "uploaded_audio.wav"

    try:
        contents = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file payload: {str(exc)}",
        )

    return run_predict_pipeline_bytes(contents, filename=filename)


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify audio file into sound-event categories",
    description="Validates, preprocesses, extracts features, and runs ML model on uploaded audio file.",
)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    """Predict sound-event category from multipart audio file upload."""
    return await run_predict_pipeline(file)


