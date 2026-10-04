"""Pydantic schemas for ML inference API request and response data."""

from typing import Dict, List
from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    """Container for top predicted class category and confidence score."""

    category: str = Field(..., description="Category label")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")


class PredictResponse(BaseModel):
    """Response schema for POST /predict inference endpoint per SRS §1 contract."""

    predicted_class: str = Field(..., description="Top predicted sound category label")
    confidence: Dict[str, float] = Field(
        ..., description="Confidence scores dictionary mapping all sound categories to float values"
    )
    top3: List[TopPrediction] = Field(..., description="Top 3 predicted categories with confidence scores")
    quality: str = Field(..., description="Audio quality classification: Good, Acceptable, Poor, or Unusable")
    waveform: List[float] = Field(..., description="Downsampled waveform amplitude array for frontend rendering")
    spectrogram: List[List[float]] = Field(
        ..., description="2D decibel-scaled Mel-spectrogram normalized matrix"
    )
    model_version: str = Field(..., description="Model version identifier")
