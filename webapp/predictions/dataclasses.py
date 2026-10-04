from dataclasses import dataclass
from typing import Dict


@dataclass
class PythonPrediction:
    audio_id: str
    predicted_class: str
    confidence: Dict[str, float]
    model_version: str
