from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioEvent:
    audio_id: str
    filename: str
    format: str
    duration: float
    upload_timestamp: str
    user_id: str
    storage_path: Optional[str] = None
