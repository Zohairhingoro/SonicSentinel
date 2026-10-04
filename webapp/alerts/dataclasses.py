from dataclasses import dataclass


@dataclass
class Alert:
    alert_id: str
    audio_id: str
    category: str
    severity: str
    status: str
