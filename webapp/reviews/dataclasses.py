from dataclasses import dataclass
from typing import Optional


@dataclass
class ReviewRecord:
    review_id: str
    audio_id: str
    reviewer_id: str
    decision: str
    comments: Optional[str] = None
