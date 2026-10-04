from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class AuditLog:
    log_id: str
    action: str
    user_id: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None
