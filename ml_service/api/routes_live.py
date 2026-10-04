"""Live audio stream window prediction endpoint route handler for ML inference microservice."""

import base64
import json
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from exceptions import MalformedRequestError
from api.routes_predict import run_predict_pipeline_bytes
from api.schemas import PredictResponse

router = APIRouter()


class LivePredictRequest(BaseModel):
    """JSON request schema for live audio window classification."""

    audio_data: str = Field(
        ...,
        description="Base64-encoded audio window string or data URL (e.g. data:audio/wav;base64,...)",
    )
    filename: Optional[str] = Field("live_window.wav", description="Optional filename or window label")


@router.post(
    "/predict/live",
    response_model=PredictResponse,
    summary="Classify live audio window chunk in real-time",
    description=(
        "Optimized endpoint for real-time live microphone window processing. Accepts base64 JSON or "
        "raw binary bytes body to minimize transmission overhead and achieve sub-3-second response target."
    ),
)
async def predict_live(request: Request) -> PredictResponse:
    """Predict sound-event category for a live audio stream window.

    Supports base64 JSON payload, raw base64 string body, or raw audio binary bytes stream.

    Args:
        request: FastAPI raw HTTP Request.

    Returns:
        PredictResponse containing predictions, confidence, top3, quality, and visualization metrics.

    Raises:
        MalformedRequestError: If payload is empty or invalid format.
    """
    raw_body = await request.body()
    contents: Optional[bytes] = None
    filename = "live_window.wav"

    if not raw_body or len(raw_body) == 0:
        raise MalformedRequestError(
            detail="Live audio prediction payload is empty (0 bytes).",
        )

    # 1. Check if payload is JSON (e.g. {"audio_data": "..."})
    content_type = request.headers.get("content-type", "").lower()
    if raw_body.startswith(b"{") or "json" in content_type:
        try:
            data = json.loads(raw_body.decode("utf-8"))
            if isinstance(data, dict):
                audio_str = data.get("audio_data", "")
                if audio_str:
                    audio_str = audio_str.strip()
                    if audio_str.startswith("data:audio/") or ";base64," in audio_str:
                        audio_str = audio_str.split(";base64,")[-1]
                    contents = base64.b64decode(audio_str)
                    filename = data.get("filename", "live_window.wav")
        except Exception as exc:
            if contents is None:
                raise MalformedRequestError(
                    detail=f"Failed to parse JSON live audio payload: {str(exc)}",
                ) from exc

    # 2. Check if payload is raw base64 data URL string
    if contents is None:
        if raw_body.startswith(b"data:audio/") or (len(raw_body) > 20 and b";base64," in raw_body[:100]):
            try:
                base64_data = raw_body.split(b";base64,")[-1]
                contents = base64.b64decode(base64_data)
            except Exception as exc:
                raise MalformedRequestError(
                    detail=f"Failed to decode base64 live audio payload: {str(exc)}",
                ) from exc
        else:
            # 3. Direct raw binary audio stream (e.g. WAV bytes)
            contents = raw_body

    if not contents or len(contents) == 0:
        raise MalformedRequestError(
            detail="Live audio prediction payload is empty (0 bytes).",
        )

    return run_predict_pipeline_bytes(contents, filename=filename)
