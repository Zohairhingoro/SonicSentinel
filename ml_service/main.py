"""Main FastAPI application module for SonicSentinel ML Inference Service."""

import json
import logging
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.routes import router
from exceptions import AudioDecodeError, MalformedRequestError, ModelInferenceError

logger = logging.getLogger("ml_service")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


app = FastAPI(
    title="SonicSentinel ML Inference Service",
    description="Stateless inference microservice for sound-event detection",
    version="1.0.0",
)


@app.exception_handler(AudioDecodeError)
async def audio_decode_exception_handler(request: Request, exc: AudioDecodeError) -> JSONResponse:
    """Handle audio decoding failures and return JSON error response without leaking stack trace."""
    logger.error("Audio decoding failure: %s", exc.detail, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error_code": "AUDIO_DECODE_FAILURE",
            "message": exc.message or "Audio decoding failure",
            "detail": exc.detail or exc.message or "Audio decoding failed.",
        },
    )


@app.exception_handler(ModelInferenceError)
async def model_inference_exception_handler(request: Request, exc: ModelInferenceError) -> JSONResponse:
    """Handle model inference failures and return JSON error response without leaking stack trace."""
    logger.error("Model inference failure: %s", exc.detail, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error_code": "MODEL_INFERENCE_FAILURE",
            "message": exc.message or "Model inference failure",
            "detail": exc.detail or exc.message or "Model inference failed.",
        },
    )


@app.exception_handler(MalformedRequestError)
async def malformed_request_exception_handler(request: Request, exc: MalformedRequestError) -> JSONResponse:
    """Handle malformed request payload errors and return JSON error response without leaking stack trace."""
    logger.warning("Malformed request: %s", exc.detail, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error_code": "MALFORMED_REQUEST",
            "message": exc.message or "Malformed request",
            "detail": exc.detail or exc.message or "Malformed request payload.",
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle FastAPI request validation errors and return JSON error response without leaking stack trace."""
    logger.warning("Request validation failure: %s", str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error_code": "MALFORMED_REQUEST",
            "message": "Malformed request",
            "detail": "Malformed request payload or invalid parameters.",
        },
    )


@app.exception_handler(json.JSONDecodeError)
async def json_decode_exception_handler(request: Request, exc: json.JSONDecodeError) -> JSONResponse:
    """Handle JSON parsing errors in request payloads and return JSON error response without leaking stack trace."""
    logger.warning("JSON decode failure: %s", str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error_code": "MALFORMED_REQUEST",
            "message": "Malformed request",
            "detail": f"Failed to parse JSON body payload: {str(exc)}",
        },
    )


app.include_router(router)
