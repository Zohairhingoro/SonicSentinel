"""Batch audio prediction endpoint route handler for ML inference microservice."""

from typing import List
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from exceptions import MalformedRequestError
from api.routes_predict import run_predict_pipeline
from api.schemas import PredictResponse

router = APIRouter()


@router.post(
    "/predict/batch",
    response_model=List[PredictResponse],
    summary="Classify multiple audio files in batch into sound-event categories",
    description="Accepts multiple audio file uploads and returns a list of classification prediction results.",
)
async def predict_batch(files: List[UploadFile] = File(...)) -> List[PredictResponse]:
    """Predict sound-event categories for a batch of uploaded audio files.

    Args:
        files: List of multipart uploaded audio files.

    Returns:
        List of PredictResponse objects matching the exact per-file response schema of POST /predict.

    Raises:
        MalformedRequestError: If files list is empty.
    """
    if not files or len(files) == 0:
        raise MalformedRequestError(
            detail="No files provided in batch upload request.",
        )

    results: List[PredictResponse] = []
    for file in files:
        response = await run_predict_pipeline(file)
        results.append(response)

    return results
