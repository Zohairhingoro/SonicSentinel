from typing import Dict
from fastapi import APIRouter
from api.routes_predict import router as predict_router
from api.routes_predict_batch import router as predict_batch_router
from api.routes_live import router as live_router

router = APIRouter()
router.include_router(predict_router)
router.include_router(predict_batch_router)
router.include_router(live_router)




@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint returning system status."""
    return {"status": "ok"}

