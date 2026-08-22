from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Dict

from app.auth.dependencies import get_current_user
from app.database.mongodb import db_client
from app.ml.prediction import predict_with_model
from app.utils.response import ok

router = APIRouter(prefix="", tags=["Prediction Engine"])

class PredictionRequest(BaseModel):
    model_id: str
    values: Dict[str, float] = Field(..., description="Key-value mapping of feature name to float value")

@router.post("/predict")
async def predict(
    request: PredictionRequest, current_user: dict = Depends(get_current_user)
):
    model_doc = await db_client.find_one("models", {"_id": request.model_id})
    if not model_doc or model_doc.get("user_id") != current_user["_id"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID {request.model_id} not found."
        )

    try:
        pred_value = predict_with_model(request.model_id, request.values)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

    return ok({
        "prediction": pred_value,
        "value": pred_value  # kept for frontend api-service.ts compatibility
    })
