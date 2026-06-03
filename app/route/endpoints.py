from functools import lru_cache

from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel

from core.enums import InputType
from services.classifer.classifier import get_classifier_service
from schemas.response import Response

router = APIRouter()
classifier_service = get_classifier_service()


@lru_cache(maxsize=1)
def get_extractor_context():
    # Local import keeps heavy OCR/ASR deps from loading
    # unless the /process endpoint is actually used.
    from app.services.extractor.context import Context

    return Context()


class ClassifyRequest(BaseModel):
    text: str


@router.post(
    "/process",
    response_model=Response,
    summary="Process image or voice file"
)
async def process_file(
    input_type: InputType = Form(...),
    file: UploadFile = File(...)
):
    try:
        context = get_extractor_context()
        result = await context.select_extractor(input_type, file)

        if isinstance(result, dict) and result.get("error"):
            return Response(
                success=False,
                error=result["error"]
            )

        return Response(
            success=True,
            data=result
        )

    except Exception as e:
        return Response(
            success=False,
            error=str(e)
        )


@router.post(
    "/classify",
    response_model=Response,
    summary="Classify text into a spending category",
)
async def classify_text(payload: ClassifyRequest):
    try:
        result = classifier_service.classify(payload.text)
        return Response(
            success=True,
            data={
                "category": result.category.value,
                "confidence": result.confidence,
            },
        )
    except Exception as e:
        return Response(success=False, error=str(e))