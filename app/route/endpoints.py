from fastapi import APIRouter, UploadFile, File, Form
from app.core.enums import InputType
from app.services.extractor.context import Context
from app.schemas.response import Response

router = APIRouter()
context = Context()


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