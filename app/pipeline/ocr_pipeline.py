import numpy as np
from PIL import ImageOps
from ..services.extractor.image.pre_processing import blur_image


class OCRPipeline:
    """Cheap guards before sending an image to the OCR LLM call.

    No longer runs a local OCR model — the actual OCR is performed by
    `LLMService.generate_with_image` in the worker. This pipeline only filters
    out blank or blurry images so we do not spend an API call on garbage.
    """

    def __init__(self, model=None):
        # `model` kept for backwards-compatible constructor signature.
        self.model = model

    def run(self, img):
        grayscale = ImageOps.grayscale(img)
        arr = np.array(grayscale)

        # Blank / uniform image (white, black, solid colour) — no content to OCR
        if arr.std() < 8.0:
            return {"error": "Image has no content (blank or uniform colour)", "ok": False}

        # Blurry image — Laplacian variance below threshold
        _lap, is_blur, _var = blur_image.is_blur(arr)
        if is_blur:
            return {"error": "Image is too blurry to read", "ok": False}

        return {"error": None, "ok": True}
