import numpy as np
from PIL import ImageOps
from ..services.extractor.image.pre_processing import blur_image, resize

class OCRPipeline:

    def __init__(self, model):
        self.model = model

    def run(self, img):
        grayscale = ImageOps.grayscale(img)
        arr = np.array(grayscale)

        # Blank / uniform image (white, black, solid colour) — no content to OCR
        if arr.std() < 8.0:
            return {"error": "Image has no content (blank or uniform colour)", "text": ""}

        # Blurry image — Laplacian variance below threshold
        _lap, is_blur, _var = blur_image.is_blur(arr)
        if is_blur:
            return {"error": "Image is too blurry to read", "text": ""}

        processed = resize.pre_process_image(grayscale)
        result = self.model.predict(processed)

        if not result or not str(result).strip():
            return {"error": "No text found in image", "text": ""}

        return {"error": None, "text": result}