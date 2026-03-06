import numpy as np
from PIL import ImageOps
from services.extractor.image.pre_processing import blur_image, resize


class OCRPipeline:

    def __init__(self, model):
        self.model = model

    def run(self, img):

        grayscale = ImageOps.grayscale(img)

        lap, is_blur, var = blur_image.is_blur(np.array(grayscale))
        if is_blur:
            return {
                "error": "Image is blur",
                "text": ""
            }

        processed = resize.pre_process_image(grayscale)

        result = self.model.predict(processed)

        return {
            "error": None,
            "text": result
        }