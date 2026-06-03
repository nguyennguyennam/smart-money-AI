import os
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR


class OCRModel:

    def __init__(self):
        print("Loading OCR models...")
        os.environ["FLAGS_use_pir_api"] = "0"
        os.environ["FLAGS_use_mkldnn"] = "1"

        # End-to-end OCR (detector + recognizer) via PaddleOCR.
        # This avoids a hard runtime dependency on VietOCR/Torch, which can be
        # fragile on Windows when Torch DLL dependencies are missing.
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang="vi",
        )

        print("OCR models loaded")

    def predict(self, img: Image.Image) -> str:

        img_np = np.array(img)

        if len(img_np.shape) == 2:
            img_np = np.stack([img_np] * 3, axis=-1)

        result = self.ocr.ocr(img_np, cls=True)

        if not result or not result[0]:
            return ""

        items = result[0]
        merged = []

        for item in items:
            points = item[0]
            text = (item[1][0] if item[1] else "").strip()
            if not text:
                continue

            x_min = int(min(p[0] for p in points))
            y_min = int(min(p[1] for p in points))
            merged.append(((x_min, y_min), text))

        if not merged:
            return ""

        merged = sorted(merged, key=lambda x: x[0][1])

        lines = []
        current_line = []
        current_y = None
        y_threshold = 15

        for (x_min, y_min), text in merged:

            if current_y is None:
                current_y = y_min
                current_line.append((x_min, text))

            elif abs(y_min - current_y) < y_threshold:
                current_line.append((x_min, text))

            else:
                current_line = sorted(current_line, key=lambda x: x[0])
                lines.append(" ".join(t[1] for t in current_line))

                current_line = [(x_min, text)]
                current_y = y_min

        if current_line:
            current_line = sorted(current_line, key=lambda x: x[0])
            lines.append(" ".join(t[1] for t in current_line))

        return "\n".join(lines)