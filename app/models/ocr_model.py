import os
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


class OCRModel:

    def __init__(self):
        print("Loading OCR models...")
        os.environ["FLAGS_use_pir_api"] = "0"
        os.environ["FLAGS_use_mkldnn"] = "1"

        # Detector
        self.detector = PaddleOCR(
            use_angle_cls=True,
            lang="vi",
            rec=False
        )

        # Recognizer
        config = Cfg.load_config_from_name("vgg_transformer")
        config["device"] = "cpu"
        config["predictor"]["beamsearch"] = False
        config["predictor"]["batch_size"] = 16

        self.recognizer = Predictor(config)

        # Warmup
        dummy = Image.fromarray(np.zeros((32, 128, 3), dtype=np.uint8))
        self.recognizer.predict(dummy)

        print("OCR models loaded")

    def predict(self, img: Image.Image) -> str:

        img_np = np.array(img)

        if len(img_np.shape) == 2:
            img_np = np.stack([img_np] * 3, axis=-1)

        result = self.detector.ocr(img_np, cls=True)

        if not result or not result[0]:
            return ""

        boxes = result[0]

        crops = []
        positions = []

        for box in boxes:
            points = box[0]

            x_min = int(min(p[0] for p in points))
            x_max = int(max(p[0] for p in points))
            y_min = int(min(p[1] for p in points))
            y_max = int(max(p[1] for p in points))

            crop = img_np[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                continue

            crops.append(Image.fromarray(crop))
            positions.append((x_min, y_min))

        if not crops:
            return ""

        texts = self.recognizer.predict_batch(crops)

        merged = list(zip(positions, texts))
        merged = sorted(merged, key=lambda x: x[0][1])

        lines = []
        current_line = []
        current_y = None
        y_threshold = 15

        for (x_min, y_min), text in merged:
            text = text.strip()
            if not text:
                continue

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