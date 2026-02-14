import io
import os
import re
from dataclasses import dataclass
from typing import Union, Tuple

import gradio as gr
import numpy as np
import pytesseract
from PIL import Image
import cv2

MAX_BYTES_SIZE = 5 * 1024 * 1024  # 5MB


@dataclass
class ImageExtractorResult:
    text: str
    width: int
    height: int
    size_bytes: int


class ImageTooLargeError(Exception):
    pass


class UnsupportedImageFormatError(Exception):
    pass


def _normalize_to_paragraph(text: str) -> str:
    """
    Convert OCR output to a single paragraph:
    - Replace newlines with spaces
    - Collapse multiple spaces
    """
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert PIL RGB image to OpenCV BGR image."""
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR (or grayscale) image to PIL image."""
    if bgr.ndim == 2:
        return Image.fromarray(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _auto_crop_text_region(bgr: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Try to find a large text/document region and crop it.
    This helps when the receipt is not full screen (phone photo).
    Returns (cropped_bgr, success).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Close gaps to form bigger contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return bgr, False

    # Take the largest contour as receipt/document candidate
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    h, w = bgr.shape[:2]
    if area < 0.10 * (h * w):
        # If the contour is too small, skip cropping
        return bgr, False

    x, y, cw, ch = cv2.boundingRect(c)

    # Add padding around crop
    pad = int(0.02 * max(h, w))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + cw + pad)
    y2 = min(h, y + ch + pad)

    return bgr[y1:y2, x1:x2], True


def _enhance_for_ocr(bgr: np.ndarray, upscale: float = 2.0) -> np.ndarray:
    """
    Enhance image for OCR:
    - Upscale to help small receipt text
    - Grayscale + CLAHE for contrast
    - Adaptive threshold for uneven lighting (phone photos)
    """
    h, w = bgr.shape[:2]
    if upscale and upscale > 1.0:
        bgr = cv2.resize(bgr, (int(w * upscale), int(h * upscale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Reduce noise while preserving edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive threshold handles shadows/light variations
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35, 11
    )
    return th


def extract_image_to_text(
    image: Union[str, bytes],
    *,
    lang: str = "vie+eng",          # better for Vietnamese receipts (often includes digits/English)
    max_bytes: int = MAX_BYTES_SIZE,
    psm: int = 6,                   # 6 works well for receipt blocks
    upscale: float = 2.0,
    auto_crop: bool = True,
) -> Tuple[ImageExtractorResult, Image.Image]:
    """
    Extract text from an image and return (result, processed_preview).
    Output text is returned as a single paragraph.
    """
    # Read bytes and size check
    if isinstance(image, str):
        size = os.path.getsize(image)
        if size > max_bytes:
            raise ImageTooLargeError(f"Image size {size} > limit {max_bytes} bytes")
        with open(image, "rb") as f:
            data = f.read()
    else:
        data = image
        size = len(data)
        if size > max_bytes:
            raise ImageTooLargeError(f"Image size {size} > limit {max_bytes} bytes")

    try:
        pil = Image.open(io.BytesIO(data))
        pil.load()

        bgr = _pil_to_bgr(pil)

        # Auto crop to the document/text region (helps phone images not full screen)
        if auto_crop:
            bgr, _ = _auto_crop_text_region(bgr)

        # Enhance for OCR
        proc = _enhance_for_ocr(bgr, upscale=upscale)
        proc_pil = _bgr_to_pil(proc)

        width, height = proc_pil.size

        # Tesseract configuration:
        # - OEM 3: default LSTM engine
        # - PSM 6: assume a uniform block of text (receipts)
        # - preserve_interword_spaces: keep spacing better (even though we later paragraphize)
        config = f'--oem 3 --psm {psm} -c preserve_interword_spaces=1'

        text = pytesseract.image_to_string(proc_pil, lang=lang, config=config)
        text = _normalize_to_paragraph(text)

        return (
            ImageExtractorResult(
                text=text,
                width=width,
                height=height,
                size_bytes=size,
            ),
            proc_pil
        )

    except Exception as e:
        raise UnsupportedImageFormatError(str(e)) from e


def _ui_extract(file_obj, lang: str, psm: int, upscale: float, auto_crop: bool):
    if file_obj is None:
        return None, None, "", "Please upload an image."

    path = getattr(file_obj, "name", None)
    if not path or not os.path.isfile(path):
        return None, None, "", "Invalid uploaded file."

    size = os.path.getsize(path)
    if size > MAX_BYTES_SIZE:
        return None, None, "", f"Image too large: {size/1024/1024:.2f}MB (max 5MB)"

    try:
        original_preview = Image.open(path).convert("RGB")

        res, processed_preview = extract_image_to_text(
            path,
            lang=lang,
            psm=int(psm),
            upscale=float(upscale),
            auto_crop=bool(auto_crop),
        )

        meta = (
            f"file={size/1024/1024:.2f}MB | processed={res.width}x{res.height} | "
            f"lang={lang} | psm={psm} | upscale={upscale:.1f} | auto_crop={auto_crop}"
        )
        text = res.text if res.text else "(no text detected)"
        return original_preview, processed_preview, meta, text

    except Exception as e:
        return None, None, "", f"Error: {e}"


def launch_ui(host: str = "127.0.0.1", port: int = 7861):
    with gr.Blocks() as demo:
        gr.Markdown("## 🧾 Receipt OCR (Paragraph Output) — supports phone photos")

        with gr.Row():
            file_in = gr.File(label="Image", file_types=["image"])

        with gr.Row():
            lang = gr.Dropdown(["vie+eng", "vie", "eng"], value="vie+eng", label="Language")
            psm = gr.Slider(3, 13, value=6, step=1, label="PSM (6 recommended)")

        with gr.Row():
            auto_crop = gr.Checkbox(value=True, label="Auto crop document/text region")
            upscale = gr.Slider(1.0, 3.0, value=2.0, step=0.1, label="Upscale (helps small text)")

        btn = gr.Button("Extract")

        with gr.Row():
            img_orig = gr.Image(label="Original", height=320)
            img_proc = gr.Image(label="Processed (used for OCR)", height=320)

        meta_out = gr.Textbox(label="Meta", lines=1)
        text_out = gr.Textbox(label="Text (paragraph)", lines=10)

        btn.click(
            _ui_extract,
            inputs=[file_in, lang, psm, upscale, auto_crop],
            outputs=[img_orig, img_proc, meta_out, text_out],
        )

    demo.launch(server_name=host, server_port=port)


if __name__ == "__main__":
    launch_ui()
