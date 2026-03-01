import cv2
import numpy as np
from PIL import Image

def pre_process_image(pil_img):

    gray = np.array(pil_img)

    gray = cv2.resize(gray, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    return Image.fromarray(enhanced)