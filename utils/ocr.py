# Taken from another research group (M. Hamdan)

from rapidocr_onnxruntime import RapidOCR
import numpy as np
from PIL import Image, ImageOps, ImageEnhance


recognizer = RapidOCR()

MISHAP_MAP = {
    "|": "1",
    "I": "1",
    "l": "1",
    "[": "1",
    "]": "1",
    "(": "1",
    ")": "1",
    "O": "0",
    "o": "0",
    "S": "5",
    "s": "5",
    "B": "8",
    "G": "6",
}


def preprocess_image(image_array: np.ndarray) -> np.ndarray:
    # Convert numpy array to PIL for processing
    if len(image_array.shape) == 3:
        img_pil = Image.fromarray(image_array)
    else:
        img_pil = Image.fromarray(image_array)

    # Grayscale
    img_gray = ImageOps.grayscale(img_pil)

    # Enhance contrast
    img_contrast = ImageEnhance.Contrast(img_gray).enhance(2.0)

    # Auto-level
    img_final = ImageOps.autocontrast(img_contrast)

    return np.array(img_final)


def extract_text(image_array: np.ndarray) -> list:
    result, _ = recognizer(image_array)
    return result if result else []


def clean_text(raw_text: str) -> str:
    text = raw_text.strip()

    # Apply character mapping
    for char, replacement in MISHAP_MAP.items():
        text = text.replace(char, replacement)

    # Keep only digits
    cleaned = ""
    for ch in text:
        if ch.isdigit():
            cleaned += ch

    return cleaned


def get_bbox_height(bbox: list) -> float:
    y_coords = [pt[1] for pt in bbox]
    return max(y_coords) - min(y_coords)


def get_bbox_width(bbox: list) -> float:
    x_coords = [pt[0] for pt in bbox]
    return max(x_coords) - min(x_coords)


def get_bbox_area(bbox: list) -> float:
    return get_bbox_width(bbox) * get_bbox_height(bbox)


def extract_all_text(image_array: np.ndarray) -> tuple[str, str]:
    ocr_results = extract_text(image_array)

    if not ocr_results:
        return "", ""

    ocr_results.sort(key=lambda r: min([pt[0] for pt in r[0]]))

    raw_text = "".join(r[1] for r in ocr_results).strip()
    cleaned = clean_text(raw_text)

    return raw_text, cleaned


def extract_by_height_threshold(
    image_array: np.ndarray, threshold_ratio: float = 0.60
) -> tuple[str, str]:
    ocr_results = extract_text(image_array)

    if not ocr_results:
        return "", ""

    # Find height threshold
    heights = [get_bbox_height(r[0]) for r in ocr_results]
    max_height = max(heights)
    threshold = max_height * threshold_ratio

    # Filter and sort left-to-right
    filtered = [r for r, h in zip(ocr_results, heights) if h >= threshold]
    filtered.sort(key=lambda r: min([pt[0] for pt in r[0]]))

    # Merge text
    raw_text = "".join(r[1] for r in filtered).strip()
    cleaned = clean_text(raw_text)

    return raw_text, cleaned
