import cv2
import easyocr
import numpy as np

FAIL_STRING: str = "UNREADABLE"
UPSCALE_FACTOR: int = 3


# Taken from Hamdan.
def extract_tag_id(tag_image: np.ndarray, reader: easyocr.Reader | None = None) -> str:
    """Take an OpenCV image of an ear tag and extract the tag id from it."""

    if tag_image is None or tag_image.size == 0 or reader is None:
        return FAIL_STRING

    filtered = cv2.bilateralFilter(tag_image, 9, 75, 75)
    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    h, w = enhanced.shape
    processed_image = cv2.resize(
        enhanced,
        (w * UPSCALE_FACTOR, h * UPSCALE_FACTOR),
        interpolation=cv2.INTER_CUBIC,
    )

    results = reader.readtext(
        processed_image,
        detail=1,
        paragraph=False,
        allowlist="0123456789",
        width_ths=0.7,
        contrast_ths=0.1,
        min_size=8,
    )

    valid_text = [
        text for (_, text, conf) in results if float(conf) >= 0.20 and text.strip()
    ]

    if valid_text:
        return " ".join(valid_text).strip()

    return FAIL_STRING
