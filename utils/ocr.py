import cv2
import easyocr
import numpy as np

FAIL_STRING: str = "UNREADABLE"
UPSCALE_FACTOR: int = 3


def preprocess_tag(tag_image):
    """Preprocess the tag image so that it is easier for OCR to work."""
    gray = cv2.cvtColor(tag_image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    if processed[0, 0] == 0:
        processed = cv2.bitwise_not(processed)

    return processed


# Taken from Hamdan.
def extract_tag_id(
    tag_image: np.ndarray, reader: easyocr.Reader | None = None, ocr_conf: float = 0.05
) -> str:
    """Take an OpenCV image of an ear tag and extract the tag id from it."""

    if tag_image is None or tag_image.size == 0 or reader is None:
        return FAIL_STRING

    processed_image = preprocess_tag(tag_image)

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
        text for (_, text, conf) in results if float(conf) >= ocr_conf and text.strip()
    ]

    if valid_text:
        return "".join(valid_text).strip()

    return FAIL_STRING
