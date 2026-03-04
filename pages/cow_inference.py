import tempfile

import cv2
import easyocr
import numpy as np
import streamlit as st
from ultralytics import YOLO

import tag_ocr

COW_COLOR = (0, 255, 0)
STROKE_WIDTH = 2
DISPLAY_HEIGHT = 480
OFFSET = 20


from cow import Cow

detected_cows: list[Cow] = []


# Load the cow detection, tag detection, and cow classification
# models, as well as the OCR reader.
@st.cache_resource
def load_resources(
    cow_detection_model: str, tag_detection_model: str, cow_classification_model: str
):
    return (
        YOLO(cow_detection_model),
        YOLO(tag_detection_model),
        YOLO(cow_classification_model),
        easyocr.Reader(["en"], gpu=True),
    )


st.set_page_config(page_title="NCAI - Cattle Monitoring Dashboard", layout="centered")

st.header("Models Configuration")
st.text(
    "Here you can configure the models used as well as the confidence interval used for bounding boxes."
)

# Take the models to be used as input.
col1, col2, col3 = st.columns([0.333, 0.333, 0.333])

cow_path = col1.text_input("Cow Detection Model:", "models/detector/yolo26l.pt")
tag_path = col2.text_input("Tag Detection Model:", "models/ear_tag_detector.pt")
class_path = col3.text_input(
    "Cow Classification Model:", "models/classifier/best_final.pt"
)

cow_confidence_threshold = st.slider("Cow Detection Confidence:", 0.0, 1.0, 0.50)
tag_confidence_threshold = st.slider("Tag Detection Confidence:", 0.0, 1.0, 0.30)

# Load the model resources as needed.
try:
    cow_detection_model, tag_detection_model, cow_behaviour_model, reader = (
        load_resources(cow_path, tag_path, class_path)
    )
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

st.header("Register By Video/Image")

uploaded_file = st.file_uploader(
    "Upload Video/Image:", type=["mp4", "avi", "mov", "jpg", "png"]
)
if not uploaded_file:
    st.stop()


temp_file = tempfile.NamedTemporaryFile(delete=False)
temp_file.write(uploaded_file.read())
video_stream = cv2.VideoCapture(temp_file.name)


st_frame = st.empty()

if not st.button("Start Tracking & Inspection", width="stretch"):
    st.stop()

while video_stream.isOpened():
    ret, frame = video_stream.read()
    if not ret:
        break

    original_height, original_width = frame.shape[:2]
    scale = DISPLAY_HEIGHT / original_height

    resized_frame = cv2.resize(frame, (int(original_width * scale), DISPLAY_HEIGHT))

    cow_results = cow_detection_model.track(
        resized_frame, persist=True, conf=cow_confidence_threshold, verbose=False
    )

    if cow_results[0].boxes is None:
        continue

    cow_boxes = cow_results[0].boxes.xyxy.cpu().numpy().astype(int)  # type: ignore
    track_ids = (
        cow_results[0].boxes.id.cpu().numpy().astype(int)  # type: ignore
        if cow_results[0].boxes.id is not None
        else [0] * len(cow_boxes)
    )

    # For each detected cow...
    for cow_box, track_id in zip(cow_boxes, track_ids):
        x1, y1, x2, y2 = cow_box

        cx1, cy1 = max(0, x1 - OFFSET), max(0, y1 - OFFSET)
        cx2, cy2 = (
            min(resized_frame.shape[1], x2 + OFFSET),
            min(resized_frame.shape[0], y2 + OFFSET),
        )

        cow_cropped = resized_frame[cy1:cy2, cx1:cx2]

        tag_results = tag_detection_model.predict(
            cow_cropped, conf=tag_confidence_threshold, verbose=False
        )
        tags = (
            tag_results[0].boxes.xyxy.cpu().numpy().astype(int)  # type: ignore
            if tag_results[0].boxes
            else []
        )

        if len(tags) == 0:
            continue

        tag_y_min = int(np.min(tags[:, 1])) + cy1  # type: ignore
        # Interpolate cow bounding box based on tag location.
        refined_y1 = int(0.7 * y1 + 0.3 * tag_y_min)

        detected_tag_texts = []

        for tag_box in tags:
            tx1, ty1, tx2, ty2 = tag_box

            hx1 = int((tx1 + cx1) / scale)
            hy1 = int((ty1 + cy1) / scale)
            hx2 = int((tx2 + cx1) / scale)
            hy2 = int((ty2 + cy1) / scale)

            tag_cropped_hi_res = frame[hy1:hy2, hx1:hx2]

            if tag_cropped_hi_res.size > 0:
                tag_rgb = cv2.cvtColor(tag_cropped_hi_res, cv2.COLOR_BGR2RGB)
                tag_text = tag_ocr.extract_tag_id(tag_cropped_hi_res, reader)
                if tag_text != "UNREADABLE":
                    detected_tag_texts.append(tag_text)

        behaviour_results = cow_behaviour_model.predict(cow_cropped, verbose=False)
        behaviour_label = (
            behaviour_results[0].names[behaviour_results[0].probs.top1]  # type: ignore
            if behaviour_results[0].probs
            else "Unknown"
        )

        tag_display = (
            f"Tag: {detected_tag_texts[0]}" if detected_tag_texts else "Tag: Hidden"
        )
        if len(detected_tag_texts) > 0:
            detected_cows.append(Cow(detected_tag_texts[0], behaviour_label))

        # Draw cow bounding box.
        cv2.rectangle(
            resized_frame,
            (x1, refined_y1),
            (x2, y2),
            COW_COLOR,
            STROKE_WIDTH,
        )

        # Draw tag id display.
        cv2.putText(
            resized_frame,
            tag_display,
            (x1, refined_y1 - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

        # Draw cow behaviour display.
        cv2.putText(
            resized_frame,
            f"BEH: {behaviour_label}",
            (x1, refined_y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    st_frame.image(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

video_stream.release()

# TODO: Link these with DB.
print(detected_cows)
