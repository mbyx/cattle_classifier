import tempfile

import cv2
import streamlit as st
from ultralytics import YOLO

ANNOTATION_COLOR: tuple[int, int, int] = (0, 255, 0)
STROKE_WIDTH: int = 2
IMAGE_HEIGHT: int = 480

BEHAVIOR_MAP: dict[str, str] = {
    "drink": "drink",
    "eat": "eat",
    "ruminate": "eat",
    "lie": "neither",
    "stand": "neither",
}

st.set_page_config(page_title="NCAI - Cattle Behaviour Classifier", layout="centered")
st.title("NCAI - Cattle Behaviour Classifier")
st.text("Interface to classify cattle behaviour for NCAI.")


st.subheader("Settings")
tracker_path: str = st.text_input("Path to Tracker Model:", "models/yolov8n.pt")
classifier_path: str = st.text_input("Path to Classifier Model:", "models/best.pt")
confidence_threshold: float = st.slider("Detection Confidence", 0.0, 1.0, 0.5)


@st.cache_resource
def load_models(tracker_path: str, classifier_path: str) -> tuple[YOLO, YOLO]:
    tracker = YOLO(tracker_path)
    classifier = YOLO(classifier_path)
    return (tracker, classifier)


st.subheader("Tracker & Classifier")
try:
    model_tracker, model_classifier = load_models(tracker_path, classifier_path)
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


uploaded_file = st.file_uploader("Upload Cattle Video:", type=["mp4", "avi", "mov"])

if not uploaded_file:
    st.info("Please upload a video file to continue.")
    st.stop()

st.success("File successfully uploaded!")

temp_file = tempfile.NamedTemporaryFile(delete=False)
temp_file.write(uploaded_file.read())

video_stream = cv2.VideoCapture(temp_file.name)

# Create a placeholder to store the video frames.
_, col2, _ = st.columns([0.05, 0.9, 0.05])
with col2:
    st_frame = st.empty()

if not st.button("Start Tracking and Behavior Classification", width="stretch"):
    st.stop()

while video_stream.isOpened():
    ret, frame = video_stream.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    aspect_ratio = width / height
    desired_width = int(IMAGE_HEIGHT * aspect_ratio)
    resized_frame = cv2.resize(frame, (desired_width, IMAGE_HEIGHT))

    results = model_tracker.track(
        resized_frame,
        persist=True,
        conf=confidence_threshold,
        verbose=False,
        tracker="bytetrack.yaml",
    )

    if not results[0].boxes:
        continue

    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)  # type: ignore
    track_ids = (
        results[0].boxes.id.cpu().numpy().astype(int)  # type: ignore
        if results[0].boxes.id is not None
        else [0] * len(boxes)
    )

    for box, track_id in zip(boxes, track_ids):
        x1, y1, x2, y2 = box

        cattle_cropped = resized_frame[y1:y2, x1:x2]

        if cattle_cropped.size == 0:
            continue

        behavior_results = model_classifier.predict(cattle_cropped, verbose=False)

        raw_behavior = behavior_results[0].names[
            behavior_results[0].probs.top1  # type: ignore
        ]

        final_behavior = BEHAVIOR_MAP.get(raw_behavior, raw_behavior)
        label = f"ID: {track_id} ({final_behavior})"

        cv2.rectangle(
            resized_frame,
            (x1, y1),
            (x2, y2),
            ANNOTATION_COLOR,
            STROKE_WIDTH,
        )
        cv2.putText(
            resized_frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,  # Font scale.
            ANNOTATION_COLOR,
            STROKE_WIDTH,
        )

    # OpenCV by default uses BGR, we need to convert to RGB.
    frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    st_frame.image(frame_rgb, use_container_width=False)

video_stream.release()
