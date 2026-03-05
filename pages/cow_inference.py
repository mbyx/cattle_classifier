import datetime
import os
import pathlib
import tempfile
from dataclasses import asdict

import cv2
import easyocr
import numpy as np
import pandas as pd
import pytz
import streamlit as st
from ultralytics import YOLO

import tag_ocr
from database import db

COW_COLOR = (0, 255, 0)
STROKE_WIDTH = 2
DISPLAY_HEIGHT = 480
OFFSET = 20


from cow import Cow

st.set_page_config(layout="centered")


# Load the cow detection, tag detection, and cow classification
# models, as well as the OCR reader.
@st.cache_resource
def load_resources(
    cow_detection_model: pathlib.Path,
    tag_detection_model: pathlib.Path,
    cow_classification_model: pathlib.Path,
):
    return (
        YOLO(cow_detection_model),
        YOLO(tag_detection_model),
        YOLO(cow_classification_model),
        easyocr.Reader(["en"], gpu=True),
    )


st.set_page_config(page_title="NCAI - Cattle Monitoring Dashboard", layout="centered")

st.header("Register By Video/Image")


with st.expander("Models Configuration"):
    st.text(
        "Here you can configure the models used as well as the confidence intervals used for bounding boxes."
    )

    # Take the models to be used as input.
    col1, col2, col3 = st.columns(3)

    cow_detector_dir = pathlib.Path("models") / "detector"
    tag_detector_dir = pathlib.Path("models")
    classifier_dir = pathlib.Path("models") / "classifier"

    cow_path = cow_detector_dir / col1.selectbox(
        "Cow Detection Model:",
        options=[
            f
            for f in os.listdir(cow_detector_dir)
            if os.path.isfile(cow_detector_dir / f)
        ],
        index=5,
        placeholder="yolo26n.pt",
    )
    tag_path = tag_detector_dir / col2.selectbox(
        "Tag Detection Model:",
        options=[
            f
            for f in os.listdir(tag_detector_dir)
            if os.path.isfile(tag_detector_dir / f)
        ],
        placeholder="ear_tag_detector.pt",
    )
    class_path = classifier_dir / col3.selectbox(
        "Cow Classification Model:",
        options=[
            f for f in os.listdir(classifier_dir) if os.path.isfile(classifier_dir / f)
        ],
        index=2,
        placeholder="best_final.pt",
    )

    cow_confidence_threshold = st.slider("Cow Detection Confidence:", 0.0, 1.0, 0.50)
    tag_confidence_threshold = st.slider("Tag Detection Confidence:", 0.0, 1.0, 0.30)

    use_ocr = st.checkbox("Use OCR", width="stretch")

# Load the model resources as needed.
try:
    cow_detection_model, tag_detection_model, cow_behaviour_model, reader = (
        load_resources(cow_path, tag_path, class_path)
    )
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


uploaded_file = st.file_uploader(
    "Upload Video/Image:", type=["mp4", "avi", "mov", "jpg", "png"]
)

st_frame = st.empty()
if st.session_state.get("last_frame") is not None and uploaded_file:
    st_frame.image(st.session_state.last_frame)


if uploaded_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    video_stream = cv2.VideoCapture(temp_file.name)

    if st.button("Start Tracking & Inspection", width="stretch"):
        st.session_state.last_frame = None
        st.session_state.detected_cows = []

        while video_stream.isOpened():
            ret, frame = video_stream.read()
            if not ret:
                break

            original_height, original_width = frame.shape[:2]
            scale = DISPLAY_HEIGHT / original_height

            resized_frame = cv2.resize(
                frame, (int(original_width * scale), DISPLAY_HEIGHT)
            )

            cow_results = cow_detection_model.track(
                resized_frame,
                persist=True,
                conf=cow_confidence_threshold,
                verbose=False,
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
                refined_y1 = int(0.7 * cy1 + 0.3 * tag_y_min)

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
                        if use_ocr:
                            tag_text = tag_ocr.extract_tag_id(
                                tag_cropped_hi_res, reader
                            )
                        else:
                            tag_text = "OCR OFF"
                        if tag_text != "UNREADABLE":
                            detected_tag_texts.append(tag_text)

                behaviour_results = cow_behaviour_model.predict(
                    cow_cropped, verbose=False
                )
                behaviour_label = (
                    behaviour_results[0].names[behaviour_results[0].probs.top1]  # type: ignore
                    if behaviour_results[0].probs
                    else "Unknown"
                )

                tag_display = (
                    f"Tag: {detected_tag_texts[0]}"
                    if detected_tag_texts
                    else "Tag: Hidden"
                )
                if len(detected_tag_texts) > 0:
                    new_cow = Cow(detected_tag_texts[0], behaviour_label)
                    st.session_state.detected_cows.append(new_cow)

                # Draw cow bounding box.
                cv2.rectangle(
                    resized_frame,
                    (cx1, refined_y1),
                    (cx2, cy2),
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

            processed_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(processed_rgb)

            st.session_state.last_frame = processed_rgb

        video_stream.release()


st.divider()
st.subheader("Registration Review")

if st.session_state.get("detected_cows") and uploaded_file:
    df_to_edit = pd.DataFrame([asdict(c) for c in st.session_state.detected_cows])

    edited_cows_df = st.data_editor(
        df_to_edit, num_rows="dynamic", hide_index=True, key="cow_editor"
    )

    if st.button("Register", use_container_width=True):
        final_cows = [Cow(**row) for row in edited_cows_df.to_dict("records")]  # type: ignore

        pk_tz = pytz.timezone("Asia/Karachi")
        now_aware = datetime.datetime.now(tz=pk_tz)

        cow_records = []
        for cow in final_cows:
            existing = st.session_state.database[
                st.session_state.database["tag"] == cow.tag
            ]

            if not existing.empty:
                current_behaviours = list(existing["behaviours"].iloc[0])
                current_images = list(existing["image_names"].iloc[0])

                if current_behaviours == ["Unknown"]:
                    current_behaviours = [cow.behaviour]
                else:
                    current_behaviours.append(cow.behaviour)

                if uploaded_file.name not in current_images:
                    current_images.append(uploaded_file.name)
            else:
                current_behaviours = [cow.behaviour]
                current_images = [uploaded_file.name]

            url, name = db.upload_image(uploaded_file)
            current_images[-1] = name

            cow_records.append(
                {
                    "tag": cow.tag,
                    "timestamp": now_aware.isoformat(),
                    "behaviours": current_behaviours,
                    "image_names": current_images,
                    "image_urls": [],
                }
            )

        new_df = pd.DataFrame(cow_records)
        db.sync_dataframe(new_df, "CowDatabase")

        st.success(f"Registered {len(final_cows)} cows to database.")

        st.session_state.last_registration = now_aware
        st.session_state.detected_cows = []

else:
    st.info("No cows detected yet. Start the tracking to populate this list.")
