import datetime
import os
import pathlib
import tempfile

import cv2
import easyocr
import numpy as np
import pandas as pd
import pytz
import streamlit as st
from ultralytics import YOLO

import utils.ocr as ocr
import utils.st
from utils.database import db

COW_COLOR = (0, 255, 0)
STROKE_WIDTH = 2
DISPLAY_HEIGHT = 480
OFFSET = 20
TARGET_OD_MODEL = "yolo26n.pt"
TARGET_CLS_MODEL = "best_ceid.pt"

st.set_page_config(page_title="NCAI - Cattle Monitoring Dashboard", layout="centered")

utils.st.initialize_session_state()


@st.cache_resource
def load_resources(cow_det, tag_det, cow_cls):
    return (
        YOLO(cow_det),
        YOLO(tag_det),
        YOLO(cow_cls),
        easyocr.Reader(["en"], gpu=True),
    )


st.header("Register By Video/Image")

with st.expander("Models Configuration"):
    col1, col2, col3 = st.columns(3)

    cow_detector_dir = pathlib.Path("models/detector")
    tag_detector_dir = pathlib.Path("models")
    classifier_dir = pathlib.Path("models/classifier")

    def safe_get_index(options, target):
        try:
            return options.index(target)
        except ValueError:
            return 0

    cow_options = [
        f for f in os.listdir(cow_detector_dir) if (cow_detector_dir / f).is_file()
    ]
    cow_path = cow_detector_dir / col1.selectbox(
        "Cow Detector:", cow_options, index=safe_get_index(cow_options, TARGET_OD_MODEL)
    )

    tag_options = [
        f for f in os.listdir(tag_detector_dir) if (tag_detector_dir / f).is_file()
    ]
    tag_path = tag_detector_dir / col2.selectbox(
        "Tag Detector:",
        tag_options,
        index=safe_get_index(tag_options, "ear_tag_detector.pt"),
    )

    class_options = [
        f for f in os.listdir(classifier_dir) if (classifier_dir / f).is_file()
    ]
    class_path = classifier_dir / col3.selectbox(
        "Classifier:",
        class_options,
        index=safe_get_index(class_options, TARGET_CLS_MODEL),
    )

    cow_conf = st.slider("Cow Confidence:", 0.0, 1.0, 0.50)
    tag_conf = st.slider("Tag Confidence:", 0.0, 1.0, 0.30)
    ocr_conf = st.slider("OCR Confidence:", 0.0, 1.0, 0.05)
    use_ocr = st.checkbox("Use OCR")


uploaded_file = st.file_uploader(
    "Upload Video/Image:", type=["mp4", "avi", "mov", "jpg", "png"]
)

if uploaded_file:
    if st.session_state.get("last_uploaded_name") != uploaded_file.name:
        st.session_state.last_frame = None
        st.session_state.detected_cows = []
        st.session_state.has_registered = False
        st.session_state.last_uploaded_name = uploaded_file.name

if not st.session_state.has_registered:
    st_frame = st.empty()

    if uploaded_file:
        image_data = uploaded_file.getvalue()

        if st.session_state.last_frame is not None:
            st_frame.image(st.session_state.last_frame, use_container_width=True)
        else:
            st_frame.image(image_data, use_container_width=True)

        if st.button("Start Tracking & Inspection", use_container_width=True):
            cow_model, tag_model, cls_model, reader = load_resources(
                cow_path, tag_path, class_path
            )
            st.session_state.detected_cows = []

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(image_data)
                tmp_path = tmp.name

            video_stream = cv2.VideoCapture(tmp_path)

            while video_stream.isOpened():
                ret, frame = video_stream.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                scale = DISPLAY_HEIGHT / h
                resized = cv2.resize(frame, (int(w * scale), DISPLAY_HEIGHT))

                results = cow_model.track(
                    resized, persist=True, conf=cow_conf, verbose=False
                )

                if results[0].boxes is not None and results[0].boxes.id is not None:
                    cow_boxes = (
                        results[0].boxes.xyxy.cpu().numpy().astype(int)  # type: ignore
                    )
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)  # type: ignore

                    for cow_box, tid in zip(cow_boxes, track_ids):
                        x1, y1, x2, y2 = cow_box
                        cx1, cy1 = max(0, x1 - OFFSET), max(0, y1 - OFFSET)
                        cx2, cy2 = min(resized.shape[1], x2 + OFFSET), min(
                            resized.shape[0], y2 + OFFSET
                        )

                        cow_crop = resized[cy1:cy2, cx1:cx2]

                        tag_results = tag_model.predict(
                            cow_crop, conf=tag_conf, verbose=False
                        )
                        tag_text = "No Tag"

                        if (
                            tag_results[0].boxes is not None
                            and len(tag_results[0].boxes) > 0
                        ):
                            tag_boxes = (
                                tag_results[0].boxes.xyxy.cpu().numpy().astype(int)  # type: ignore
                            )

                            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in tag_boxes]
                            best_idx = np.argmax(areas)
                            tx1, ty1, tx2, ty2 = tag_boxes[best_idx]

                            tag_crop = cow_crop[ty1:ty2, tx1:tx2]
                            if tag_crop.size > 0:
                                tag_text = (
                                    ocr.extract_tag_id(tag_crop, reader, ocr_conf)
                                    if use_ocr
                                    else f"ID_{tid}"
                                )

                            cv2.rectangle(
                                resized,
                                (tx1 + cx1, ty1 + cy1),
                                (tx2 + cx1, ty2 + cy1),
                                COW_COLOR,
                                STROKE_WIDTH,
                            )

                        beh_res = cls_model.predict(cow_crop, verbose=False)
                        label = (
                            beh_res[0].names[beh_res[0].probs.top1]  # type: ignore
                            if beh_res[0].probs
                            else "Unknown"
                        )

                        st.session_state.detected_cows.append(
                            {"tag": tag_text, "behaviour": label}
                        )

                        cv2.rectangle(
                            resized, (cx1, cy1), (cx2, cy2), COW_COLOR, STROKE_WIDTH
                        )
                        cv2.putText(
                            resized,
                            f"Tag: {tag_text} | {label}",
                            (x1, cy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            COW_COLOR,
                            STROKE_WIDTH,
                        )

                processed_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                st.session_state.last_frame = processed_rgb
                st_frame.image(processed_rgb, use_container_width=True)

            video_stream.release()
            os.remove(tmp_path)
            st.success("Finished detection!")

    st.divider()
    if st.session_state.detected_cows and uploaded_file:
        st.subheader("Registration Review")

        with st.form("registration_form"):
            edited_df = st.data_editor(
                pd.DataFrame(st.session_state.detected_cows),
                num_rows="dynamic",
                hide_index=True,
                use_container_width=True,
                key="editor_inside_form",
            )

            submit_button = st.form_submit_button(
                "Finalize and Register", use_container_width=True
            )

            if submit_button:
                final_cows = [row for row in edited_df.to_dict("records")]  # type: ignore

                pk_tz = pytz.timezone("Asia/Karachi")
                now_aware = datetime.datetime.now(tz=pk_tz)

                cow_records = []
                for cow in final_cows:
                    existing = st.session_state.database[
                        st.session_state.database["tag"] == cow["tag"]
                    ]

                    if not existing.empty:
                        current_behaviours = list(existing["behaviours"].iloc[0])
                        current_images = list(existing["image_names"].iloc[0])

                        if current_behaviours == ["Unknown"]:
                            current_behaviours = [cow["behaviour"]]
                        else:
                            current_behaviours.append(cow["behaviour"])

                        if uploaded_file.name not in current_images:
                            current_images.append(uploaded_file.name)
                    else:
                        current_behaviours = [cow["behaviour"]]
                        current_images = [uploaded_file.name]

                    name = db.upload_image(uploaded_file)
                    current_images[-1] = name

                    cow_records.append(
                        {
                            "tag": cow["tag"],
                            "timestamp": now_aware.isoformat(),
                            "behaviours": current_behaviours,
                            "image_names": current_images,
                            "image_urls": [],
                        }
                    )

                new_df = pd.DataFrame(cow_records)
                db.sync_dataframe(new_df, "CowDatabase")

                st.success(f"Registered {len(final_cows)} cows to database.")
    else:
        st.info("No cows detected.")
