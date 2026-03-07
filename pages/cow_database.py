import datetime

import humanize
import pandas as pd
import pytz
import streamlit as st
from streamlit_extras.st_keyup import st_keyup

import utils.st
from utils.database import db

utils.st.initialize_session_state()

st.header("Cow Database", divider="gray")
st.set_page_config(layout="wide")

df = st.session_state.database

if df is None or df.empty:
    st.error("Data file not found or database is empty.")
    st.stop()


search_tag = st_keyup("Search by Tag ID", placeholder="e.g: 451222")


df_to_show = df.drop(columns=["Total Behaviours", "Distinct count"], errors="ignore")

if search_tag:
    filtered_df = df_to_show[
        df_to_show["tag"].astype(str).str.contains(search_tag, case=False)
    ]
else:
    filtered_df = df_to_show

with st.expander("Raw Data"):
    selection = st.dataframe(
        filtered_df,
        on_select="rerun",
        selection_mode="single-row",
        use_container_width=True,
        hide_index=True,
    )

selected_data = None


if selection.selection.rows:  # type: ignore
    selected_row_index = selection.selection.rows[0]  # type: ignore
    selected_data = filtered_df.iloc[selected_row_index]
elif not filtered_df.empty:
    selected_data = filtered_df.iloc[0]

if selected_data is not None:
    tag_id = selected_data["tag"]
    st.header(f"Profile: Tag #{tag_id}")

    col1, col2, col3 = st.columns(3)

    col1.metric("Health Status", selected_data.get("Health", "Unknown"))

    img_list = selected_data.get("image_names", [])
    img_list = [image for image in img_list if image != ""]
    total_pics = len(img_list) if isinstance(img_list, list) else 0
    col2.metric("Pictures Stored", total_pics)

    raw_ts = selected_data.get("timestamp")
    if pd.notnull(raw_ts):
        db_time = pd.to_datetime(raw_ts, format="ISO8601")
        readable_time = humanize.naturaltime(
            datetime.datetime.now(tz=pytz.timezone("Asia/Karachi")) - db_time
        )
    else:
        readable_time = "N/A"

    col3.metric("Last Updated", readable_time)

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Image Gallery")
        image_options = selected_data.get("image_names", [])
        image_options = [image for image in image_options if image != ""]
        image_urls = selected_data.get("image_urls", [])

        if isinstance(image_options, list) and len(image_options) > 0:
            selected_img_name = st.selectbox("Select Image:", options=image_options)

            if selected_img_name:
                with st.container(border=True):
                    frame = st.empty()
                    frame.image(
                        db.get_private_image_url(selected_img_name),
                        caption=f"Filename: {selected_img_name}",
                        use_container_width=True,
                    )
            else:
                st.error("URL for this image is missing from the database.")

        else:
            st.warning("No images found for this Tag ID.")

    with c2:
        st.subheader("Behavioural Analysis")

        actual_behaviours = selected_data.get("behaviours", [])

        if isinstance(actual_behaviours, list) and len(actual_behaviours) > 0:
            options = ["eat", "drink", "stand", "Unknown"]
            selected_filter = st.multiselect(
                "Filter Behaviours", options=options, default=options
            )

            counts = {b: actual_behaviours.count(b) for b in selected_filter}

            chart_df = pd.DataFrame(
                list(counts.items()), columns=["Behaviour", "Count"]
            )

            st.bar_chart(chart_df, x="Behaviour", y="Count", color="#3498db")
        else:
            st.info("No behaviour data recorded.")

else:
    st.warning("No matches found.")
