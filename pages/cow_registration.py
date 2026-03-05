import datetime

import pandas as pd
import pytz
import streamlit as st

from database import db

st.set_page_config(layout="centered")

left_spacer, center_column, right_spacer = st.columns([0.1, 0.8, 0.1])


with center_column:
    with st.form("cow_registration_form", clear_on_submit=True, border=True):
        st.header("Tag Registration", divider="gray", text_alignment="center")
        st.markdown("###### **Please enter the Tag ID:**")
        tag = st.text_input(
            "Tag ID Input", label_visibility="collapsed", placeholder="e.g. 12345"
        )
        submit = st.form_submit_button(
            "Register Cow", use_container_width=True, type="primary"
        )

    if submit:
        tag = tag.strip()
        if not tag:
            st.error("Validation Error: The Tag ID cannot be empty.")
        elif len(tag) < 3:
            st.warning("Validation Error: Tag ID is too short to be valid.")
        else:
            existing_cow = pd.DataFrame()

            if (
                not st.session_state.database.empty
                and "tag" in st.session_state.database.columns
            ):
                existing_cow = st.session_state.database[
                    st.session_state.database["tag"] == tag
                ]

            if existing_cow.empty:
                behaviours_list = ["Unknown"]
                image_names_list = []
                image_urls_list = []

                new_record = {
                    "tag": tag,
                    "behaviours": behaviours_list,
                    "image_names": image_names_list,
                    "image_urls": image_urls_list,
                }

                new_df = pd.DataFrame([new_record])
                db.sync_dataframe(new_df, "CowDatabase")

                st.session_state.last_registration = datetime.datetime.now(
                    tz=pytz.timezone("Asia/Karachi")
                )
                st.success(f"Cow with tag {tag} successfully registered!")
            else:
                st.warning("Cow is already registered in database!")
