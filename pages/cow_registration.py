import streamlit as st

from cow import Cow

# Centering the layout
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
        if not tag.strip():
            st.error("Validation Error: The Tag ID cannot be empty.")
        elif len(tag) < 3:
            st.warning("Validation Error: Tag ID is too short to be valid.")
        else:
            st.success(f"Cow ID: {tag} registered successfully!")
            # TODO: Link these with DB.
            print(f"[DB] APPEND {Cow(tag, behaviour='Unknown')}")
