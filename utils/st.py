import streamlit as st

from utils.database import db

TABLE_NAME: str = "CowDatabase"


def initialize_session_state() -> None:
    """Initialize all used session state."""

    if "database" not in st.session_state:
        with st.spinner("Fetching Data..."):
            st.session_state.database = db.fetch_processed_data(TABLE_NAME)

    if "detected_cows" not in st.session_state:
        st.session_state.detected_cows = []

    if "last_frame" not in st.session_state:
        st.session_state.last_frame = None

    if "has_registered" not in st.session_state:
        st.session_state.has_registered = False
