import streamlit as st

from database import db

st.set_page_config(page_title="NCAI - Cattle Monitoring")

# Global initialization
if "database" not in st.session_state:
    with st.spinner("Analyzing Herd Health..."):
        st.session_state.database = db.fetch_processed_data("CowDatabase")
if "detected_cows" not in st.session_state:
    st.session_state.detected_cows = []
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None
if "last_registration" not in st.session_state:
    st.session_state.last_registration = None

# Navigation
pg = st.navigation(
    [
        st.Page("pages/home.py", title="Home", default=True),
        st.Page("pages/cow_inference.py", title="Cow Inference"),
        st.Page("pages/cow_registration.py", title="Cow Registration"),
        st.Page("pages/cow_database.py", title="Cow Database"),
    ],
    position="top",
)
pg.run()
