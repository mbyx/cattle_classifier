import pathlib

import streamlit as st

import utils.st

st.set_page_config(
    page_title="NCAI - Cattle Monitoring", initial_sidebar_state="collapsed"
)

utils.st.initialize_session_state()

pages_directory = pathlib.Path("pages")

pg = st.navigation(
    [
        st.Page(pages_directory / "home.py", title="Home", default=True),
        st.Page(pages_directory / "cow_inference.py", title="Cow Inference"),
        st.Page(pages_directory / "cow_registration.py", title="Cow Registration"),
        st.Page(pages_directory / "cow_database.py", title="Cow Database"),
    ],
    position="top",
)

pg.run()
