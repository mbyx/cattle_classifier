import streamlit as st

pg = st.navigation(
    [
        st.Page("pages/home.py", title="Home"),
        st.Page("pages/cow_inference.py", title="Cow Inference"),
        st.Page("pages/cow_registration.py", title="Cow Registration"),
        st.Page("pages/cow_database.py", title="Cow Database"),
    ],
    position="top",
)
pg.run()
