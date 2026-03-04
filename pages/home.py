import streamlit as st

st.title("Cattle Dashboard", text_alignment="center")

st.info("Monitor real-time behavioral data and manage your livestock registry below.")

col1, col2, col3 = st.columns(3)
# TODO: Link these with DB.
col1.metric("Total Cows Registered", "N/A")
col2.metric("Last Update", "10m ago")
col3.metric("Database Status", "Online")

st.divider()

col1, col2, col3 = st.columns(3)

if col1.button("Register By Tag", width="stretch"):
    st.switch_page("pages/cow_registration.py")
if col2.button("Register By Image", width="stretch", type="primary"):
    st.switch_page("pages/cow_inference.py")
if col3.button("View Database", width="stretch"):
    st.switch_page("pages/cow_database.py")
