import datetime

import humanize
import pandas as pd
import plotly.express as px
import pytz
import streamlit as st

st.set_page_config(layout="centered")

st.title("Cattle Dashboard")

df = st.session_state.database

health_overview_column, activity_distribution_column = st.columns(2)

with health_overview_column:
    if "Health" in df.columns:
        health_counts = df["Health"].value_counts().reset_index()
        health_counts.columns = ["Status", "Count"]

        fig = px.pie(
            health_counts,
            values="Count",
            names="Status",
            hole=0.4,
            color="Status",
            color_discrete_map={"Healthy": "#2ecc71", "Unhealthy": "#e74c3c"},
        )

        fig.update_layout(
            margin=dict(t=20, b=20, l=0, r=0),
            height=300,
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5
            ),
        )

        st.subheader("Herd Health Overview", anchor=False)
        st.plotly_chart(fig, use_container_width=True)

with activity_distribution_column:
    all_behaviours = []

    if "behaviours" in df.columns:
        for row_list in df["behaviours"]:
            if isinstance(row_list, list):
                all_behaviours.extend(row_list)

    if all_behaviours:
        target_behaviours = ["eat", "drink", "stand"]
        obs_series = pd.Series(all_behaviours).value_counts().reset_index()
        obs_series.columns = ["Behaviour", "Count"]
        totals = obs_series[obs_series["Behaviour"].isin(target_behaviours)]

        if not totals.empty:
            fig_activity = px.pie(
                totals,
                values="Count",
                names="Behaviour",
                hole=0.4,
                color="Behaviour",
                color_discrete_map={
                    "eat": "#3498db",
                    "drink": "#9b59b6",
                    "stand": "#f1c40f",
                },
            )
            fig_activity.update_layout(
                margin=dict(t=20, b=20, l=0, r=0),
                height=300,
                legend=dict(
                    orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5
                ),
            )
            st.subheader("Key Activity Distribution", anchor=False)
            st.plotly_chart(fig_activity, use_container_width=True)
        else:
            st.warning("No matches for 'eat', 'drink', or 'stand'.")
    else:
        st.warning("No behavioral data found.")

st.info("Monitor real-time behavioral data and manage your livestock registry below.")

total_cows_column, last_updated_column, pictures_stored_column = st.columns(3)

total_cows = len(df) if df is not None else 0
total_cows_column.metric("Total Cows Registered", total_cows)

local_tz = pytz.timezone("Asia/Karachi")

raw_time = st.session_state.get("last_registration")
if raw_time is None and "timestamp" in df.columns:
    latest_db_time = pd.to_datetime(df["timestamp"], format="ISO8601").max()
    if pd.notnull(latest_db_time):
        raw_time = latest_db_time

if raw_time is not None:
    if hasattr(raw_time, "to_pydatetime"):
        raw_time = raw_time.to_pydatetime()
    readable_time = humanize.naturaltime(datetime.datetime.now(tz=local_tz) - raw_time)
else:
    readable_time = "N/A"

last_updated_column.metric("Last Updated", readable_time)

if "image_names" in df.columns:
    total_pics = (
        df["image_names"]
        .apply(lambda x: len([img for img in x if img]) if isinstance(x, list) else 0)
        .sum()
    )
else:
    total_pics = 0

pictures_stored_column.metric("Pictures Stored", total_pics)

st.divider()

tag_col, img_col, db_col = st.columns(3)

if tag_col.button("Register By Tag", use_container_width=True):
    st.switch_page("pages/cow_registration.py")
if img_col.button("Register By Image", use_container_width=True, type="primary"):
    st.switch_page("pages/cow_inference.py")
if db_col.button("View Database", use_container_width=True):
    st.switch_page("pages/cow_database.py")
