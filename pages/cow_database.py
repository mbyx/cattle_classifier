import pandas as pd
import streamlit as st

st.header("Cow Database", divider="gray")

try:
    # TODO: Link these with DB.
    df_raw = pd.read_csv("dummy_data.csv", sep=", ", engine="python")
except FileNotFoundError:
    st.error("Data file not found. Please ensure 'dummy_data.csv' exists.")
    st.stop()

df_pivot = pd.crosstab(df_raw["Tag"], df_raw["Behaviour"])
df_pivot["Total Behaviours"] = df_pivot.sum(axis=1)
df_display = df_pivot.reset_index()

st.info(
    "Select a row in the table to view the behavior distribution for that specific cow."
)

selection = st.dataframe(
    df_display,
    on_select="rerun",
    selection_mode="single-row",
    use_container_width=True,
    hide_index=True,
)

if selection.selection.rows:  # type: ignore
    selected_row_index = selection.selection.rows[0]  # type: ignore
    selected_tag = df_display.iloc[selected_row_index]["Tag"]

    st.subheader(f"Behavior Profile: Tag #{selected_tag}")

    chart_cols = [c for c in df_display.columns if c not in ["Tag", "Total Behaviours"]]
    chart_data = df_display.iloc[selected_row_index][chart_cols]

    st.bar_chart(chart_data)
