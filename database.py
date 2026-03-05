import uuid

import pandas as pd
import streamlit as st
from st_supabase_connection import SupabaseConnection


class DBManager:
    def __init__(self):

        self.conn = st.connection("supabase", type=SupabaseConnection)

    def fetch_processed_data(self, table_name: str) -> pd.DataFrame:
        try:

            response = self.conn.table(table_name).select("*").execute()
            df = pd.DataFrame(response.data)

            if df.empty:
                return pd.DataFrame()

            df["behaviours"] = df["behaviours"].apply(
                lambda x: x if isinstance(x, list) else []
            )

            df["Total Behaviours"] = df["behaviours"].apply(len)
            df["Distinct count"] = df["behaviours"].apply(lambda x: len(set(x)))

            activity_threshold = df["Total Behaviours"].quantile(0.25)
            if pd.isna(activity_threshold):
                activity_threshold = 0

            def calculate_health(row):
                if (
                    row["Total Behaviours"] < activity_threshold
                    or row["Distinct count"] < 2
                ):
                    return "Unhealthy"
                return "Healthy"

            df["Health"] = df.apply(calculate_health, axis=1)
            return df

        except Exception as e:
            st.error(f"Database Error: {e}")
            return pd.DataFrame()

    def sync_dataframe(self, df: pd.DataFrame, table_name: str):
        """
        Sends data to Supabase. Since 'behaviours' is a list in the DF,
        upsert() handles it as a JSONB array automatically.
        """
        if df.empty:
            return

        records = df.to_dict(orient="records")
        try:

            self.conn.table(table_name).upsert(records).execute()  # type: ignore
            st.cache_data.clear()

            st.cache_data.clear()

            updated_df = self.fetch_processed_data(table_name)
            st.session_state.database = updated_df

            st.toast("Database Synced!")
        except Exception as e:
            st.error(f"Sync failed: {e}")

    def upload_image(
        self, uploaded_file, bucket_name: str = "cow_images"
    ) -> tuple[str, str]:
        try:
            file_extension = uploaded_file.name.split(".")[-1]
            unique_name = f"{uuid.uuid4().hex}.{file_extension}"

            self.conn.client.storage.from_(bucket_name).upload(
                path=unique_name,
                file=uploaded_file.getvalue(),
                file_options={"content-type": uploaded_file.type},
            )

            public_url = self.conn.client.storage.from_(bucket_name).get_public_url(
                unique_name
            )

            return (public_url, unique_name)

        except Exception as e:
            st.error(f"Storage Upload Failed: {e}")
            return ("", "")

    def get_private_image_url(
        self, file_path: str, bucket_name: str = "cow_images"
    ) -> str:
        res = self.conn.client.storage.from_(bucket_name).create_signed_url(
            file_path, 3600
        )
        return res["signedURL"]


db = DBManager()
