import psycopg2
from psycopg2 import OperationalError
import streamlit as st

def get_connection():
    try:
        conn = psycopg2.connect(
            user=st.secrets["POOLER_USER"],
            password=st.secrets["POOLER_PASSWORD"],
            host=st.secrets["POOLER_HOST"],
            port=st.secrets["POOLER_PORT"],
            dbname=st.secrets["POOLER_DATABASE"]
        )
        # st.success("✅ Connected successfully.")
        return conn

    except OperationalError as e:
        st.error("Sorry the credentials restricted to access due to the limitation.")
        st.exception(e)
        return None
