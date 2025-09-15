import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="IRONMAN Splits — Smoke", layout="wide")
st.title("IRONMAN Splits — Smoke Test")

st.write("Working directory:", os.getcwd())
st.write("Python:", os.sys.version)
st.write("Streamlit:", st.__version__)
st.write("Exists long.csv:", os.path.exists("long.csv"))

if not os.path.exists("long.csv"):
    st.error("long.csv not found at repo root. Run the GitHub Action, then Restart the app.")
    st.stop()

try:
    df = pd.read_csv("long.csv")
    st.success(f"Loaded long.csv with {len(df):,} rows and {len(df.columns)} columns.")
    st.dataframe(df.head(20), use_container_width=True)
except Exception as e:
    st.error("Error reading long.csv")
    st.exception(e)
