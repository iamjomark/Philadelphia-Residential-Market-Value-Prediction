import io
import streamlit as st
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from custom_components import pre_preprocessing, normalize_ordinal_columns
from io import BytesIO
from sklearn.pipeline import Pipeline
import joblib
from huggingface_hub import hf_hub_download

# === Utils ===
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')
    output.seek(0)
    return output

# === Load model and pipeline ===
def load_pipeline():
    try:
        # pipeline = joblib.load('pipelines.joblib')
        file_path = hf_hub_download(
                    repo_id="faaza/house-price-pipeline",  
                    filename="pipelines.joblib",      
                    repo_type="model"                  
                )

        pipeline = joblib.load(file_path)

        st.session_state["pipeline"] = pipeline
        st.session_state["feature_names"] =  pipeline.named_steps["preprocessing"].get_feature_names_out()
        return pipeline
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

# === App Main ===
def app():
    tab1, tab2 = st.tabs(["📁 Prediction", "📄 Docs"])

    pipeline = st.session_state.get("pipeline") or load_pipeline()
    if pipeline is None:
        return

    with tab1:
        required_cols = [
            "type_heater", "basements", "topography", "homestead_exemption",
            "geographic_ward", "garage_type", "parcel_shape", "view_type",
            "interior_condition", "exterior_condition", "zoning", "off_street_open",
            "year_built", "number_of_bathrooms", "number_of_bedrooms", "total_livable_area",
            "number_of_rooms", "number_stories", "total_area", "frontage", "depth", "fireplaces"
        ]
        st.markdown(f"Upload an Excel or CSV file with property data to predict house values. **Required columns**: `{', '.join(required_cols)}`.")

        uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
                st.subheader("📄 Uploaded Data Preview")
                st.dataframe(df.head())

                missing = [col for col in required_cols if col not in df.columns]
                if missing:
                    st.error(f"⚠️ Missing required columns: {missing}")
                    return

                if st.button("🔮 Predict All"):
                    prediction = pipeline.predict(df)
                    df["predicted_value"] = prediction
                    st.session_state["prediction_df"] = df

                    st.subheader("🎯 Prediction Results")
                    st.dataframe(df[required_cols + ["predicted_value"]])

                    st.download_button(
                        label="📥 Download Results",
                        data=to_excel(df),
                        file_name="house_price_predictions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

            except Exception as e:
                st.error(f"❌ Failed to process file: {e}")

    with tab2:
        st.write("### How to Predict (Batch Housing Price Dataset)")
        st.markdown("""
            1. Go to the **Prediction** tab to upload your CSV file.
            2. Make sure your file includes all required columns with the correct format.
            3. Once uploaded, the prediction results will be displayed automatically in the same tab.
            4. Click the **Save** button to download the results. Your browser will trigger a download popup automatically.
            """)

if __name__ == "__main__":
    app()