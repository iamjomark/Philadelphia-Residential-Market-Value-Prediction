import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from conn import get_connection
import datetime
from custom_components import pre_preprocessing, normalize_ordinal_columns
from sklearn.pipeline import Pipeline
import joblib
import pickle
from huggingface_hub import hf_hub_download

plt.rcParams['text.usetex'] = False


# def tab2_shap_local(pipeline, transformed_input):
def app():
    tab1, tab2 = st.tabs(["📁 Prediction", "📈 Results"])

    with tab1:
        st.write("Enter property details below to get a fast and fair **market value estimate**.")
        if "pipeline" not in st.session_state:
            try:
                file_path = hf_hub_download(
                    repo_id="faaza/house-price-pipeline",  
                    filename="pipelines.joblib",      
                    repo_type="model"                  
                )

                pipeline = joblib.load(file_path)

                st.session_state["pipeline"] = pipeline
                st.session_state["feature_names"] =  pipeline.named_steps["preprocessing"].get_feature_names_out()
                    # st.success("✅ Model and preprocessing loaded successfully.")
            except Exception as e:
                st.error(f"❌ Failed to load pipeline: {e}")
                st.stop()

        pipeline = st.session_state["pipeline"]
        feature_names = st.session_state["feature_names"]

        # Input form
        # === CATEGORICAL OPTIONS ===
        type_heater_map = {
             'Hot Air (Ducts)': 'A', 'Hot Water (Radiators)': 'B', 'Electric Baseboard': 'C', 'Heat Pump (Outside Unit)': 'D', 'Other': 'E',  'Radiant': 'G', 'No Heater': 'H'
        }
        basements_map = {
            'Full – Unknown Finish': 'D', 'Full Unfinished': 'C', 'Unknown Size - Finished': 'I',
            'Full Finished': 'A', 'Partial - Unknown Finish': 'H', 'Partial Unfinished': 'G', 'Partial Finished': 'E',
            'Full Semi-Finished': 'B', 'Unknown Size - Unfinished': 'J', 'Partial Semi-Finished': 'F', 'No Basement': '0'
        }
        topography_map = {
             'Above Street Level': 'A', 'Below Street Level ': 'B', 'Flood Plain': 'C', 'Rocky': 'D', 'Level': 'F', 'Other (Not Identified)': 'E',
        }
        garage_type_map = {
            'Built-in/ Basement': 'A', 'Attached Garage': 'B', 'Detached Garage': 'C',
            'Converted': 'F', 'Self Park': 'S', 'Attendant': 'T', 'No Garage': '0'
        }
        parcel_shape_map = {
            'Irregular': 'A', 'Grossly Irregular': 'B', 'Triangular': 'C', 'Right of way': 'D', 'Rectangular/Default': 'E'
        }
        view_type_map = {
            'Cityscape/ Skyline': 'A', 'Flowing Water': 'B', 'Park/Green Area': 'C',
            'Commercial': 'D', 'Industrial': 'E', 'Edifice/ Landmark': 'H', 'Typical/ Other': 'I', 'Not Applicable': '0'
        }
        zoning_map = {
            'Multi-Family RM1': 'RM1', 'Multi-Family RM2': 'RM2', 'Multi-Family RM3': 'RM3',
            'Multi-Family RM4': 'RM4', 'Mixed Use RMX1': 'RMX1', 'Mixed Use RMX2': 'RMX2',
            'Mixed Use RMX3': 'RMX3', 'Single-Family Attached RSA': 'RSA', 'Single-Family Attached RSA1': 'RSA1',
            'Single-Family Attached RSA2': 'RSA2', 'Single-Family Attached RSA3': 'RSA3', 'Single-Family Attached RSA4': 'RSA4',
            'Single-Family Attached RSA5': 'RSA5', 'Single-Family Detached RSD1': 'RSD1', 'Single-Family Detached RSD2': 'RSD2',
            'Single-Family Detached RSD3': 'RSD3', 'Two-Family Attached RTA1': 'RTA1'
        }
        interior_condition_map = {
            'Newer Construction': 1, 'Rehabilitated': 2, 'Above Average': 3, 'Rehabilitated': 4, 
            'Average': 5, 'Below Average': 6, 'Vacant': 7, 'Sealed': 8, 'Not Applicable': 0
        } 
        exterior_condition_map = {
            'Newer Construction': 1, 'Rehabilitated': 2, 'Above Average': 3, 'Rehabilitated': 4, 
            'Average': 5, 'Below Average': 6, 'Vacant': 7, 'Sealed': 8, 'Not Applicable': 0
        }

        # === FORM ===
        st.title("🏠 House Price Prediction")

        type_heater_display = st.selectbox("Type of Heater", list(type_heater_map.keys()))
        basements_display = st.selectbox("Basement Type", list(basements_map.keys()))
        topography_display = st.selectbox("Topography", list(topography_map.keys()))
        garage_type_display = st.selectbox("Garage Type", list(garage_type_map.keys()))
        parcel_shape_display = st.selectbox("Parcel Shape", list(parcel_shape_map.keys()))
        view_type_display = st.selectbox("View Type", list(view_type_map.keys()))
        zoning_display = st.selectbox("Zoning Type", list(zoning_map.keys()))

        homestead_exemption = st.number_input("Homestead Exemption", min_value=0.0, step=100.0, value=0.0)
        geographic_ward = st.number_input("Geographic Ward", min_value=1.0, max_value=66.0, step=1.0, value=1.0)
        interior_condition = st.selectbox("Interior Condition", list(interior_condition_map.keys()))
        exterior_condition = st.selectbox("Exterior Condition", list(interior_condition_map.keys()))

        off_street_open = st.number_input("Off-Street Open Parking", min_value=0.0, max_value=30.0, step=1.0, value=0.0)
        fireplaces = st.number_input("Number of Fireplaces", min_value=0.0, max_value=10.0, step=1.0, value=0.0)
        year_built = st.number_input("Year Built", min_value=1800, max_value=2025, step=1, value=2000)

        number_of_bathrooms = st.number_input("Number of Bathrooms", min_value=0.0, max_value=12.0, step=0.5, value=1.0)
        number_of_bedrooms = st.number_input("Number of Bedrooms", min_value=0.0, max_value=21.0, step=1.0, value=3.0)
        number_of_rooms = st.number_input("Number of Rooms", min_value=0.0, max_value=32.0, step=1.0, value=6.0)
        number_stories = st.number_input("Number of Stories", min_value=0.0, max_value=40.0, step=1.0, value=2.0)

        total_livable_area = st.number_input("Total Livable Area (sqft)", min_value=0.0, step=10.0, value=1500.0)
        total_area = st.number_input("Total Lot Area", min_value=0.0, step=10.0, value=2000.0)
        frontage = st.number_input("Frontage (ft)", min_value=0.0, step=1.0, value=16.0)
        depth = st.number_input("Depth (ft)", min_value=0.0, step=1.0, value=77.5)

        # === Predict Button ===
        if st.button("🔮 Predict House Value"):
            # Map display → value
            df_model = pd.DataFrame([{
                'type_heater': type_heater_map[type_heater_display],
                'basements': basements_map[basements_display],
                'topography': topography_map[topography_display],
                'homestead_exemption': homestead_exemption,
                'geographic_ward': geographic_ward,
                'garage_type': garage_type_map[garage_type_display],
                'parcel_shape': parcel_shape_map[parcel_shape_display],
                'view_type': view_type_map[view_type_display],
                'interior_condition': interior_condition_map[interior_condition],
                'exterior_condition': exterior_condition_map[exterior_condition],
                'zoning': zoning_map[zoning_display],
                'off_street_open': off_street_open,
                'fireplaces': fireplaces,
                'year_built': year_built,
                'number_of_bathrooms': number_of_bathrooms,
                'number_of_bedrooms': number_of_bedrooms,
                'total_livable_area': total_livable_area,
                'number_of_rooms': number_of_rooms,
                'number_stories': number_stories,
                'total_area': total_area,
                'frontage': frontage,
                'depth': depth
            }])

            st.subheader("🔍 Input Summary")
            st.dataframe(df_model)

            try:
                pipeline = st.session_state["pipeline"]
                df_model = pre_preprocessing(df_model)
                df_model = normalize_ordinal_columns(df_model)
                prediction = pipeline.predict(df_model)

                # Store results in session state
                st.session_state["df_model"] = df_model
                st.session_state["prediction"] = float(prediction[0])
                
                # === Show results ===
                st.subheader("💰 Prediction Result")
                st.success(f"Estimated House Market Value: **${prediction[0]:,.2f}**")

            except Exception as e:
                st.error(f"❌ An error occurred during prediction:\n{e}")
        
        if "prediction" in st.session_state and "df_model" in st.session_state:
            if st.button("💾 Save to History"):
                if not st.session_state.get("authenticated", False):
                    st.warning("⚠️ You need to login first to save prediction history.")
                else:
                    try:
                        conn = get_connection()
                        if conn is None:
                            st.error("⚠️ Could not connect to database.")
                        else:
                            cur = conn.cursor()
                            now = datetime.datetime.now()
                            username = st.session_state.get("username", None)

                            if username is None:
                                st.error("⚠️ Username not found in session.")
                            else:
                                cur.execute("SELECT user_id FROM user_dex WHERE name = %s", (username,))
                                result = cur.fetchone()
                                if result is None:
                                    st.error(f"⚠️ User '{username}' not found in database.")
                                else:
                                    user_id = result[0]
                                    input_df = st.session_state["df_model"]
                                    prediction = st.session_state["prediction"]    

                                    values = (
                                        user_id,
                                        now,
                                        input_df.at[0, 'type_heater'],
                                        input_df.at[0, 'basements'],
                                        input_df.at[0, 'topography'],
                                        input_df.at[0, 'homestead_exemption'],
                                        input_df.at[0, 'geographic_ward'],
                                        input_df.at[0, 'garage_type'],
                                        input_df.at[0, 'parcel_shape'],
                                        input_df.at[0, 'view_type'],
                                        input_df.at[0, 'interior_condition'],
                                        input_df.at[0, 'exterior_condition'],
                                        input_df.at[0, 'zoning'],
                                        input_df.at[0, 'off_street_open'],
                                        input_df.at[0, 'has_fireplace'],
                                        input_df.at[0, 'year_built'],
                                        input_df.at[0, 'number_of_bathrooms'],
                                        input_df.at[0, 'number_of_bedrooms'],
                                        input_df.at[0, 'total_livable_area'],
                                        input_df.at[0, 'number_of_rooms'],
                                        input_df.at[0, 'number_stories'],
                                        input_df.at[0, 'total_area'],
                                        input_df.at[0, 'frontage'],
                                        input_df.at[0, 'depth'],
                                        prediction
                                    )

                                    insert_query = """
                                        INSERT INTO history_housing (
                                            user_id, date, type_heater, basements, topography,
                                            homestead_exemption, geographic_ward, garage_type, parcel_shape, view_type,
                                            interior_condition, exterior_condition, zoning, off_street_open, fireplaces,
                                            year_built, number_of_bathrooms, number_of_bedrooms, total_livable_area,
                                            number_of_rooms, number_stories, total_area, frontage, depth, predicted_value
                                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                    """

                                    cur.execute(insert_query, values)
                                    conn.commit()
                                    st.success("✅ Prediction saved to history.")
                            cur.close()
                            conn.close()
                    except Exception as e:
                        st.error(f"❌ Failed to save prediction history: {e}")
    with tab2:
        if "df_model" in st.session_state and "pipeline" in st.session_state:
            try:
                pipeline = st.session_state["pipeline"]  # full pipeline with preprocessing + model
                df_model = st.session_state["df_model"]  # raw input data
                prediction = pipeline.predict(df_model)

                preprocessor = pipeline.named_steps["preprocessing"]
                regressor = pipeline.named_steps["regressor"]
                X_transformed = preprocessor.transform(df_model)
                explainer = shap.Explainer(
                    regressor,
                    feature_names=preprocessor.get_feature_names_out()
                )
                shap_values = explainer(X_transformed)

                # === Visualisasi SHAP ===
                st.markdown(
                    "<h3 style='text-align: center;'>SHAP Water Plot</h3>",
                    unsafe_allow_html=True
                )
                fig1, ax1 = plt.subplots(figsize=(10, 5))
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig1)
                
                st.markdown("""
                #### What You’re Seeing in the SHAP Waterfall Plot:
                - Red bars mean the feature is **pushing the prediction higher** — it’s boosting the outcome.
                - Blue bars mean the feature is **pulling the prediction lower** — it’s holding the outcome back.
                """)

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.write("")
                    st.write("")
                    st.write("")
                    st.markdown(
                        "<h4 style='text-align: center;'>SHAP Values Data</h3>",
                        unsafe_allow_html=True
                    )
                    feature_names = preprocessor.get_feature_names_out()
                    shap_vals_sample = shap_values[0].values
                    df_shap = pd.DataFrame({
                        'Feature': feature_names,
                        'SHAP Value': shap_vals_sample,
                        'Impact': ['Increase' if v > 0 else 'Decrease' for v in shap_vals_sample]
                    }).sort_values(by='SHAP Value', key=abs, ascending=False)
                    st.dataframe(df_shap, use_container_width=True)


            except Exception as e:
                st.error(f"❌ Failed to generate SHAP explanation: {e}")

        else:
            st.info("ℹ️ Make a prediction first to see SHAP explanation.")


if __name__ == "__main__":
    app()
