import streamlit as st
import joblib
import numpy as np
import streamlit.components.v1 as components
import os

# Assuming the files are in the same directory as the script
base_dir = os.path.dirname(__file__)
scaler_path = os.path.join(base_dir, 'scaler.pkl')
model_path = os.path.join(base_dir, 'model.pkl')

# Load the scaler and model
scaler = joblib.load(scaler_path)
model = joblib.load(model_path)

st.title("House Prediction App")

# Read the HTML file content
with open(os.path.join(base_dir, "templates/House_prediction.html"), "r") as file:
    html_content = file.read()

# Display HTML content
components.html(html_content, height=600)

# Form inputs
lot_area = st.number_input('Lot Area', min_value=0)
total_bsmt_sf = st.number_input('Total Basement SF', min_value=0)
bedroom_abv_gr = st.number_input('Bedrooms Above Grade', min_value=0)
tot_rms_abv_grd = st.number_input('Total Rooms Above Grade', min_value=0)
total_bath = st.number_input('Total Bathrooms', min_value=0)

if st.button('Predict'):
    try:
        # Collect input data
        input_data = [
            lot_area,
            total_bsmt_sf,
            bedroom_abv_gr,
            tot_rms_abv_grd,
            total_bath
        ]

        # Convert to numpy array and reshape
        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

        # Scale input data
        scaled_input_data = scaler.transform(input_data_as_numpy_array)

        # Make prediction
        prediction = model.predict(scaled_input_data)[0]

        # Display result
        st.success(f'Predicted Sale Price: ₹{prediction:,.2f}')
    except Exception as e:
        st.error(f'Error: {e}')
