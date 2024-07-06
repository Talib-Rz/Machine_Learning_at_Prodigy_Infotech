import streamlit as st
import joblib
import numpy as np

# Load the scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

st.title("House Prediction App")

# Read the HTML file content
with open("templates/House_prediction.html", "r") as file:
    html_content = file.read()

# Display HTML content
st.markdown(html_content, unsafe_allow_html=True)

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
