from flask import Flask, render_template, request
import streamlit as st
import joblib
import numpy as np

House_prediction_app = Flask(__name__)

# Load the scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')


@House_prediction_app.route('/')
def home():
    return render_template('House_prediction.html')


@House_prediction_app.route('/predict', methods=['POST'])
def predict_price():
    try:
        # Retrieve form data
        input_data = [
            float(request.form['LotArea']),
            float(request.form['TotalBsmtSF']),
            float(request.form['BedroomAbvGr']),
            float(request.form['TotRmsAbvGrd']),
            float(request.form['TotalBath'])
        ]

        # Convert to numpy array and reshape
        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

        # Scale input data
        scaled_input_data = scaler.transform(input_data_as_numpy_array)

        # Make prediction
        prediction = model.predict(scaled_input_data)[0]

        # Format result
        result = f'Predicted Sale Price: ₹{prediction:,.2f}'
    except Exception as e:
        result = f'Error: {e}'

    return render_template('House_prediction.html', result=result)


if __name__ == '__main__':
    House_prediction_app.run(debug=True)
