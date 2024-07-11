import streamlit as st
import joblib
import numpy as np

# Load the scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')

# Function to load HTML file
def load_html_file(file_path):
    with open(file_path, 'r') as f:
        html_content = f.read()
    return html_content

def predict_price(LotArea, TotalBsmtSF, BedroomAbvGr, TotRmsAbvGrd, TotalBath):
    try:
        # Prepare input data
        input_data = np.array([LotArea, TotalBsmtSF, BedroomAbvGr, TotRmsAbvGrd, TotalBath]).reshape(1, -1)

        # Scale input data
        scaled_input_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_input_data)[0]

        return prediction
    except Exception as e:
        return f'Error: {e}'

def main():
    st.title('House Price Prediction')
    html_content = load_html_file('House_prediction.html')
    st.write('Enter the details below to predict the house sale price:')

    # Input fields
    LotArea = st.number_input('Lot Area')
    TotalBsmtSF = st.number_input('Total Basement Area')
    BedroomAbvGr = st.number_input('Bedrooms Above Grade')
    TotRmsAbvGrd = st.number_input('Total Rooms Above Grade')
    TotalBath = st.number_input('Total Bathrooms')

    if st.button('Predict'):
        prediction = predict_price(LotArea, TotalBsmtSF, BedroomAbvGr, TotRmsAbvGrd, TotalBath)
        if (prediction < 0) :
          st.write('Enter valid details')
        else:
          st.write(f'Predicted Sale Price: ${prediction:,.2f}')
        
if __name__ == '__main__':
    main()
