import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load trained model
model = joblib.load("xgb_model.pkl")

# Streamlit UI
st.title("🚗 CO₂ Emission Prediction App")

st.sidebar.header("Input Features")
vehicle_class = st.sidebar.selectbox('Vehicle Class',
    ['COMPACT', 'SUV - SMALL', 'MID-SIZE', 'TWO-SEATER', 'MINICOMPACT',
     'SUBCOMPACT', 'FULL-SIZE', 'STATION WAGON - SMALL',
     'SUV - STANDARD', 'VAN - CARGO', 'VAN - PASSENGER',
     'PICKUP TRUCK - STANDARD', 'MINIVAN', 'SPECIAL PURPOSE VEHICLE',
     'STATION WAGON - MID-SIZE', 'PICKUP TRUCK - SMALL']
)

engine_size = st.sidebar.slider("Engine Size (L)", 1.0, 10.0, step=0.1)
cylinders = st.sidebar.slider("Cylinders", 2, 16, step=1)
transmission = st.sidebar.selectbox("Transmission Type", ['AS', 'M', 'AV', 'AM', 'A'])
fuel_type = st.sidebar.selectbox("Fuel Type", ['Z', 'D', 'X', 'E', 'N'])

fuel_consumption_city = st.sidebar.slider("Fuel Consumption (City) (L/100 km)", 4, 30)
fuel_consumption_hwy = st.sidebar.slider("Fuel Consumption (Highway) (L/100 km)", 4, 30)
fuel_consumption_comb = st.sidebar.slider("Fuel Consumption (Combined) (L/100 km)", 4, 25)
fuel_consumption_comb_mpg = st.sidebar.slider("Fuel Consumption (Combined) (MPG)", 10, 70)

# Create DataFrame
input_data = pd.DataFrame({
    'vehicle_class': [vehicle_class],
    'engine_size': [engine_size],
    'cylinders': [cylinders],
    'transmission': [transmission],
    'fuel_type': [fuel_type],
    'fuel_consumption_city': [fuel_consumption_city],
    'fuel_consumption_hwy': [fuel_consumption_hwy],
    'fuel_consumption_comb(l/100km)': [fuel_consumption_comb],
    'fuel_consumption_comb(mpg)': [fuel_consumption_comb_mpg]
})

# 🔹 Encode Categorical Features
encoder = LabelEncoder()

# Load label encoder (if used in training)
input_data['vehicle_class'] = encoder.fit_transform(input_data['vehicle_class'])
input_data['transmission'] = encoder.fit_transform(input_data['transmission'])
input_data['fuel_type'] = encoder.fit_transform(input_data['fuel_type'])

# Predict CO2 emissions
if st.button("Predict CO₂ Emissions"):
    prediction = model.predict(input_data)
    st.success(f"🚗 Estimated CO₂ Emissions: **{prediction[0]:.2f} g/km**")

# CSV Batch Prediction
st.subheader("📂 Upload CSV for Bulk Predictions")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    data = pd.read_csv(uploaded_file)
    
    # Show the raw data
    st.write("Original Data:")
    st.write(data.head())
    
    # Ensure required columns exist
    required_columns = ['vehicle_class', 'engine_size', 'cylinders', 'transmission', 'fuel_type',
                       'fuel_consumption_city', 'fuel_consumption_hwy', 'fuel_consumption_comb(l/100km)',
                       'fuel_consumption_comb(mpg)']
        
    if not all(col in data.columns for col in required_columns):
        st.error("Uploaded CSV is missing required columns. Please check the format.")
    else:
        # Create a copy with only the required columns
        prediction_data = data[required_columns].copy()
        
        # Encode categorical features using the same encoding as the trained model
        for col in ['vehicle_class', 'transmission', 'fuel_type']:
            prediction_data[col] = encoder.fit_transform(prediction_data[col])

        # Predict CO₂ emissions
        try:
            predictions = model.predict(prediction_data)
            data['Predicted CO₂ Emissions (g/km)'] = predictions
            
            # Display results
            st.write("Prediction Results:")
            st.write(data)
            
            # Download button
            csv = data.to_csv(index=False)
            st.download_button("Download Predictions", csv, file_name="co2_predictions.csv", mime="text/csv")
        
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.write("Try using only the required columns in your CSV file or ensure all columns have the correct data types.")