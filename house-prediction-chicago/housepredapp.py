import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("model.pkl")

st.title("🏠 House Price Predictor")

st.write("Enter house details:")

# Numeric inputs
longitude = st.number_input("Longitude", value=-122.23)
latitude = st.number_input("Latitude", value=37.88)
housing_median_age = st.slider("Housing Median Age", 1, 100, 20)
total_rooms = st.number_input("Total Rooms", value=1000)
total_bedrooms = st.number_input("Total Bedrooms", value=200)
population = st.number_input("Population", value=500)
households = st.number_input("Households", value=200)
median_income = st.number_input("Median Income", value=5.0)

# Categorical input
ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)

# Create input dataframe
input_df = pd.DataFrame({
    "longitude": [longitude],
    "latitude": [latitude],
    "housing_median_age": [housing_median_age],
    "total_rooms": [total_rooms],
    "total_bedrooms": [total_bedrooms],
    "population": [population],
    "households": [households],
    "median_income": [median_income],
    "ocean_proximity": [ocean_proximity]
})

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated House Price: ${prediction:,.2f}")