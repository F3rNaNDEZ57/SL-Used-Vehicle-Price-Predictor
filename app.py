import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Sri Lanka Vehicle Price Predictor", layout="wide")

st.title("🚗 Sri Lanka Vehicle Price Estimator")
st.markdown("Predict market prices for used vehicles using Machine Learning")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('models/catboost_vehicle_price.pkl')

model = load_model()

# Load data for dropdown options
df = pd.read_csv('data/processed/vehicles_clean.csv')

# Sidebar inputs
st.sidebar.header("Vehicle Details")

manufacturer = st.sidebar.selectbox("Manufacturer", df['Manufacturer'].unique())
model_name = st.sidebar.selectbox("Model", df[df['Manufacturer']==manufacturer]['Model'].unique())
year = st.sidebar.slider("Model Year", int(df['Model Year'].min()), 2024, 2015)
mileage = st.sidebar.number_input("Mileage (km)", 0, 500000, 50000)
fuel = st.sidebar.selectbox("Fuel Type", df['Fuel Type'].unique())
transmission = st.sidebar.selectbox("Transmission", df['Transmission'].unique())
condition = st.sidebar.selectbox("Condition", df['Condition'].unique())
body = st.sidebar.selectbox("Body Type", df['body_type'].unique())
engine = st.sidebar.number_input("Engine CC", 600, 5000, 1300)
color = st.sidebar.selectbox("Colour", df['Colour'].unique())
location = st.sidebar.selectbox("Location", df['location'].unique())

# Create input dataframe
input_data = pd.DataFrame({
    'Manufacturer': [manufacturer],
    'Model': [model_name],
    'Model Year': [year],
    'vehicle_age': [2024 - year],
    'mileage_km': [mileage],
    'Fuel Type': [fuel],
    'Transmission': [transmission],
    'Condition': [condition],
    'body_type': [body],
    'engine_cc': [engine],
    'is_luxury': [1 if manufacturer in ['BMW', 'Mercedes Benz', 'Audi'] else 0],
    'is_registered': [1],
    'Colour': [color],
    'location': [location],
    'Power': [engine]  # Approximation
})

# Predict
if st.sidebar.button("Estimate Price"):
    pred_log = model.predict(input_data)
    pred_price = np.expm1(pred_log)[0]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Estimated Price", f"Rs {pred_price:,.0f}")
    
    # Price range (±10%)
    col2.metric("Fair Range", f"Rs {pred_price*0.9:,.0f} - Rs {pred_price*1.1:,.0f}")
    
    # Compare to market average
    market_avg = df[(df['Manufacturer']==manufacturer) & (df['Model']==model_name)]['price_lkr'].mean()
    if not pd.isna(market_avg):
        diff = ((pred_price - market_avg) / market_avg) * 100
        col3.metric("vs Market Avg", f"{diff:+.0f}%")
    
    st.subheader("Input Summary")
    st.write(input_data)

st.markdown("---")
st.info("This model uses CatBoost algorithm trained on PatPat.lk listings. SHAP analysis shows Model Year, Mileage, and Brand are top price drivers.")