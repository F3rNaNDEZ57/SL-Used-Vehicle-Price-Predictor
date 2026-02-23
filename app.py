import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Sri Lanka Vehicle Price Predictor", layout="wide")

# Load model and data (cached)
@st.cache_resource
def load_model():
    return joblib.load('models/catboost_vehicle_price.pkl')

@st.cache_data
def load_data():
    return pd.read_csv('data/processed/vehicles_clean.csv')

model = load_model()
df = load_data()

# Initialize SHAP explainer (cached)
@st.cache_resource
def get_explainer():
    return shap.TreeExplainer(model)

explainer = get_explainer()

st.title("🚗 Sri Lanka Vehicle Price Estimator")
st.markdown("Predict market prices for used vehicles using Machine Learning (CatBoost)")

# Navigation
page = st.sidebar.radio("📍 Navigation", ["Price Predictor", "Model Analytics & SHAP"])

if page == "Price Predictor":
    st.header("Vehicle Price Prediction with Explanation")
    
    # Sidebar inputs
    st.sidebar.header("Vehicle Details")
    
    manufacturer = st.sidebar.selectbox("Manufacturer", sorted(df['Manufacturer'].unique()))
    model_name = st.sidebar.selectbox("Model", sorted(df[df['Manufacturer']==manufacturer]['Model'].unique()))
    year = st.sidebar.slider("Model Year", int(df['Model Year'].min()), 2024, 2015)
    mileage = st.sidebar.number_input("Mileage (km)", 0, 500000, 50000, help="Total kilometers driven")
    fuel = st.sidebar.selectbox("Fuel Type", sorted(df['Fuel Type'].unique()))
    transmission = st.sidebar.selectbox("Transmission", sorted(df['Transmission'].unique()))
    condition = st.sidebar.selectbox("Condition", sorted(df['Condition'].unique()))
    body = st.sidebar.selectbox("Body Type", sorted(df['body_type'].unique()))
    engine = st.sidebar.number_input("Engine CC", 600, 5000, 1300)
    color = st.sidebar.selectbox("Colour", sorted(df['Colour'].unique()))
    location = st.sidebar.selectbox("Location", sorted(df['location'].unique()))
    
    # Calculate derived features
    vehicle_age = 2024 - year
    km_per_year = mileage / (vehicle_age + 1)
    is_luxury = 1 if manufacturer in ['BMW', 'Mercedes Benz', 'Audi'] else 0
    is_registered = 1
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Manufacturer': [manufacturer],
        'Model': [model_name],
        'Model Year': [year],
        'vehicle_age': [vehicle_age],
        'mileage_km': [mileage],
        'km_per_year': [km_per_year],
        'Fuel Type': [fuel],
        'Transmission': [transmission],
        'Condition': [condition],
        'body_type': [body],
        'engine_cc': [engine],
        'is_luxury': [is_luxury],
        'is_registered': [is_registered],
        'Colour': [color],
        'location': [location]
    })
    
    # Predict button
    if st.sidebar.button("🔮 Estimate Price", type="primary"):
        with st.spinner("Calculating prediction and explanations..."):
            # Get prediction (model outputs log scale)
            pred_log = model.predict(input_data)[0]
            pred_price = np.expm1(pred_log)
            
            # Calculate SHAP values for this specific prediction
            shap_values = explainer.shap_values(input_data)
            
            # Handle both old and new shap versions
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            # Create explanation dataframe
            feature_names = input_data.columns.tolist()
            shap_contributions = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            
            explanation_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': input_data.iloc[0].values,
                'SHAP_Value': shap_contributions,
                'Abs_Impact': np.abs(shap_contributions)
            }).sort_values('Abs_Impact', ascending=False)
            
            # Convert log contributions to approximate LKR impact
            # Approximate: exp(log_pred + shap) - exp(log_pred) ≈ pred * shap (for small shap)
            explanation_df['Price_Impact_LKR'] = pred_price * explanation_df['SHAP_Value']
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Estimated Price", f"Rs {pred_price:,.0f}")
            
            # Price range (±10%)
            col2.metric("Fair Range", f"Rs {pred_price*0.9:,.0f} - Rs {pred_price*1.1:,.0f}")
            
            # Compare to market average
            market_avg = df[(df['Manufacturer']==manufacturer) & (df['Model']==model_name)]['price_lkr'].mean()
            if not pd.isna(market_avg) and market_avg > 0:
                diff = ((pred_price - market_avg) / market_avg) * 100
                col3.metric("vs Market Avg", f"{diff:+.0f}%", delta=f"{diff:+.1f}%")
            
            st.markdown("---")
            
            # LOCAL EXPLANATION SECTION
            st.subheader("🔍 How was this price calculated?")
            st.markdown("This breakdown shows which features pushed the price **up** (positive) or **down** (negative) for this specific vehicle.")
            
            # Top influencers
            col_up, col_down = st.columns(2)
            
            with col_up:
                st.markdown("**📈 Price Increasers**")
                positive = explanation_df[explanation_df['SHAP_Value'] > 0].head(5)
                for _, row in positive.iterrows():
                    impact_lkr = row['Price_Impact_LKR']
                    st.markdown(f"""
                    <div style='background-color: #000000; padding: 8px; border-radius: 5px; margin: 4px 0; border-left: 4px solid #34a853;'>
                        <b>{row['Feature']}</b>: {row['Value']}<br>
                        <span style='color: #137333; font-size: 0.9em;'>+Rs {impact_lkr:,.0f}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_down:
                st.markdown("**📉 Price Decreasers**")
                negative = explanation_df[explanation_df['SHAP_Value'] < 0].head(5)
                for _, row in negative.iterrows():
                    impact_lkr = abs(row['Price_Impact_LKR'])
                    st.markdown(f"""
                    <div style='background-color: #000000; padding: 8px; border-radius: 5px; margin: 4px 0; border-left: 4px solid #ea4335;'>
                        <b>{row['Feature']}</b>: {row['Value']}<br>
                        <span style='color: #c5221f; font-size: 0.9em;'>-Rs {impact_lkr:,.0f}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Visualization of contributions
            st.markdown("**Feature Impact Visualization**")
            top_features = explanation_df.head(10).copy()
            top_features['Color'] = top_features['SHAP_Value'].apply(lambda x: 'Increased Price' if x > 0 else 'Decreased Price')
            
            fig = px.bar(
                top_features,
                x='Price_Impact_LKR',
                y='Feature',
                orientation='h',
                color='Color',
                color_discrete_map={'Increased Price': '#34a853', 'Decreased Price': '#ea4335'},
                title=f"Top 10 Price Drivers for this {manufacturer} {model_name}",
                labels={'Price_Impact_LKR': 'Impact on Price (Rs)', 'Feature': ''},
                height=400
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Base value info
            st.info(f"""
            **Base Value (Average):** Rs {np.expm1(explainer.expected_value):,.0f}  
            **Final Prediction:** Rs {pred_price:,.0f}  
            **Difference:** Rs {pred_price - np.expm1(explainer.expected_value):,.0f}
            """)
            
            # Show input summary
            with st.expander("View Input Details"):
                st.write(input_data.T.rename(columns={0: 'Value'}))

else:  # Model Analytics Page
    st.header("📊 Model Analytics & Global SHAP Explanations")
    st.markdown("Understanding how the model makes decisions across all vehicles in the dataset.")
    
    # Load sample for global analysis (faster than full dataset)
    @st.cache_data
    def get_sample_shap(n=500):
        df_sample = df.sample(n=n, random_state=42)
        X_sample = df_sample.drop('price_lkr', axis=1)
        shap_vals = explainer.shap_values(X_sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        return shap_vals, X_sample
    
    with st.spinner("Loading SHAP analysis..."):
        shap_vals, X_sample = get_sample_shap(500)
        
        # Feature Importance (Mean Absolute SHAP)
        st.subheader("🏆 Global Feature Importance")
        st.markdown("Features ranked by their average impact on price predictions across all cars.")
        
        importance_df = pd.DataFrame({
            'Feature': X_sample.columns,
            'Importance': np.abs(shap_vals).mean(0)
        }).sort_values('Importance', ascending=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                importance_df.head(15),
                x='Importance',
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale='Viridis',
                height=500
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Top 10 Drivers:**")
            for i, row in importance_df.head(10).iterrows():
                st.markdown(f"{i+1}. **{row['Feature']}** ({row['Importance']:.3f})")
        
        st.markdown("---")
        
        # SHAP Summary Plot
        st.subheader("📈 SHAP Summary Plot (Beeswarm)")
        st.markdown("Each dot represents a car. Color indicates feature value (red=high, blue=low). Position shows impact on price.")
        
        # Check if pre-generated plot exists
        if Path('models/shap_summary.png').exists():
            st.image('models/shap_summary.png', use_container_width=True)
        else:
            st.info("Generating summary plot... (this may take a moment)")
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_vals, X_sample, show=False, max_display=15)
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Dependence Plots
        st.subheader("🔗 Feature Dependence Analysis")
        st.markdown("How specific features interact with the predicted price.")
        
        feature_to_plot = st.selectbox(
            "Select feature to analyze:",
            options=['mileage_km', 'vehicle_age', 'engine_cc', 'km_per_year', 'Model Year'],
            index=0
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{feature_to_plot} vs Price Impact**")
            
            # Check if pre-generated plot exists for mileage
            if feature_to_plot == 'mileage_km' and Path('models/shap_mileage_dependence.png').exists():
                st.image('models/shap_mileage_dependence.png', use_container_width=True)
            else:
                # Generate on the fly
                fig, ax = plt.subplots(figsize=(8, 6))
                shap.dependence_plot(feature_to_plot, shap_vals, X_sample, show=False, ax=ax)
                plt.tight_layout()
                st.pyplot(fig)
        
        with col2:
            st.markdown("**Distribution of Feature Values**")
            fig2 = px.histogram(df, x=feature_to_plot, nbins=50, height=400)
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")
        
        # Model Performance Stats
        st.subheader("📉 Model Performance Statistics")
        
        if Path('data/processed/test_predictions.csv').exists():
            test_df = pd.read_csv('data/processed/test_predictions.csv')
            test_df['error'] = test_df['actual_price'] - test_df['predicted_price']
            test_df['abs_error'] = np.abs(test_df['error'])
            test_df['pct_error'] = (test_df['abs_error'] / test_df['actual_price']) * 100
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAPE", f"{test_df['pct_error'].mean():.2f}%")
            col2.metric("Avg Error", f"Rs {test_df['abs_error'].mean():,.0f}")
            col3.metric("Max Error", f"Rs {test_df['abs_error'].max():,.0f}")
            col4.metric("R² Score", f"{1 - (test_df['error']**2).sum()/((test_df['actual_price'] - test_df['actual_price'].mean())**2).sum():.3f}")
            
            # Error distribution
            fig_err = px.histogram(test_df, x='pct_error', nbins=30, 
                                  title="Prediction Error Distribution (%)",
                                  labels={'pct_error': 'Percentage Error (%)'})
            st.plotly_chart(fig_err, use_container_width=True)
        else:
            st.warning("Test predictions file not found. Run training to see performance metrics.")

st.sidebar.markdown("---")
st.sidebar.info("""
**About this model:**
- Algorithm: CatBoost (Gradient Boosting)
- Data: PatPat.lk listings (Sri Lanka)
- Features: 15 (including engineered features like km/year)
- MAPE: ~11.4%
""")