import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

def explain_model():
    print("Loading model and data...")
    model = joblib.load('models/catboost_vehicle_price.pkl')
    df = pd.read_csv('data/processed/vehicles_clean.csv')
    
    # Sample 1000 rows for SHAP (speed)
    X_sample = df.drop('price_lkr', axis=1).sample(n=min(1000, len(df)), random_state=42)
    
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # 1. Summary plot (global importance)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig('models/shap_summary.png', dpi=150, bbox_inches='tight')
    print("Saved: models/shap_summary.png")
    
    # 2. Bar plot (feature importance)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig('models/shap_importance.png', dpi=150, bbox_inches='tight')
    print("Saved: models/shap_importance.png")
    
    # 3. Individual explanation example (force plot for first instance)
    # Convert to HTML for report
    shap.force_plot(explainer.expected_value, shap_values[0], X_sample.iloc[0], 
                    matplotlib=True, show=False)
    plt.savefig('models/shap_force_plot.png', dpi=150, bbox_inches='tight')
    print("Saved: models/shap_force_plot.png")
    
    # 4. Dependence plot (Price vs Mileage)
    if 'mileage_km' in X_sample.columns:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot('mileage_km', shap_values, X_sample, show=False)
        plt.tight_layout()
        plt.savefig('models/shap_mileage_dependence.png', dpi=150, bbox_inches='tight')
        print("Saved: models/shap_mileage_dependence.png")
    
    print("\nSHAP analysis complete! Check the 'models' folder for plots.")

if __name__ == "__main__":
    explain_model()