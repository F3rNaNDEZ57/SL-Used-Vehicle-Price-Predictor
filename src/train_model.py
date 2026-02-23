import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from pathlib import Path

def train_model():
    # Load data
    df = pd.read_csv('data/processed/vehicles_clean.csv')
    
    # Features and target (is_registered removed)
    X = df.drop('price_lkr', axis=1)
    y = df['price_lkr']
    
    # DEBUG: Check features
    print("Features used:", X.columns.tolist())
    
    # Log transform target (prices are skewed)
    y_log = np.log1p(y)
    
    # Categorical features for CatBoost (removed is_registered)
    categorical_features = [
        'Manufacturer', 'Model', 'Fuel Type', 'Transmission', 
        'Condition', 'body_type', 'power_category', 'Colour', 'location'
    ]
    
    # Only keep columns that exist in X
    cat_features = [col for col in categorical_features if col in X.columns]
    
    print(f"\nCategorical features: {cat_features}")
    print(f"Numerical features: {[c for c in X.columns if c not in cat_features]}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    
    # Initialize CatBoost with better parameters
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        cat_features=cat_features,
        verbose=100,
        random_seed=42,
        early_stopping_rounds=50,
        bagging_temperature=0.5,
        random_strength=2
    )
    
    print("\nTraining CatBoost model...")
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    # Predictions (convert back from log scale)
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n=== MODEL PERFORMANCE ===")
    print(f"RMSE: Rs {rmse:,.0f}")
    print(f"MAE: Rs {mae:,.0f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²: {r2:.3f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': model.feature_names_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n=== TOP 10 FEATURES ===")
    print(importance.head(10))
    
    # Save model
    model_path = Path('models/catboost_vehicle_price.pkl')
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save test data for SHAP analysis
    test_data = X_test.copy()
    test_data['actual_price'] = y_true
    test_data['predicted_price'] = y_pred
    test_data.to_csv('data/processed/test_predictions.csv', index=False)
    
    # Calculate residuals for error analysis
    test_data['error'] = test_data['actual_price'] - test_data['predicted_price']
    test_data['abs_error'] = np.abs(test_data['error'])
    print(f"\nWorst predictions (top 5 overpriced):")
    print(test_data.nlargest(5, 'error')[['Manufacturer', 'Model', 'actual_price', 'predicted_price', 'error']])
    
    return model, X_test, y_test

if __name__ == "__main__":
    train_model()