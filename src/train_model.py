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
    
    # Features and target
    X = df.drop('price_lkr', axis=1)
    y = df['price_lkr']
    
    # Log transform target (prices are skewed)
    y_log = np.log1p(y)
    
    # Categorical features indices for CatBoost
    categorical_features = [
        'Manufacturer', 'Model', 'Fuel Type', 'Transmission', 
        'Condition', 'body_type', 'power_category', 'Colour', 'location'
    ]
    
    # Only keep columns that exist
    cat_features = [col for col in categorical_features if col in X.columns]
    
    print(f"Categorical features: {cat_features}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    
    # Initialize CatBoost
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        cat_features=cat_features,
        verbose=100,
        random_seed=42,
        early_stopping_rounds=50
    )
    
    print("Training CatBoost model...")
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
    
    return model, X_test, y_test

if __name__ == "__main__":
    train_model()