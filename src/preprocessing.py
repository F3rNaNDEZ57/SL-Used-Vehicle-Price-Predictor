import pandas as pd
import numpy as np
import re
from pathlib import Path

def extract_price(price_str):
    """Convert 'Rs: 12,500,000' to 12500000"""
    if pd.isna(price_str):
        return np.nan
    cleaned = re.sub(r'[^\d]', '', str(price_str))
    return float(cleaned) if cleaned else np.nan

def extract_metadata(metadata_str):
    """Extract date, location, AND mileage from metadata"""
    if pd.isna(metadata_str):
        return pd.Series([np.nan, np.nan, np.nan])
    
    parts = str(metadata_str).split('|')
    date = parts[0].strip() if len(parts) > 0 else np.nan
    
    location_part = parts[1].strip() if len(parts) > 1 else ""
    
    # Extract mileage (number at the end)
    mileage_match = re.search(r'(\d{1,3}(?:,\d{3})*)\s*$', location_part)
    mileage = float(mileage_match.group(1).replace(',', '')) if mileage_match else np.nan
    
    # Extract location (text before the number)
    location = re.sub(r'\d{1,3}(?:,\d{3})*\s*$', '', location_part).strip()
    
    return pd.Series([date, location, mileage])

def clean_vehicle_type(vehicle_type):
    """Extract body type from 'Car(Sedan)', 'Car()'"""
    if pd.isna(vehicle_type):
        return 'Unknown'
    match = re.search(r'\((.*?)\)', str(vehicle_type))
    return match.group(1) if match and match.group(1) else 'Unknown'

def remove_outliers(df):
    """Remove impossible values"""
    print(f"Before outlier removal: {df.shape}")
    
    # Price outliers
    df = df[(df['price_lkr'] > 100000) & (df['price_lkr'] < 50000000)]
    
    # Age outliers
    df = df[(df['vehicle_age'] >= 0) & (df['vehicle_age'] <= 50)]
    
    # Mileage outliers (>500k km is suspicious for private cars)
    df = df[(df['mileage_km'].isna()) | (df['mileage_km'] <= 500000)]
    
    print(f"After outlier removal: {df.shape}")
    return df

def preprocess_data(input_path, output_path):
    print("Loading raw data...")
    df = pd.read_csv(input_path)
    
    print(f"Original shape: {df.shape}")
    
    # Remove exact duplicates
    df = df.drop_duplicates()
    print(f"After dedup: {df.shape}")
    
    # Clean Price
    print("Cleaning prices...")
    df['price_lkr'] = df['Raw_Price'].apply(extract_price)
    
    # Extract metadata (includes mileage now)
    print("Extracting location, dates, and mileage...")
    df[['listing_date', 'location', 'mileage_km']] = df['Raw_Metadata'].apply(extract_metadata)
    
    # Clean Vehicle Type
    df['body_type'] = df['Vehicle Type'].apply(clean_vehicle_type)
    
    # Handle Registration Year
    df['is_registered'] = df['Register Year'] != 'Ref:'
    df['registration_year'] = pd.to_numeric(df['Register Year'], errors='coerce')
    
    # Feature Engineering
    current_year = 2024
    df['vehicle_age'] = current_year - df['Model Year']
    
    # Luxury brand flag
    luxury_brands = ['BMW', 'Mercedes Benz', 'Audi', 'Tesla']
    df['is_luxury'] = df['Manufacturer'].isin(luxury_brands).astype(int)
    
    # Engine CC
    df['engine_cc'] = pd.to_numeric(df['Engine/Motor Capacity'], errors='coerce')
    
    # Power categories for tax brackets
    df['power_category'] = pd.cut(df['engine_cc'], 
                                  bins=[0, 800, 1000, 1500, 2000, 9999],
                                  labels=['<800cc', '800-1000cc', '1000-1500cc', '1500-2000cc', '>2000cc'])
    
    # Remove outliers
    df = remove_outliers(df)
    
    # Handle missing mileage - IMPUTATION
    print(f"Missing mileage before imputation: {df['mileage_km'].isna().sum()}")
    
    # Impute missing mileage based on age group median
    df['mileage_km'] = df.groupby('vehicle_age')['mileage_km'].transform(
        lambda x: x.fillna(x.median())
    )
    
    # If still missing (new age groups), use global median
    global_median = df['mileage_km'].median()
    df['mileage_km'] = df['mileage_km'].fillna(global_median)
    
    print(f"Missing mileage after imputation: {df['mileage_km'].isna().sum()}")
    
    # Create km_per_year feature
    df['km_per_year'] = df['mileage_km'] / (df['vehicle_age'] + 1)
    
    # Select final columns (REMOVED 'Power' - duplicates engine_cc)
    model_cols = [
        'Manufacturer', 'Model', 'Model Year', 'vehicle_age',
        'mileage_km', 'km_per_year',  # NEW FEATURE ADDED
        'Fuel Type', 'Transmission', 'Condition',
        'body_type', 'engine_cc', 'is_luxury', 'is_registered',
        'Colour', 'location', 'price_lkr'
    ]
    
    df_clean = df[model_cols].copy()
    
    # Drop rows with missing target
    df_clean = df_clean.dropna(subset=['price_lkr'])
    
    print(f"Final shape: {df_clean.shape}")
    print(f"\nFeature correlations with price:")
    corr_data = df_clean[['vehicle_age', 'mileage_km', 'km_per_year', 'engine_cc', 'price_lkr']].corr()['price_lkr'].sort_values()
    print(corr_data)
    
    # Save
    df_clean.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    return df_clean

if __name__ == "__main__":
    input_file = Path("data/raw/vehicles_raw.csv")
    output_file = Path("data/processed/vehicles_clean.csv")
    
    df = preprocess_data(input_file, output_file)