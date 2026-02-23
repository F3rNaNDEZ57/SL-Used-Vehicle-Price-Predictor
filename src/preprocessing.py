import pandas as pd
import numpy as np
import re
from pathlib import Path

def extract_price(price_str):
    """Convert 'Rs: 12,500,000' to 12500000"""
    if pd.isna(price_str):
        return np.nan
    # Remove 'Rs: ' and commas, keep only numbers
    cleaned = re.sub(r'[^\d]', '', str(price_str))
    return float(cleaned) if cleaned else np.nan

def extract_mileage(mileage_str):
    """Handle 'km Find', '82,285', empty strings"""
    if pd.isna(mileage_str) or 'Find' in str(mileage_str):
        return np.nan
    # Remove non-numeric except dots
    cleaned = re.sub(r'[^\d.]', '', str(mileage_str))
    try:
        return float(cleaned) if cleaned else np.nan
    except:
        return np.nan

def extract_metadata(metadata_str):
    """Extract date, location, AND mileage from metadata"""
    if pd.isna(metadata_str):
        return pd.Series([np.nan, np.nan, np.nan])
    
    parts = str(metadata_str).split('|')
    date = parts[0].strip() if len(parts) > 0 else np.nan
    
    # Location and mileage are mixed in part 2
    location_part = parts[1].strip() if len(parts) > 1 else ""
    
    # Extract mileage (number at the end after location)
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
    
    # Clean Mileage
    print("Cleaning mileage...")
    df['mileage_km'] = df['Raw_Mileage'].apply(extract_mileage)
    
    # Extract metadata
    print("Extracting location and dates...")
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
    
    # Power categories (Sri Lanka tax brackets)
    df['engine_cc'] = pd.to_numeric(df['Engine/Motor Capacity'], errors='coerce')
    df['power_category'] = pd.cut(df['Engine/Motor Capacity'], 
                                  bins=[0, 800, 1000, 1500, 2000, 9999],
                                  labels=['<800cc', '800-1000cc', '1000-1500cc', '1500-2000cc', '>2000cc'])
    
    # Remove outliers (price > 50M or < 100k likely errors)
    df = df[(df['price_lkr'] > 100000) & (df['price_lkr'] < 50000000)]
    
    # Remove impossible ages
    df = df[(df['vehicle_age'] >= 0) & (df['vehicle_age'] <= 50)]
    
    # Select final columns for modeling
    model_cols = [
        'Manufacturer', 'Model', 'Model Year', 'vehicle_age',
        'mileage_km', 'Fuel Type', 'Transmission', 'Condition',
        'body_type', 'engine_cc', 'is_luxury', 'is_registered',
        'Colour', 'location', 'Power', 'price_lkr'
    ]
    
    df_clean = df[model_cols].copy()
    
    # Drop rows with missing target
    df_clean = df_clean.dropna(subset=['price_lkr'])
    
    print(f"Final shape: {df_clean.shape}")
    print(f"Missing values:\n{df_clean.isnull().sum()}")
    
    # Save
    df_clean.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    return df_clean

if __name__ == "__main__":
    input_file = Path("data/raw/vehicles_raw.csv")
    output_file = Path("data/processed/vehicles_clean.csv")
    
    df = preprocess_data(input_file, output_file)