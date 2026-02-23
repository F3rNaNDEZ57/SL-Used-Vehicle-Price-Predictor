# 🚗 Sri Lanka Vehicle Price Predictor

A machine learning-powered web application that predicts market prices
for used vehicles in Sri Lanka using CatBoost gradient boosting, with
full SHAP (SHapley Additive exPlanations) interpretability to explain
pricing factors.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

## 📋 Overview

This project scrapes vehicle listing data from
[patpat.lk](https://www.patpat.lk), cleans and engineers features,
trains a CatBoost regression model to predict vehicle prices in LKR (Sri
Lankan Rupees), and provides an interactive Streamlit dashboard with
explainable AI features.

## 🏗️ Project Structure

    sl_vehicle_price_predicter/
    ├── app.py
    ├── requirements.txt
    ├── data/
    │   ├── raw/
    │   │   └── vehicles_raw.csv
    │   └── processed/
    │       ├── vehicles_clean.csv
    │       └── test_predictions.csv
    ├── models/
    │   ├── catboost_vehicle_price.pkl
    │   ├── shap_summary.png
    │   ├── shap_importance.png
    │   ├── shap_force_plot.png
    │   └── shap_mileage_dependence.png
    └── src/
        ├── scrape.py
        ├── preprocessing.py
        ├── train_model.py
        └── explain.py

## 🚀 Quick Start

### Installation

``` bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### Run Full Pipeline

``` bash
python src/scrape.py
python src/preprocessing.py
python src/train_model.py
python src/explain.py
streamlit run app.py
```

## 📊 Model Performance

-   **MAPE**: \~11.4%
-   **RMSE**: \~Rs 1.2M
-   **R²**: \~0.89

## 🔍 Explainability

Uses SHAP to provide: - Local prediction explanations - Global feature
importance - Dependence plots

## 🛠️ Tech Stack

-   CatBoost
-   Streamlit
-   SHAP
-   Scikit-learn
-   Pandas / NumPy
-   BeautifulSoup

## 📝 License

Educational use only.

------------------------------------------------------------------------

Developed for University of Moratuwa - ML Assignment
