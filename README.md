# Food Demand Prediction Using Machine Learning

A comprehensive ML application for predicting food demand for restaurants and food delivery services using historical data and feature engineering.

## Project Overview

This project implements both a **regression model** to estimate the exact quantity of food demand and a **classification model** to categorize demand into 'Low', 'Medium', or 'High'. Built with Streamlit for an interactive end-to-end ML pipeline.

## Features

- **Exploratory Data Analysis**: Visualize demand trends over time and understand feature correlations
- **Automated Feature Engineering**: Time-based (Hour, Weekend) and statistical features (Lag, Rolling Mean/Std)
- **Predictive Modeling**: OLS, Ridge, and Lasso Regression for forecasting
- **Performance Evaluation**: Compare models using R², RMSE, MAE metrics
- **Interactive Dashboard**: User-friendly interface for data processing, training, and prediction

## Model Performance

| Model | R² Score | Accuracy |
|-------|----------|----------|
| OLS Linear Regression | 96.40% | ✓ |
| Ridge Regression | 97.11% | ✓ |
| **Lasso Regression** | **97.84%** | **Best** |
| Keras Neural Network | 95.47% | ✓ |

## Project Structure

```
ML_Project_Food/
├── app.py                      # Streamlit dashboard
├── train_model.py              # Model training script
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── COMMANDS.md                 # Run commands documentation
├── src/
│   ├── dataset/
│   │   ├── food_demand_dataset.xlsx       # Original dataset
│   │   └── clean_synthetic_food_demand.csv  # Clean synthetic dataset
│   ├── models/
│   │   ├── food_demand_model.h5          # Keras model (95.47%)
│   │   ├── food_demand_model_sklearn.pkl  # Scikit-learn model (97.84%)
│   │   ├── scaler.pkl                    # Feature scaler
│   │   ├── polynomial_features.pkl        # Polynomial transformer
│   │   ├── label_encoder_*.pkl           # Label encoders
│   │   └── model_info.json               # Model metadata
│   ├── preprocessing.py        # Data validation & preprocessing
│   ├── feature_engineering.py # Feature creation functions
│   ├── models.py             # ML model training & evaluation
│   └── visualization.py      # Plotting functions
└── .venv/                    # Virtual environment
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ML_Project_Food

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Train Models
```bash
python train_model.py
```

### 2. Run Streamlit Dashboard
```bash
streamlit run app.py
```

## Usage Guide

The Streamlit app provides a 6-step wizard:

1. **Homepage**: Overview of the application
2. **Step 1: Get Your Data**: Upload CSV or use example data
3. **Step 2: Check Your Data**: Validate and visualize data
4. **Step 3: Prepare Your Data**: Feature engineering
5. **Step 4: Teach the Computer**: Train regression & classification models
6. **Step 5: See Results**: View model performance metrics
7. **Step 6: Make Predictions**: Enter details for demand forecasting

## Key Features

### Target Variable
- **food_demand**: Continuous value representing number of items/plates needed

### Input Features
- **Categorical**: Item_Category, Cuisine_Type, Weather_Type
- **Numerical**: Preparation_Time, Shelf_Life, Price, Discount_Percentage, Population_Density
- **Historical**: Past_Average_Demand, Demand_Last_Week
- **Binary**: Is_Holiday, Special_Event
- **Engineered**: Lag features, Rolling statistics, Interaction terms

## Model Details

### Regression Models
- **OLS Linear Regression**: Ordinary least squares baseline
- **Ridge Regression**: L2 regularization for multicollinearity
- **Lasso Regression**: L1 regularization for feature selection

### Classification Models
- **Decision Tree**: Interpretable rule-based classification
- **SVM**: Support Vector Machine with probability estimates
- **Naive Bayes**: Probabilistic classification

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| R² Score | Coefficient of determination (target: >95%) |
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| Accuracy | Classification accuracy |
| F1 Score | Harmonic mean of precision and recall |

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn
- tensorflow, keras
- streamlit
- matplotlib, seaborn

## Team

Developed as a machine learning project demonstration.

## License

MIT License
