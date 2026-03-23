import pandas as pd
import numpy as np

def create_features(df):
    """
    Automatically generates features: Hour, Is_Weekend, Lags, and Rolling statistics.
    """
    df = df.copy()
    
    # Time-based features
    df['Hour'] = df['timestamp'].dt.hour
    df['Is_Weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Lag features
    df['Lag_1'] = df['food_demand'].shift(1)
    df['Lag_7'] = df['food_demand'].shift(7)
    df['Lag_14'] = df['food_demand'].shift(14)
    
    # Rolling features
    df['Rolling_Mean_7'] = df['food_demand'].shift(1).rolling(window=7).mean()
    df['Rolling_Mean_14'] = df['food_demand'].shift(1).rolling(window=14).mean()
    df['Rolling_Std_7'] = df['food_demand'].shift(1).rolling(window=7).std()
    
    # Past Average Demand & Demand Last Week (proxy calculations if not in original)
    if 'Past_Average_Demand' not in df.columns:
        df['Past_Average_Demand'] = df['food_demand'].expanding().mean().shift(1)
    if 'Demand_Last_Week' not in df.columns:
        df['Demand_Last_Week'] = df['food_demand'].shift(7)
        
    # Drop rows with NaN values created by shift/rolling
    df = df.dropna().reset_index(drop=True)
    
    return df

def create_classification_target(df):
    """
    Creates demand_category (Low, Medium, High) based on quantiles.
    """
    df = df.copy()
    quantiles = df['food_demand'].quantile([0.33, 0.66]).values
    
    def classify(x):
        if x <= quantiles[0]:
            return 'Low'
        elif x <= quantiles[1]:
            return 'Medium'
        else:
            return 'High'
            
    df['demand_category'] = df['food_demand'].apply(classify)
    return df
