from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd

def train_test_split_time_series(df, target_col, feature_cols, test_size=0.2):
    """
    Split data for time series (no shuffle).
    """
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    return X_train, X_test, y_train, y_test

def get_regression_model(model_name):
    models = {
        'OLS Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso()
    }
    return models.get(model_name)

def get_classification_model(model_name):
    models = {
        'Decision Tree Classifier': DecisionTreeClassifier(),
        'Support Vector Machine (SVM)': SVC(probability=True),
        'Naïve Bayes': GaussianNB()
    }
    return models.get(model_name)

def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2 Score": r2}

def evaluate_classification(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    # Using 'weighted' average for multi-class classification
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1, "Confusion Matrix": cm}
