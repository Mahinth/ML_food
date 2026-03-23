import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_demand_vs_time(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df, x='timestamp', y='food_demand', ax=ax)
    ax.set_title("Food Demand vs Time")
    plt.xticks(rotation=45)
    return fig

def plot_actual_vs_predicted(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Demand')
    ax.set_ylabel('Predicted Demand')
    ax.set_title('Actual vs Predicted Demand')
    return fig

def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    return fig

def plot_correlation_heatmap(df, feature_cols):
    fig, ax = plt.subplots(figsize=(12, 10))
    # Select only numeric for correlation
    corr = df[feature_cols + ['food_demand']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    return fig

def plot_model_comparison(metrics_df, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_df.plot(kind='bar', ax=ax)
    ax.set_title(title)
    ax.set_ylabel('Score')
    plt.xticks(rotation=0)
    return fig
