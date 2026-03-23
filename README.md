# FOOD DEMAND PREDICTION USING MACHINE LEARNING

## Project Abstract
This project focuses on predicting food demand for restaurants and food delivery services using historical data and feature engineering. It implements both a regression model to estimate the exact quantity of food demand and a classification model to categorize demand into 'Low', 'Medium', or 'High'. The application is built using Streamlit, providing an interactive end-to-end ML pipeline.

## Key Objectives
1.  **Exploratory Data Analysis**: Visualize demand trends over time and understand feature correlations.
2.  **Automated Feature Engineering**: Implement time-based (Hour, Weekend) and statistical features (Lag, Rolling Mean/Std) automatically.
3.  **Predictive Modeling**: Train classical ML models like OLS Regression and Decision Trees for both forecasting and classification.
4.  **Performance Evaluation**: Compare models using standard metrics like RMSE, R² for regression and Accuracy, F1-Score for classification.
5.  **Interactive Deployment**: Create a user-friendly dashboard for data processing, training, and real-time prediction.

## Future Scope
- **Real-time Integration**: Fetching external data such as live weather or traffic conditions.
- **Deep Learning Models**: Implementing LSTM or GRU for more complex time-series patterns.
- **Explainable AI (XAI)**: Using SHAP or LIME to explain model predictions for better business decisions.
- **Optimization Engine**: Recommending inventory stock levels based on predicted demand.

## Viva Questions & Answers
1.  **Q: Why use lag features in food demand prediction?**
    *   **A**: Lag features capture the dependency of current demand on past behavior (e.g., yesterday's demand), which is crucial for time-series data.
2.  **Q: What is the difference between classification and regression in this project?**
    *   **A**: Regression predicts a continuous numerical value (e.g., 145 units), while classification predicts a discrete category (e.g., High Demand).
3.  **Q: Why do we use Rolling Mean and Rolling Std?**
    *   **A**: Rolling Mean smooths out fluctuations to show trends, while Rolling Std provides a measure of demand volatility over a specific window.
4.  **Q: Why was no shuffling used in the train-test split?**
    *   **A**: In time-series data, the order of events matters. Shuffling would leak future information into the training set, leading to unrealistic results.
5.  **Q: How are 'Low', 'Medium', and 'High' categories determined?**
    *   **A**: They are calculated using statistical quantiles (0.33 and 0.66) of the existing demand data, ensuring categories are relative to the business's specific data range.

## Instructions to Run the App
1.  **Install Python**: Ensure Python 3.8+ is installed.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit App**:
    ```bash
    streamlit run app.py
    ```
4.  **Use the Dashboard**: 
    - Upload a CSV containing `timestamp` and `food_demand`.
    - Navigate through the steps in the sidebar: Validation -> Feature Engineering -> Training -> Results.
