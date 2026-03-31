from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
model = joblib.load('src/models/food_demand_model_sklearn.pkl')
scaler = joblib.load('src/models/scaler.pkl')
poly = joblib.load('src/models/polynomial_features.pkl')

with open('src/models/model_info.json', 'r') as f:
    model_info = json.load(f)

class PredictionRequest(BaseModel):
    past_avg_demand: float
    demand_last_week: float
    price: float
    population_density: float
    discount: float
    prep_time: float
    shelf_life: float
    item_category: int
    cuisine_type: int
    weather: int
    is_holiday: int
    special_event: int

@app.get("/")
def root():
    return {"message": "Food Demand Prediction API", "model": "Lasso Regression", "accuracy": model_info["r2_score"]}

@app.get("/model-info")
def get_model_info():
    return model_info

@app.post("/predict")
def predict(request: PredictionRequest):
    lag_1 = request.past_avg_demand * 0.95
    lag_2 = request.past_avg_demand * 0.90
    lag_3 = request.past_avg_demand * 0.88
    lag_7 = request.demand_last_week
    
    rolling_mean_3 = (lag_1 + lag_2 + lag_3) / 3
    rolling_mean_5 = (lag_1 + lag_2 + lag_3 + request.demand_last_week) / 4
    rolling_mean_7 = request.demand_last_week
    rolling_std_3 = 5.0
    rolling_std_7 = 10.0
    rolling_mean_14 = request.demand_last_week * 0.98
    expanding_mean = request.past_avg_demand
    expanding_std = 15.0
    
    price_discount = request.price * (1 - request.discount/100)
    demand_ratio = request.past_avg_demand / (request.demand_last_week + 1)
    demand_product = request.past_avg_demand * request.demand_last_week
    demand_diff = request.demand_last_week - request.past_avg_demand
    prep_shelf_ratio = request.prep_time / (request.shelf_life + 1)
    pop_weather = request.population_density * request.weather
    pop_price = request.population_density * request.price / 100
    holiday_event = request.is_holiday * request.special_event
    holiday_cuisine = request.is_holiday * request.cuisine_type
    category_cuisine = request.item_category * request.cuisine_type
    
    features = [
        request.prep_time, request.shelf_life, request.price, request.discount,
        request.past_avg_demand, request.demand_last_week, request.population_density,
        request.is_holiday, request.special_event, request.item_category,
        request.cuisine_type, request.weather,
        12, 3, 3, 0,
        lag_1, lag_2, lag_3, lag_7,
        rolling_mean_3, rolling_mean_5, rolling_mean_7,
        rolling_std_3, rolling_std_7, rolling_mean_14,
        expanding_mean, expanding_std,
        price_discount, demand_ratio, demand_product, demand_diff,
        prep_shelf_ratio, pop_weather, pop_price,
        holiday_event, holiday_cuisine, category_cuisine
    ]
    
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    X_poly = poly.transform(X_scaled)
    
    prediction = model.predict(X_poly)[0]
    prediction = float(max(25, min(250, prediction)))
    
    if prediction < 80:
        level = "Low"
    elif prediction < 150:
        level = "Medium"
    else:
        level = "High"
    
    return {
        "prediction": int(prediction),
        "level": level,
        "stock_min": int(prediction * 1.1),
        "stock_recommended": int(prediction * 1.25)
    }
