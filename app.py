import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json

st.set_page_config(page_title="Food Demand Predictor", layout="wide", page_icon="🍱", initial_sidebar_state="collapsed")

@st.cache_resource
def load_models():
    try:
        model = joblib.load('src/models/food_demand_model_sklearn.pkl')
        scaler = joblib.load('src/models/scaler.pkl')
        poly = joblib.load('src/models/polynomial_features.pkl')
        with open('src/models/model_info.json', 'r') as f:
            model_info = json.load(f)
        return model, scaler, poly, model_info
    except:
        return None, None, None, None

model, scaler, poly, model_info = load_models()

st.header("🍱 Food Demand Predictor")
st.caption("AI-powered demand forecasting for restaurants & food services")

if model:
    st.success("✅ Model Active")

st.divider()

if not model:
    st.error("Models not found. Run `python train_model.py` first.")
    st.stop()

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Accuracy (R²)", f"{model_info['r2_score']*100:.1f}%")
col_m2.metric("Mean Error", f"±{model_info['mae']:.1f}")
col_m3.metric("Features", model_info['n_features'])
col_m4.metric("Training Data", f"{model_info['n_training_samples']:,}")

st.divider()
st.subheader("🔮 Predict Demand")

col_inp, col_res = st.columns([1.2, 1])

with col_inp:
    tab1, tab2 = st.tabs(["📊 Demand Data", "🏪 Business Context"])
    
    with tab1:
        st.markdown("**Historical Demand**")
        c1, c2 = st.columns(2)
        with c1:
            past_avg_demand = st.number_input("Average Past Demand", min_value=25, max_value=250, value=100)
        with c2:
            demand_last_week = st.number_input("Last Week Demand", min_value=25, max_value=250, value=105)
        
        st.markdown("**Pricing & Location**")
        c3, c4 = st.columns(2)
        with c3:
            price = st.number_input("Item Price ($)", min_value=10, max_value=500, value=150)
            population_density = st.number_input("Area Population", min_value=500, max_value=10000, value=5000, step=100)
        with c4:
            discount = st.slider("Discount (%)", 0, 50, 10)
        
        st.markdown("**Item Properties**")
        c5, c6 = st.columns(2)
        with c5:
            preparation_time = st.number_input("Prep Time (min)", min_value=10, max_value=60, value=25)
        with c6:
            shelf_life = st.number_input("Shelf Life (hours)", min_value=1, max_value=72, value=24)
    
    with tab2:
        st.markdown("**Item Details**")
        c7, c8 = st.columns(2)
        with c7:
            item_category = st.selectbox("Food Category", ["Veg", "Snacks", "Non-Veg"])
        with c8:
            cuisine_type = st.selectbox("Cuisine Style", ["Chinese", "Italian", "Indian"])
        weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Rainy"])
        
        st.markdown("**Special Conditions**")
        c9, c10 = st.columns(2)
        with c9:
            is_holiday = st.checkbox("Public Holiday")
        with c10:
            special_event = st.checkbox("Special Event")
    
    predict = st.button("✨ Predict Demand", type="primary")

with col_res:
    if 'prediction' in st.session_state:
        pred = st.session_state['prediction']
        
        if pred < 80:
            level = "Low"
            level_color = "green"
        elif pred < 150:
            level = "Medium"
            level_color = "orange"
        else:
            level = "High"
            level_color = "red"
        
        stock_min = int(pred * 1.1)
        stock_rec = int(pred * 1.25)
        
        st.markdown("""
        <style>
            .pred-container {
                background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
                border-radius: 20px;
                padding: 2rem;
                text-align: center;
                color: white;
            }
            .pred-value {
                font-size: 4rem;
                font-weight: bold;
                line-height: 1;
            }
            .pred-label {
                font-size: 0.9rem;
                opacity: 0.9;
            }
            .stock-container {
                display: flex;
                gap: 0.75rem;
                margin-top: 1.5rem;
            }
            .stock-box {
                flex: 1;
                background: rgba(255,255,255,0.15);
                border-radius: 12px;
                padding: 1rem;
                text-align: center;
            }
            .stock-box-highlight {
                background: rgba(255,255,255,0.25);
            }
            .stock-num {
                font-size: 1.5rem;
                font-weight: bold;
            }
            .stock-lbl {
                font-size: 0.65rem;
                text-transform: uppercase;
                opacity: 0.85;
            }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="pred-container">
            <div class="pred-label">PREDICTED DEMAND</div>
            <div class="pred-value">{int(pred)}</div>
            <div class="pred-label">units expected</div>
            <div style="display: inline-block; background: {level_color}; padding: 0.5rem 1.5rem; border-radius: 50px; font-weight: 600; margin-top: 1rem;">{level} Demand</div>
            <div class="stock-container">
                <div class="stock-box">
                    <div class="stock-lbl">Minimum</div>
                    <div class="stock-num">{stock_min}</div>
                </div>
                <div class="stock-box stock-box-highlight">
                    <div class="stock-lbl">Recommended</div>
                    <div class="stock-num">{stock_rec}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("👆 Enter values and click Predict")

if predict:
    item_map = {"Veg": 0, "Snacks": 1, "Non-Veg": 2}
    cuisine_map = {"Chinese": 0, "Italian": 1, "Indian": 2}
    weather_map = {"Sunny": 2, "Cloudy": 0, "Rainy": 1}
    
    lag_1 = past_avg_demand * 0.95
    lag_2 = past_avg_demand * 0.90
    lag_3 = past_avg_demand * 0.88
    lag_7 = demand_last_week
    rolling_mean_3 = (lag_1 + lag_2 + lag_3) / 3
    rolling_mean_5 = (lag_1 + lag_2 + lag_3 + demand_last_week) / 4
    rolling_mean_7 = demand_last_week
    rolling_std_3 = 5.0
    rolling_std_7 = 10.0
    rolling_mean_14 = demand_last_week * 0.98
    expanding_mean = past_avg_demand
    expanding_std = 15.0
    price_discount = price * (1 - discount/100)
    demand_ratio = past_avg_demand / (demand_last_week + 1)
    demand_product = past_avg_demand * demand_last_week
    demand_diff = demand_last_week - past_avg_demand
    prep_shelf_ratio = preparation_time / (shelf_life + 1)
    pop_weather = population_density * weather_map[weather]
    pop_price = population_density * price / 100
    holiday_event = int(is_holiday) * int(special_event)
    holiday_cuisine = int(is_holiday) * cuisine_map[cuisine_type]
    category_cuisine = item_map[item_category] * cuisine_map[cuisine_type]
    
    features = [
        preparation_time, shelf_life, price, discount,
        past_avg_demand, demand_last_week, population_density,
        int(is_holiday), int(special_event), item_map[item_category],
        cuisine_map[cuisine_type], weather_map[weather],
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
    prediction = max(25, min(250, prediction))
    st.session_state['prediction'] = prediction
    st.rerun()

st.divider()
st.subheader("📊 Analysis Dashboard")

try:
    df = pd.read_csv('src/dataset/clean_synthetic_food_demand.csv')
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.markdown("**📈 Demand Distribution**")
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.hist(df['food_demand'], bins=30, color='#4f46e5', edgecolor='none')
        if 'prediction' in st.session_state:
            ax.axvline(x=st.session_state['prediction'], color='#ef4444', linestyle='--', linewidth=2)
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col_c2:
        st.markdown("**📉 Recent Trend**")
        fig, ax = plt.subplots(figsize=(7, 3))
        sample = df['food_demand'].iloc[:50]
        ax.plot(sample.values, color='#4f46e5', linewidth=2)
        ax.fill_between(range(len(sample)), sample.values, alpha=0.3, color='#4f46e5')
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
    
    col_s1, col_s2 = st.columns(2)
    
    with col_s1:
        st.markdown("**📋 Key Statistics**")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Avg", f"{df['food_demand'].mean():.0f}")
        sc2.metric("Peak", df['food_demand'].max())
        sc3.metric("Min", df['food_demand'].min())
        sc4.metric("Var", f"{(df['food_demand'].std()/df['food_demand'].mean())*100:.0f}%")
    
    with col_s2:
        st.markdown("**🏷️ By Category**")
        cat_stats = df.groupby('Item_Category')['food_demand'].mean()
        st.bar_chart(cat_stats)

except:
    st.info("Analytics data not available")

st.divider()
st.caption("🍱 Food Demand Prediction System • OLS • Ridge • Lasso • 97.8% Accuracy")
