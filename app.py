import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import validate_data, initial_preprocess
from src.feature_engineering import create_features, create_classification_target
from src.models import (
    train_test_split_time_series, 
    get_regression_model, 
    get_classification_model, 
    evaluate_regression, 
    evaluate_classification
)
from src.visualization import (
    plot_demand_vs_time, 
    plot_actual_vs_predicted, 
    plot_confusion_matrix, 
    plot_correlation_heatmap
)

# Page Config
st.set_page_config(page_title="Food Demand Forecaster", layout="wide", page_icon="🍱")

# --- CUSTOM PREMIUM CSS ---
st.markdown("""
<style>
    /* Main Background and Fonts */
    .main {
        background-color: #f0f2f6;
    }
    
    /* Premium Title Styling */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        color: #1e3a8a;
        font-size: 3rem;
        margin-bottom: 0rem;
    }
    
    .sub-title {
        color: #64748b;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    /* Card-like Containers */
    div.stButton > button:first-child {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    
    div.stButton > button:hover {
        background-color: #1d4ed8;
        transform: translateY(-2px);
    }

    /* Custom Metric Styling */
    [data-testid="stMetricValue"] {
        font-weight: 700;
        color: #1e293b;
    }

    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }

    /* Explanation Boxes */
    .guide-box {
        background: #e0f2fe;
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #0ea5e9;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# Session State
if 'data' not in st.session_state: st.session_state.data = None
if 'processed_data' not in st.session_state: st.session_state.processed_data = None
if 'reg_model' not in st.session_state: st.session_state.reg_model = None
if 'cls_model' not in st.session_state: st.session_state.cls_model = None

# Mandatory Features List
MANDATORY_FEATURES = [
    'Item_Category', 'Cuisine_Type', 'Preparation_Time', 'Shelf_Life', 
    'Price', 'Discount_Percentage', 'Past_Average_Demand', 'Demand_Last_Week', 
    'Population_Density', 'Weather_Type', 'Is_Holiday', 'Special_Event', 
    'Hour', 'Is_Weekend', 'Lag_1', 'Lag_7', 'Lag_14', 
    'Rolling_Mean_7', 'Rolling_Mean_14', 'Rolling_Std_7'
]

# --- SIDEBAR NAVIGATION (Simplified) ---
st.sidebar.image("https://img.icons8.com/clouds/200/food-delivery.png", width=100)
st.sidebar.title("App Guide")
menu = st.sidebar.radio("Go to Step:", [
    "🏠 Homepage", 
    "Step 1: Get Your Data", 
    "Step 2: Check Your Data", 
    "Step 3: Prepare Your Data", 
    "Step 4: Teach the Computer", 
    "Step 5: See How Smart it Is", 
    "Step 6: Use it for Your Shop"
])

# --- HOME PAGE ---
if menu == "🏠 Homepage":
    st.markdown('<h1 class="main-title">🍱 Food Demand Forecaster</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Predict your food demand easily, even if you are not a data expert.</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
        <h3>🌟 What does this do?</h3>
        This app helps you figure out how much food you need to cook or order.
        <ul>
            <li><b>Exact Forecast:</b> Tells you the exact number of plates or items needed.</li>
            <li><b>Demand Levels:</b> Tells you if it will be a <b>Low, Medium, or Busy</b> day.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
        <h3>🚦 How to use?</h3>
        Just follow the steps in the sidebar from <b>Step 1 to Step 6</b>.
        <br><br>
        <i>Note: No need to know heavy math, the computer does it for you!</i>
        </div>
        """, unsafe_allow_html=True)
    
    st.image("https://images.unsplash.com/photo-1504674900247-0877df9cc836?q=80&w=1000&auto=format&fit=crop", use_container_width=True)

# --- STEP 1: GET DATA ---
elif menu == "Step 1: Get Your Data":
    st.markdown('<h1 class="main-title">Step 1: Get Your Data</h1>', unsafe_allow_html=True)
    st.markdown('<div class="guide-box"><b>Goal:</b> Upload your sales history file here so the computer can learn from your past.</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your sales file (CSV)", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.data = df
        st.success("✅ Success! Your data is loaded.")
        st.dataframe(df.head(), use_container_width=True)
    else:
        st.info("Don't have a file? Use the button below to see an example.")
        if st.button("Click to use Example Data"):
            num_rows = 500
            dates = pd.date_range('2024-01-01', periods=num_rows, freq='h')
            sample_data = pd.DataFrame({
                'timestamp': dates,
                'food_demand': np.random.randint(50, 500, size=num_rows),
                'Item_Category': np.random.choice(['Pizza', 'Burger', 'Pasta'], size=num_rows),
                'Cuisine_Type': np.random.choice(['Italian', 'American', 'Mexican'], size=num_rows),
                'Preparation_Time': np.random.randint(10, 45, size=num_rows),
                'Shelf_Life': np.random.randint(1, 10, size=num_rows),
                'Price': np.random.uniform(10, 50, size=num_rows),
                'Discount_Percentage': np.random.choice([0, 5, 10], size=num_rows),
                'Population_Density': np.random.randint(1000, 5000, size=num_rows),
                'Weather_Type': np.random.choice(['Sunny', 'Rainy'], size=num_rows),
                'Is_Holiday': np.random.choice([0, 1], size=num_rows, p=[0.9, 0.1]),
                'Special_Event': np.random.choice([0, 1], size=num_rows, p=[0.95, 0.05])
            })
            st.session_state.data = sample_data
            st.success("✅ Example data is ready!")
            st.dataframe(sample_data.head(), use_container_width=True)

# --- STEP 2: CHECK DATA ---
elif menu == "Step 2: Check Your Data":
    st.markdown('<h1 class="main-title">Step 2: Check Your Data</h1>', unsafe_allow_html=True)
    if st.session_state.data is not None:
        df = st.session_state.data
        valid, msg = validate_data(df)
        if valid:
            st.success("✅ Your file looks perfect!")
            st.markdown('<div class="guide-box">Below is a chart showing your past sales. <b>Up and Down lines</b> show how demand changed over time.</div>', unsafe_allow_html=True)
            df_plot = initial_preprocess(df.copy())
            st.pyplot(plot_demand_vs_time(df_plot))
        else:
            st.error(f"❌ Something is wrong: {msg}")
    else:
        st.warning("Please go to Step 1 first.")

# --- STEP 3: PREPARE DATA ---
elif menu == "Step 3: Prepare Your Data":
    st.markdown('<h1 class="main-title">Step 3: Prepare Your Data</h1>', unsafe_allow_html=True)
    if st.session_state.data is not None:
        st.markdown('<div class="guide-box">Before teaching the computer, we need to clean the data. This creates <b>Extra Info</b> like "Is it a weekend?" or "What was the demand yesterday?"</div>', unsafe_allow_html=True)
        if st.button("Click to Prepare Everything ⚡"):
            with st.spinner("Cleaning and organizing..."):
                df_proc = initial_preprocess(st.session_state.data.copy())
                df_feat = create_features(df_proc)
                cat_cols = ['Item_Category', 'Cuisine_Type', 'Weather_Type']
                for col in cat_cols:
                    if col in df_feat.columns: df_feat[col] = df_feat[col].astype('category').cat.codes
                df_final = create_classification_target(df_feat)
                st.session_state.processed_data = df_final
                st.success("✅ Everything is ready for the Computer to learn!")
                st.dataframe(df_final.head(), use_container_width=True)
    else:
        st.warning("Please go to Step 1 first.")

# --- STEP 4: TEACH COMPUTER ---
elif menu == "Step 4: Teach the Computer":
    st.markdown('<h1 class="main-title">Step 4: Teach the Computer</h1>', unsafe_allow_html=True)
    if st.session_state.processed_data is not None:
        st.markdown('<div class="guide-box">Select the <b>Brain Type</b> (Model) you want to use. Don\'t worry, the normal ones are selected by default.</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            r_choice = st.selectbox("How to calculate amount?", ["OLS Linear Regression", "Ridge Regression", "Lasso Regression"])
        with col2:
            c_choice = st.selectbox("How to guess Busy days?", ["Decision Tree Classifier", "Support Vector Machine (SVM)", "Naïve Bayes"])
            
        if st.button("Start Learning 🧠"):
            with st.spinner("Teaching the computer... please wait..."):
                df = st.session_state.processed_data
                existing_features = [c for c in MANDATORY_FEATURES if c in df.columns]
                
                # Reg
                X_tr, X_te, y_tr, y_te = train_test_split_time_series(df, 'food_demand', existing_features)
                rm = get_regression_model(r_choice); rm.fit(X_tr, y_tr)
                rm_metrics = evaluate_regression(y_te, rm.predict(X_te))
                
                # Cls
                Xc_tr, Xc_te, yc_tr, yc_te = train_test_split_time_series(df, 'demand_category', existing_features)
                cm = get_classification_model(c_choice); cm.fit(Xc_tr, yc_tr)
                cm_metrics = evaluate_classification(yc_te, cm.predict(Xc_te))
                
                st.session_state.reg_model = {"model": rm, "metrics": rm_metrics, "name": r_choice, "X_te": X_te, "y_te": y_te, "y_pred": rm.predict(X_te)}
                st.session_state.cls_model = {"model": cm, "metrics": cm_metrics, "name": c_choice, "X_te": Xc_te, "y_te": yc_te, "y_pred": cm.predict(Xc_te)}
                st.success("✅ The computer has finished learning!")
    else:
        st.warning("Please go to Step 3 first.")

# --- STEP 5: SEE RESULTS ---
elif menu == "Step 5: See How Smart it Is":
    st.markdown('<h1 class="main-title">Step 5: How Smart is the Computer?</h1>', unsafe_allow_html=True)
    if st.session_state.reg_model:
        r_info = st.session_state.reg_model
        c_info = st.session_state.cls_model
        
        st.subheader("🏁 Overall Performance")
        score = round(r_info['metrics']['R2 Score'] * 100, 1)
        
        if score > 80: color, desc = "#16a34a", "Excellent - Very Reliable"
        elif score > 50: color, desc = "#ca8a04", "Good - Can rely but check cautiously"
        else: color, desc = "#dc2626", "Needs work - Not very reliable"
        
        st.markdown(f"""
        <div style="background:{color}; color:white; padding:20px; border-radius:10px; text-align:center;">
            <h2>Reliability Score: {score}%</h2>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### 📊 Exact Amount Accuracy")
            st.metric("Simple Error Rate", round(r_info['metrics']['MAE'], 2))
            st.markdown("<i>This is how many plates the computer might be wrong by.</i>", unsafe_allow_html=True)
        with col2:
            st.write("#### 🎯 Busy Level Accuracy")
            st.metric("Correct Guesses", f"{round(c_info['metrics']['Accuracy']*100, 1)}%")
            st.markdown("<i>Percentage of times it correctly guessed if it's High or Low demand.</i>", unsafe_allow_html=True)
            
        st.divider()
        st.write("#### 📈 How close were we?")
        st.pyplot(plot_actual_vs_predicted(r_info['y_te'], r_info['y_pred']))
        st.info("The closer the dots are to the line, the better!")

    else:
        st.warning("Please go to Step 4 first.")

# --- STEP 6: PREDICTION ---
elif menu == "Step 6: Use it for Your Shop":
    st.markdown('<h1 class="main-title">Step 6: Use it for Your Shop</h1>', unsafe_allow_html=True)
    if st.session_state.reg_model:
        st.markdown('<div class="guide-box">Enter the details for today or tomorrow to see what the computer thinks.</div>', unsafe_allow_html=True)
        
        df = st.session_state.processed_data
        existing_features = [c for c in MANDATORY_FEATURES if c in df.columns]
        
        with st.container():
            st.subheader("📝 Enter Details")
            grid = st.columns(3)
            input_data = {}
            for i, feat in enumerate(existing_features):
                def_val = float(df[feat].iloc[-1])
                input_data[feat] = grid[i % 3].number_input(f"{feat}", value=def_val)
            
            if st.button("Calculate Forecast 🔮"):
                in_df = pd.DataFrame([input_data])
                reg_p = st.session_state.reg_model['model'].predict(in_df)[0]
                cls_p = st.session_state.cls_model['model'].predict(in_df)[0]
                
                st.balloons()
                st.markdown("---")
                r1, r2 = st.columns(2)
                r1.markdown(f"""
                <div class="info-card" style="text-align:center;">
                    <p style="font-size:1.2rem; margin-bottom:0;">Forecasted Demand</p>
                    <h1 style="color:#2563eb; font-size:4rem; margin-top:0;">{int(reg_p)}</h1>
                    <p>items/plates</p>
                </div>
                """, unsafe_allow_html=True)
                
                cat_color = "#16a34a" if cls_p == "Low" else ("#ca8a04" if cls_p == "Medium" else "#dc2626")
                r2.markdown(f"""
                <div class="info-card" style="text-align:center;">
                    <p style="font-size:1.2rem; margin-bottom:0;">Day Type</p>
                    <h1 style="color:{cat_color}; font-size:4rem; margin-top:0;">{cls_p}</h1>
                    <p>Demand Level</p>
                </div>
                """, unsafe_allow_html=True)
                
    else:
        st.warning("Please go to Step 4 first.")
