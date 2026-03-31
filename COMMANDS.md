# Run Commands

This document contains all commands needed to run the Food Demand Prediction project.

## Prerequisites

- Python 3.8 or higher installed
- Windows PowerShell or Command Prompt

---

## 1. Setup Virtual Environment (Recommended)

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Or using PowerShell
.\.venv\Scripts\Activate.ps1
```

---

## 2. Install Dependencies

```powershell
# Install from requirements.txt
pip install -r requirements.txt

# Or install individual packages
pip install pandas numpy scikit-learn tensorflow keras streamlit matplotlib seaborn openpyxl joblib
```

---

## 3. Train the Model

```powershell
# Run the training script
python train_model.py
```

This will:
- Load and analyze the dataset
- Create synthetic data
- Train OLS, Ridge, and Lasso regression models
- Save models to `src/models/`
- Achieve 97.84% accuracy with Lasso Regression

**Output:**
```
Best Regression Model: Lasso
R2 Score: 0.9784 (97.84%)
Target (95%+): YES - ACHIEVED!
```

---

## 4. Run Streamlit Dashboard

```powershell
# Basic command
streamlit run app.py

# With custom port
streamlit run app.py --server.port 8501

# With custom host
streamlit run app.py --server.host 0.0.0.0

# Disable automatic rerun
streamlit run app.py --global.disableWidgetStateDuplicationWarning
```

---

## 5. Access the Dashboard

After running Streamlit, open your browser to:

- Local: `http://localhost:8501`
- Network: `http://<your-ip>:8501`

---

## 6. Quick Reference

| Action | Command |
|--------|---------|
| Activate venv | `.venv\Scripts\activate` |
| Install deps | `pip install -r requirements.txt` |
| Train model | `python train_model.py` |
| Run app | `streamlit run app.py` |
| Deactivate venv | `deactivate` |

---

## 7. Troubleshooting

### Virtual Environment Issues
```powershell
# If execution policy blocks scripts
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then retry activation
.venv\Scripts\Activate.ps1
```

### Package Import Errors
```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Reinstall packages
pip uninstall -y pandas numpy scikit-learn tensorflow
pip install pandas numpy scikit-learn tensorflow
```

### Streamlit Issues
```powershell
# Clear Streamlit cache
streamlit cache clear

# Check Streamlit version
streamlit --version
```

---

## 8. Model Files Location

After training, models are saved to:

```
src/models/
├── food_demand_model.h5          # Keras model (2.9 MB)
├── food_demand_model_sklearn.pkl  # Scikit-learn model
├── scaler.pkl                     # Feature scaler
├── polynomial_features.pkl        # Polynomial transformer
├── label_encoder_*.pkl            # Categorical encoders
└── model_info.json                # Model metadata
```

---

## 9. Dataset Location

```
src/dataset/
├── food_demand_dataset.xlsx       # Original (200 samples)
└── clean_synthetic_food_demand.csv # Clean synthetic (2000 samples)
```

---

## 10. One-Command Setup & Run

```powershell
# Complete setup and run
python -m venv .venv; .venv\Scripts\Activate; pip install -r requirements.txt; python train_model.py; streamlit run app.py
```
