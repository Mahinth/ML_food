# Run Commands - Food Demand Prediction

## Quick Start

### 1. Run Frontend (HTTP Server)
```powershell
.venv\Scripts\python.exe -m http.server 8888
```
Then open: **http://localhost:8888/index.html**

### 2. Run Backend (API) - Optional
```powershell
.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```
API Docs: **http://localhost:8000/docs**

### 3. Run Streamlit (All-in-one)
```powershell
.venv\Scripts\streamlit.exe run app.py
```

### 4. Train Model
```powershell
.venv\Scripts\python.exe train_model.py
```

---

## Full Setup (One Command)
```powershell
# Terminal 1: Backend
.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Frontend  
.venv\Scripts\python.exe -m http.server 8888
```

---

## Access URLs

| Service | URL |
|---------|-----|
| Frontend | http://localhost:8888/index.html |
| API | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |

---

## If 404 Error
Make sure to type the URL manually in browser address bar:
```
http://localhost:8888/index.html
```
