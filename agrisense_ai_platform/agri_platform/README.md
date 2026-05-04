# 🌾 AgriSense AI — Agricultural Intelligence Platform

> **Production-grade SaaS-level Crop Yield Prediction & Agricultural Analytics System**

![Python](https://img.shields.io/badge/Python-3.11+-4ade80?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-38bdf8?style=flat-square&logo=streamlit&logoColor=white)
![ML](https://img.shields.io/badge/ML-XGBoost%20%7C%20LightGBM%20%7C%20RF-fbbf24?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-86efac?style=flat-square)

---

## 🎯 Overview

AgriSense AI is a full-stack AI-powered agricultural intelligence platform that predicts crop yields, analyzes farming trends, and provides actionable insights using advanced machine learning. Built for data scientists, agronomists, and policy makers.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🌾 **Dashboard** | Real-time KPIs, yield trends, regional analytics, correlation heatmaps |
| 🔮 **Yield Prediction** | Instant ML predictions with confidence scores and farming recommendations |
| 📉 **Forecasting** | Multi-year time-series forecasting with confidence intervals |
| 🧠 **Insight Engine** | Automated correlation analysis & feature importance insights |
| 🌦 **Live Weather** | OpenWeatherMap integration with agricultural impact assessment |
| 📊 **Model Performance** | Full evaluation: RMSE, MAE, R², residual plots, feature importance |

---

## 🤖 ML Pipeline

- **Linear Regression** (Ridge)
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **XGBoost** *(if installed)*
- **LightGBM** *(if installed)*
- Auto-selects best model by R² score
- 5-fold cross-validation
- Feature engineering (CPI, Rainfall Efficiency, Fertilizer Score)

---

## 🏗️ Project Structure

```
agri_platform/
├── app.py                    # Main Streamlit application
├── src/
│   ├── data_loader.py        # Data ingestion & validation
│   ├── preprocessing.py      # Cleaning, encoding, scaling
│   ├── feature_engineering.py# CPI, efficiency ratios, lag features
│   ├── train_models.py       # Model training & persistence
│   ├── predict.py            # Real-time inference
│   ├── evaluate.py           # Metrics & residual analysis
│   ├── forecasting.py        # Prophet / trend forecasting
│   └── insights.py           # Automated insight generation
├── data/
│   ├── generate_data.py      # Dataset generator (mirrors Kaggle dataset)
│   └── crop_yield.csv        # Generated dataset (5,000 records)
├── models/                   # Saved model artifacts (.pkl)
├── logs/                     # Application logs
├── reports/                  # Exported CSV reports
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### 1. Clone / extract the project
```bash
cd agri_platform
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate the dataset (already done — skip if `data/crop_yield.csv` exists)
```bash
python data/generate_data.py
```

### 5. Run the platform
```bash
streamlit run app.py
```

Open → `http://localhost:8501`

---

## 🌦 Live Weather Setup (Optional)

1. Sign up at [openweathermap.org](https://openweathermap.org) (free tier)
2. Copy your API key
3. Paste it in the **Live Weather** page of the app

---

## 📊 Dataset

- **Source**: Kaggle — Agriculture Crop Yield Dataset  
  https://www.kaggle.com/datasets/samuelotiattakorah/agriculture-crop-yield
- **Records**: 5,000 synthetic records (mimics Kaggle structure)
- **Features**: Crop Type, Region, Season, Area, Rainfall, Temperature, Fertilizer, Pesticide, Humidity
- **Target**: `Yield_Per_Ha` (kg per hectare)

---

## 📈 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit + Plotly + Custom CSS (Glassmorphism) |
| ML Models | scikit-learn, XGBoost, LightGBM |
| Forecasting | Facebook Prophet / Linear Trend |
| Data | Pandas, NumPy |
| Persistence | Joblib |
| Weather API | OpenWeatherMap |
| Fonts | Google Fonts (Outfit, JetBrains Mono) |

---

## 🚀 Future Improvements

- [ ] Satellite imagery integration (NDVI index)
- [ ] Soil health scoring module
- [ ] Multi-language support (Hindi, Tamil, Marathi)
- [ ] Mobile-responsive PWA wrapper
- [ ] PDF report generation with ReportLab
- [ ] REST API backend with FastAPI
- [ ] Docker containerization
- [ ] LSTM deep learning forecasting
- [ ] Crop recommendation system (multi-class classifier)

---

## 👨‍💻 Author

Built with ❤️ as a production-grade portfolio project showcasing end-to-end ML engineering, data science, and SaaS UI design.

---

*AgriSense AI · v2.1.0 · MIT License*
