# End-to-End Time Series Forecasting System with FastAPI

## Project Overview

This project is a production-style end-to-end time series forecasting system built for forecasting the next 8 weeks of sales for each state using historical sales data.

The system:
- Trains multiple forecasting models
- Performs feature engineering for time series data
- Automatically compares model performance
- Selects the best model for each state
- Serves forecasts through a REST API using FastAPI

---

# Problem Statement

Forecast the next 8 weeks (56 days) of sales for each state using historical sales data while:
- Handling missing dates and missing values
- Capturing seasonality and trends
- Preventing data leakage
- Automatically selecting the best performing model
- Serving predictions via API

---

# Models Implemented

The following forecasting models were trained and compared:

1. ARIMA / SARIMA
2. Facebook Prophet
3. XGBoost with lag-based features
4. LSTM (Deep Learning)

Model comparison was done using validation RMSE.

---

# Feature Engineering

The following time series features were created:

## Lag Features
- lag_1
- lag_7
- lag_30

## Rolling Statistics
- rolling_mean_7
- rolling_std_7
- rolling_mean_30

## Date-Based Features
- day_of_week
- month
- holiday_flag

---

# Time Series Validation Strategy

To prevent data leakage:
- Chronological train-validation splitting was used
- Future data was never exposed during training
- Validation was performed using time-aware forecasting logic

---

# Missing Data Handling

The system handles:
- Missing dates using continuous date indexing
- Missing values using forward fill/interpolation methods


# Model Selection Pipeline

Workflow:

```text
Raw Excel Dataset
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Train Multiple Models
        ↓
Evaluate RMSE
        ↓
Select Best Model
        ↓
Save Model Artifacts
        ↓
Serve Predictions via FastAPI
```

---

# Project Structure

```text
forecasting-system/
│
├── api/
│   └── app.py
│
├── src/
│   ├── train.py
│   ├── features.py
│   ├── forecast.py
│   └── utils.py
│
├── artifacts/
│   ├── model_map.pkl
│   └── trained_models/
│
├── data/
│
├── notebooks/
│
├── requirements.txt
│
└── README.md
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- Prophet
- XGBoost
- TensorFlow / Keras
- FastAPI
- Uvicorn
- Joblib

---

# API Endpoints

## Home Endpoint

```http
GET /
```

Response:

```json
{
  "message": "Forecast API running"
}
```

---

## Forecast Endpoint

```http
GET /forecast?state=alabama
```

Example Response:

```json
{
  "state": "alabama",
  "model_used": "xgboost",
  "forecast_horizon_days": 56,
  "forecast_8_weeks": [
    {
      "date": "2026-05-08",
      "prediction": "226,140,896.00"
    }
  ]
}
```

---

# How to Run the Project

## 1. Clone Repository

```bash
git clone <repository-url>
cd forecasting-system
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

### Windows

```bash
venv\Scripts\activate
```

### Mac/Linux

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run API Server

```bash
uvicorn api.app:app --reload
```

---

## 5. Open Swagger Docs

Visit:

```text
http://127.0.0.1:8000/docs
```

---

# Model Evaluation

Models were evaluated using RMSE (Root Mean Squared Error).

Lower RMSE indicates better forecasting performance.

The best-performing model for each state is automatically selected and saved in:

```text
artifacts/model_map.pkl
```

---

# Future Improvements

Possible future enhancements:
- Better recursive forecasting strategy for XGBoost
- Improved LSTM inference pipeline
- Hyperparameter tuning
- Docker deployment
- CI/CD integration
- Cloud deployment (AWS/GCP/Azure)
- Real-time monitoring and logging
- Batch forecasting endpoint

---

# Conclusion

This project demonstrates:
- End-to-end forecasting pipeline design
- Time series feature engineering
- Multi-model forecasting comparison
- Automated model selection
- REST API deployment using FastAPI
- Production-style backend architecture

The system is scalable, modular, and designed to resemble a real-world machine learning backend service.
