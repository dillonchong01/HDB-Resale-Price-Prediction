# 🏠 HDB Resale Price Predictor

An end-to-end web application built with a **FastAPI REST API Backend** and a **XGBoost Model** that predicts Singapore HDB resale prices using historical transaction data and location-based features.

🔗 Live Demo: https://hdb-resale-prediction.vercel.app/
🔗 Swagger UI: https://hdb-resale-price-prediction.onrender.com/docs

---

## 📌 Project Overview

This project combines machine learning, geospatial feature engineering, and a Flask-based web API to provide real-time resale price predictions.

The web application supports:

- ⚡ Real-time price prediction via a REST API
- 🧠 Auto-population of town, remaining lease, and floor area using historical transaction data  
- 🚇 Dynamic distance calculation to nearest MRT station and shopping mall, using the OneMap API 

---

## 🤖 Machine Learning Approach

The prediction model is trained using **XGBoost Regressor** on cleaned resale transaction data enriched with engineered features:

- Town  
- Flat Type  
- Floor Area (sqm)  
- Floor Level  
- Remaining Lease  
- Transaction Year & Month  
- Mature Estate Classification  
- Distance to Nearest MRT  
- Distance to Nearest Mall  

Distances are computed using geocoding (OneMap API) and Haversine distance calculations.

**Evaluation Metric:** Mean Absolute Percentage Error (MAPE)

*The XGBoost model achieved approximately 3.67% MAPE using 5-fold OOF Cross Validation*

---

**⚠️Disclaimer:** 
Predictions are based on historical data patterns and should not be used as the sole basis for financial decisions.