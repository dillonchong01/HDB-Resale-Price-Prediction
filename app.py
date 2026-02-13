import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import re
from Pipeline.utils import compute_nearest_for_address

app = FastAPI()

model = None
category_mappings = None
hdb_features = None
mrt_df = None
mall_df = None

MATURE_ESTATES = [
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH", "CENTRAL",
    "CLEMENTI", "GEYLANG", "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS",
    "QUEENSTOWN", "SERANGOON", "TAMPINES", "TOA PAYOH"
]

# -----------------------------
# Request Schemas
# -----------------------------

class LookupRequest(BaseModel):
    address: str
    flat_type: str

class PredictRequest(BaseModel):
    address: str
    town: str
    flat_type: str
    floor_area: float
    floor_level: int
    remaining_lease: int

# -----------------------------
# Load Data
# -----------------------------

def load_data():
    global model, category_mappings, hdb_features, mrt_df, mall_df

    saved = joblib.load("xgboost_model.pkl")
    model = saved["model"]
    category_mappings = saved["categories"]

    hdb_features = pd.read_csv('Datasets/HDB_Features.csv')
    mrt_df = pd.read_csv('Datasets/MRT_LatLong.csv')
    mall_df = pd.read_csv('Datasets/Mall_LatLong.csv')

load_data()

# -----------------------------
# Utility Functions
# -----------------------------

def clean_address(address):
    cleaned = address.upper()
    cleaned = re.sub(r'\bBLK\b', '', cleaned)
    cleaned = re.sub(r'\bBLOCK\b', '', cleaned)
    cleaned = re.sub(r'\bAVENUE\b', 'AVE', cleaned)
    cleaned = re.sub(r'\bSTREET\b', 'ST', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def lookup_address_features(cleaned_address, flat_type=None):
    match = hdb_features[hdb_features['Address'] == cleaned_address]

    if match.empty:
        return {"found": False}

    row = match.iloc[0]

    floor_area_map = {}
    if 'Floor Area Map' in row and pd.notna(row['Floor Area Map']):
        import ast
        floor_area_map = ast.literal_eval(row['Floor Area Map'])

    result = {
        "found": True,
        "town": row["Town"],
        "lease_commence_date": int(row["Lease Commence Date"]),
        "mature": int(row["Mature"]),
        "distance_to_mrt": float(row["Distance_to_MRT"]),
        "distance_to_mall": float(row["Distance_to_Mall"]),
        "floor_area_map": floor_area_map
    }

    if flat_type and flat_type in floor_area_map:
        result["floor_area"] = floor_area_map[flat_type]

    return result

# -----------------------------
# Routes
# -----------------------------

@app.post("/api/lookup-address")
def lookup_address(data: LookupRequest):

    cleaned = clean_address(data.address)
    flat_type = data.flat_type.strip().upper()

    features = lookup_address_features(cleaned, flat_type)

    if not features["found"]:
        return {
            "found": False,
            "cleaned_address": cleaned
        }

    remaining_lease = 99 - (2026 - features["lease_commence_date"])

    response = {
        "found": True,
        "cleaned_address": cleaned,
        "town": features["town"],
        "remaining_lease": remaining_lease,
        "mature": features["mature"],
        "distance_to_mrt": features["distance_to_mrt"],
        "distance_to_mall": features["distance_to_mall"],
        "floor_area_map": features["floor_area_map"],
        "floor_area": features.get("floor_area")
    }

    return response


@app.post("/api/predict")
def predict(data: PredictRequest):

    town = data.town.strip().upper()
    flat_type = data.flat_type.strip().upper()

    cleaned_address = clean_address(data.address)

    distance_to_mrt = None
    distance_to_mall = None
    mature = None

    features = lookup_address_features(cleaned_address)

    if features["found"]:
        mature = features["mature"]
        distance_to_mrt = features["distance_to_mrt"]
        distance_to_mall = features["distance_to_mall"]

    if distance_to_mrt is None or distance_to_mall is None:
        mature = 1 if town in MATURE_ESTATES else 0
        result = compute_nearest_for_address(cleaned_address, mrt_df, mall_df)

        if not result:
            raise HTTPException(status_code=400, detail="Geocoding failed")

        distance_to_mrt = result["Distance_to_MRT"]
        distance_to_mall = result["Distance_to_Mall"]

    now = datetime.now()

    features_df = pd.DataFrame([{
        "Town": town,
        "Flat Type": flat_type,
        "Floor Area": data.floor_area,
        "Year": now.year,
        "Month": now.month,
        "Floor Level": data.floor_level,
        "Remaining Lease": data.remaining_lease,
        "Mature": mature,
        "Distance_to_MRT": distance_to_mrt,
        "Distance_to_Mall": distance_to_mall
    }])

    for col in category_mappings:
        features_df[col] = pd.Categorical(
            features_df[col],
            categories=category_mappings[col]
        )

    prediction = model.predict(features_df)[0]
    rounded_prediction = round(prediction / 1000) * 1000

    return {
        "predicted_price": int(rounded_prediction),
        "distance_to_mrt": round(distance_to_mrt, 3),
        "distance_to_mall": round(distance_to_mall, 3)
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }