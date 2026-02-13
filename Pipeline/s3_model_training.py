import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import joblib

# File Paths
CLEANED_DF = r"Datasets\Cleaned_Resale_Data.csv"
HDB_FEATURES = r"Datasets\HDB_Features.csv"

df = pd.read_csv(CLEANED_DF)
hdb_features = pd.read_csv(HDB_FEATURES)

# Merge Distance_to_MRT and Distance_to_Mall into df
merged_df = pd.merge(
    df,
    hdb_features[['Address', 'Distance_to_MRT', 'Distance_to_Mall']],
    on='Address',
    how='left'
)

# Define Target and Categorical Features
target = "Resale Price"
cat_features = ["Flat Type", "Town"]

# Lock Category Levels
category_mappings = {}
for col in cat_features:
    merged_df[col] = merged_df[col].astype("category")
    category_mappings[col] = merged_df[col].cat.categories

# Define X and y
X = merged_df.drop(columns=[target, "Address"])
y = merged_df[target]

# 5 Fold OOF CV to Find Best No. of Estimators
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
best_iterations = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}")

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = xgb.XGBRegressor(
        objective="reg:absoluteerror",
        learning_rate=0.05,
        max_depth=6,
        n_estimators=20000,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="mape",
        early_stopping_rounds=100,
        device="cuda"
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    oof_preds[valid_idx] = model.predict(X_valid)
    best_iterations.append(model.best_iteration)

    fold_mape = mean_absolute_percentage_error(y_valid, oof_preds[valid_idx])
    print(f"Fold MAPE: {fold_mape:.4f}")
    print(f"Best iteration: {model.best_iteration}")

# Best OOF Mape and Mean Iterations
oof_mape = mean_absolute_percentage_error(y, oof_preds)
print(f"\nOOF MAPE: {oof_mape:.4f}")
best_iteration = int(np.mean(best_iterations))
print(f"\nUsing best_iteration = {best_iteration}")

# Train Model on Full Dataset
final_model = xgb.XGBRegressor(
    objective="reg:absoluteerror",
    learning_rate=0.05,
    max_depth=6,
    n_estimators=15000,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",
    enable_categorical=True,
    eval_metric="mape",
    device="cuda"
)

final_model.fit(X, y)

# Save Model
joblib.dump({"model": final_model, "categories": category_mappings}, "xgboost_model.pkl")