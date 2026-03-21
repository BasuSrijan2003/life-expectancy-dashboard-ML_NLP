"""
train_model.py
Run this ONCE before launching the app.
It creates model/rf_model.pkl, model/feature_names.json, model/feature_stats.json
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# ── 1. LOAD ────────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv("data/Life Expectancy Data.csv")

# ── 2. CLEAN ───────────────────────────────────────────────────────────────────
print("Cleaning data...")
df.columns = df.columns.str.strip()
df["Status"] = df["Status"].map({"Developing": 0, "Developed": 1})
df = df.drop(columns=["Country", "infant deaths", "thinness 5-9 years"])
df["Life expectancy"] = df["Life expectancy"].fillna(df["Life expectancy"].mean())
df = df.fillna(df.mean(numeric_only=True))

print(f"  Cleaned shape: {df.shape}")
print(f"  Missing values remaining: {df.isnull().sum().sum()}")

# ── 3. SPLIT ───────────────────────────────────────────────────────────────────
X = df.drop(columns=["Life expectancy"])
y = df["Life expectancy"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 4. TRAIN ───────────────────────────────────────────────────────────────────
print("\nTraining Random Forest (100 trees)...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# ── 5. EVALUATE ────────────────────────────────────────────────────────────────
predictions = model.predict(X_test)
r2  = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("\n── Model Performance ─────────────────────────────")
print(f"  R² Score : {r2:.4f}  ({r2*100:.2f}%)")
print(f"  MAE      : {mae:.2f} years")
print("──────────────────────────────────────────────────")

# ── 6. SAVE ────────────────────────────────────────────────────────────────────
print("\nSaving model artifacts...")

joblib.dump(model, "model/rf_model.pkl")
print("  ✓ model/rf_model.pkl")

feature_names = X_train.columns.tolist()
with open("model/feature_names.json", "w") as f:
    json.dump(feature_names, f, indent=2)
print("  ✓ model/feature_names.json")

feature_stats = {}
for col in feature_names:
    feature_stats[col] = {
        "min":  round(float(X[col].min()), 4),
        "max":  round(float(X[col].max()), 4),
        "mean": round(float(X[col].mean()), 4),
    }
with open("model/feature_stats.json", "w") as f:
    json.dump(feature_stats, f, indent=2)
print("  ✓ model/feature_stats.json")

print("\n✓ Done. Now run:  streamlit run app.py")