"""
Lance ce script EN LOCAL dans ton environnement avec :
  pip install scikit-learn==1.6.1 numpy==1.26.4 pandas joblib
  python retrain_model.py
 
Il va lire get_around_pricing_project.csv, réentraîner le modèle
et sauvegarder pricing_model.joblib — tu n'as plus qu'à l'uploader sur HF.
"""
 
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
 
# ── 1. Chargement des données ──────────────────────────────────────────────
df = pd.read_csv("get_around_pricing_project.csv")
 
# ── 2. Features / Target ───────────────────────────────────────────────────
TARGET = "rental_price_per_day"
 
NUM_FEATURES  = ["mileage", "engine_power"]
CAT_FEATURES  = ["model_key", "fuel", "paint_color", "car_type"]
BOOL_FEATURES = [
    "private_parking_available",
    "has_gps",
    "has_air_conditioning",
    "automatic_car",
    "has_getaround_connect",
    "has_speed_regulator",
    "winter_tires",
]
 
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES + BOOL_FEATURES
 
X = df[ALL_FEATURES]
y = df[TARGET]
 
# ── 3. Preprocessing ───────────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ("num",  StandardScaler(),                          NUM_FEATURES),
    ("cat",  OneHotEncoder(handle_unknown="ignore", sparse_output=False),    CAT_FEATURES),
    ("bool", "passthrough",                             BOOL_FEATURES),
])
 
# ── 4. Pipeline ────────────────────────────────────────────────────────────
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
    )),
])
 
# ── 5. Entraînement ────────────────────────────────────────────────────────
pipeline.fit(X, y)
print("✅ Modèle entraîné")
 
# ── 6. Sauvegarde ──────────────────────────────────────────────────────────
joblib.dump(pipeline, "pricing_model.joblib")
print("✅ pricing_model.joblib sauvegardé — upload-le sur HF !")
 
# ── 7. Test rapide ─────────────────────────────────────────────────────────
test = pd.DataFrame([[
    "Renault", 80000, 120, "diesel", "black", "sedan",
    True, True, True, False, True, True, False
]], columns=ALL_FEATURES)
 
pred = pipeline.predict(test)[0]
print(f"✅ Test prediction : {pred:.0f} € / jour")
