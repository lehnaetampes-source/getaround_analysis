from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Any
import joblib
import pandas as pd
import numpy as np
 
app = FastAPI(docs_url=None, redoc_url=None)
 
# ── Load model ────────────────────────────────────────────────────────────
try:
    model = joblib.load("pricing_model.joblib")
except:
    model = None
 
# ── Input schema ──────────────────────────────────────────────────────────
class PredictInput(BaseModel):
    input: List[List[Any]]
 
COLUMNS = [
    "model_key", "mileage", "engine_power", "fuel", "paint_color",
    "car_type", "private_parking_available", "has_gps",
    "has_air_conditioning", "automatic_car", "has_getaround_connect",
    "has_speed_regulator", "winter_tires"
]
 
# ── /predict ──────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(data: PredictInput):
    df = pd.DataFrame(data.input, columns=COLUMNS)
    # Convert types
    for col in ["mileage", "engine_power"]:
        df[col] = df[col].astype(float)
    for col in ["private_parking_available", "has_gps", "has_air_conditioning",
                "automatic_car", "has_getaround_connect", "has_speed_regulator", "winter_tires"]:
        df[col] = df[col].astype(bool)
    predictions = model.predict(df).tolist()
    return {"prediction": predictions}
 
# ── /docs ─────────────────────────────────────────────────────────────────
@app.get("/docs", response_class=HTMLResponse)
def docs():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Getaround API Documentation</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700;800&display=swap');
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'DM Sans', sans-serif; background: #FAFAFA; color: #1A1A1A; }
  header { background: #fff; border-bottom: 2px solid #E8DFF0; padding: 24px 48px; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 24px; font-weight: 800; }
  header span { color: #7B2D8B; }
  header .badge { background: #7B2D8B; color: white; font-size: 11px; padding: 4px 10px; border-radius: 20px; font-weight: 600; letter-spacing: 1px; }
  main { max-width: 860px; margin: 48px auto; padding: 0 24px; }
  h2 { font-size: 20px; font-weight: 800; margin-bottom: 8px; color: #1A1A1A; }
  .subtitle { color: #888; font-size: 14px; margin-bottom: 40px; }
  .endpoint { background: white; border: 1px solid #E8DFF0; border-radius: 12px; padding: 28px; margin-bottom: 24px; }
  .endpoint-header { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
  .method { background: #7B2D8B; color: white; font-size: 12px; font-weight: 700; padding: 4px 12px; border-radius: 6px; letter-spacing: 1px; }
  .method.get { background: #2D8B6B; }
  .path { font-size: 18px; font-weight: 700; font-family: monospace; }
  .desc { color: #555; font-size: 14px; margin-bottom: 16px; }
  .section-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #7B2D8B; margin-bottom: 8px; margin-top: 16px; }
  pre { background: #F5F0F7; border-radius: 8px; padding: 16px; font-size: 13px; overflow-x: auto; line-height: 1.6; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; }
  th { background: #F5F0F7; padding: 10px 14px; text-align: left; font-weight: 700; color: #7B2D8B; }
  td { padding: 10px 14px; border-bottom: 1px solid #E8DFF0; }
  tr:last-child td { border-bottom: none; }
  .note { background: #F5F0F7; border-left: 4px solid #7B2D8B; border-radius: 0 8px 8px 0; padding: 12px 16px; font-size: 13px; color: #555; margin-top: 16px; }
</style>
</head>
<body>
<header>
  <h1>get<span>around</span></h1>
  <div class="badge">API DOCS</div>
</header>
<main>
  <h2>Getaround Pricing API</h2>
  <p class="subtitle">API de prédiction de prix pour l'optimisation tarifaire des véhicules Getaround.</p>
 
  <!-- /predict -->
  <div class="endpoint">
    <div class="endpoint-header">
      <span class="method">POST</span>
      <span class="path">/predict</span>
    </div>
    <p class="desc">Prédit le prix optimal par jour pour un ou plusieurs véhicules en fonction de leurs caractéristiques.</p>
 
    <div class="section-label">Input — Body JSON</div>
    <table>
      <tr><th>Champ</th><th>Type</th><th>Description</th></tr>
      <tr><td>input</td><td>array of arrays</td><td>Liste de véhicules, chaque véhicule est un tableau de 13 valeurs</td></tr>
    </table>
 
    <div class="section-label">Ordre des valeurs par véhicule</div>
    <table>
      <tr><th>#</th><th>Champ</th><th>Type</th><th>Exemple</th></tr>
      <tr><td>0</td><td>model_key</td><td>string</td><td>"Renault"</td></tr>
      <tr><td>1</td><td>mileage</td><td>float</td><td>80000</td></tr>
      <tr><td>2</td><td>engine_power</td><td>float</td><td>120</td></tr>
      <tr><td>3</td><td>fuel</td><td>string</td><td>"diesel"</td></tr>
      <tr><td>4</td><td>paint_color</td><td>string</td><td>"black"</td></tr>
      <tr><td>5</td><td>car_type</td><td>string</td><td>"sedan"</td></tr>
      <tr><td>6</td><td>private_parking_available</td><td>bool</td><td>true</td></tr>
      <tr><td>7</td><td>has_gps</td><td>bool</td><td>true</td></tr>
      <tr><td>8</td><td>has_air_conditioning</td><td>bool</td><td>true</td></tr>
      <tr><td>9</td><td>automatic_car</td><td>bool</td><td>false</td></tr>
      <tr><td>10</td><td>has_getaround_connect</td><td>bool</td><td>true</td></tr>
      <tr><td>11</td><td>has_speed_regulator</td><td>bool</td><td>true</td></tr>
      <tr><td>12</td><td>winter_tires</td><td>bool</td><td>false</td></tr>
    </table>
 
    <div class="section-label">Exemple de requête</div>
    <pre>curl -X POST https://your-space.hf.space/predict \\
  -H "Content-Type: application/json" \\
  -d '{"input": [["Renault", 80000, 120, "diesel", "black", "sedan", true, true, true, false, true, true, false]]}'</pre>
 
    <div class="section-label">Exemple de réponse</div>
    <pre>{"prediction": [145.0]}</pre>
 
    <div class="note">💡 Vous pouvez passer plusieurs véhicules en même temps dans le tableau <code>input</code>.</div>
  </div>
 
  <!-- /health -->
  <div class="endpoint">
    <div class="endpoint-header">
      <span class="method get">GET</span>
      <span class="path">/health</span>
    </div>
    <p class="desc">Vérifie que l'API est en ligne et que le modèle est bien chargé.</p>
    <div class="section-label">Exemple de réponse</div>
    <pre>{"status": "ok", "model": "loaded"}</pre>
  </div>
 
</main>
</body>
</html>
"""
 
# ── /health ───────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "loaded" if model else "unavailable"}