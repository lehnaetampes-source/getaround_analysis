# 🚗 Getaround Analysis

Getaround est le Airbnb des voitures. Ce projet analyse les retards de restitution et prédit le prix optimal des locations.

## 🎯 Objectif

- **Analyse** : comprendre l'impact des retards sur les locations suivantes et recommander un seuil minimum entre deux locations
- **ML** : prédire le prix optimal de location journalier à partir des caractéristiques du véhicule

## 🔗 Liens

- **Dashboard** : https://NANA12A-getaround-pricing-simulator.hf.space
- **API** : https://nana12a-getaround-api.hf.space/predict
- **FASTAPI** : https://nana12a-getaround-api.hf.space/predict
- **Documentation API** : https://nana12a-getaround-api.hf.space/docs
- **health check** : https://nana12a-getaround-api.hf.space/health

## 📁 Structure du projet

```
├── api.py                        # FastAPI — endpoint /predict /docs /health
├── app.py                        # Dashboard Streamlit — 3 pages
├── Getaround_EDA_FINAL.ipynb     # Analyse exploratoire des retards
├── Getaround_ML_Pricing.ipynb    # Entraînement du modèle de pricing
├── resave_model.py               # Script pour réentraîner le modèle
├── test_api.py                   # Script de test de l'API
├── pricing_model.joblib          # Modèle entraîné (HistGradientBoosting)
├── Dockerfile                    # Configuration Docker
├── requirements.txt              # Dépendances Python
└── start.sh                      # Script de démarrage
```

## 🤖 Modèle ML

- **Algorithme** : HistGradientBoostingRegressor
- **Features** : kilométrage, puissance, carburant, marque, type, équipements
- **Performances** : R²=0.743, MAE=10.7 €/jour

## 📊 Résultats EDA

- 44.1% des locations sont rendues en retard
- Seulement 1.3% créent un conflit avec la location suivante
- **Recommandation** : seuil de 60 minutes, scope Connect en priorité
  - 65% des conflits résolus
  - Seulement 0.8% du CA affecté

## 🚀 Lancer en local

```bash
# Installer les dépendances
pip install -r requirements.txt

# Lancer le dashboard
streamlit run app.py

# Lancer l'API (dans un autre terminal)
uvicorn api:app --host 0.0.0.0 --port 8000

# Tester l'API
python test_api.py
```

## 📡 Utiliser l'API

```python
import requests

response = requests.post(
    "https://nana12a-getaround-api.hf.space/predict",
    json={
        "input": [["Renault", 80000, 120, "diesel", "black", "sedan",
                   True, True, True, False, True, True, False]]
    }
)
print(response.json())
# {"prediction": [143.43]}
```
