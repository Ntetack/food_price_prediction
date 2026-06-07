# 🌍 Senegal Food Price Forecasting — Application

Application de prévision des prix alimentaires sur les marchés sénégalais.  
**Backend** : FastAPI · **Frontend** : Streamlit · **Modèle** : LSTM

---

## 📁 Structure du projet

```
projet/
│
├── notebook/
│   └── senegal_price_forecasting_v2.ipynb   ← entraînement des modèles
│
├── models/                                   ← générés par le notebook
│   ├── lstm_h1.keras
│   ├── lstm_h3.keras
│   ├── lstm_h6.keras
│   ├── lstm_h1_scaler.pkl
│   ├── lstm_h3_scaler.pkl
│   ├── lstm_h6_scaler.pkl
│   ├── scaler_lstm_features.pkl
│   └── metadata.pkl
│
├── wfp_food_prices_sen.csv                   ← données brutes
│
├── backend/
│   └── main.py                               ← API FastAPI
│
├── frontend/
│   └── app.py                                ← Interface Streamlit
│
└── requirements.txt
```

---

## ⚙️ Installation

```bash
# 1. Créer un environnement virtuel
conda activate ml_env   # ou python -m venv venv && source venv/bin/activate

# 2. Installer les dépendances
pip install -r requirements.txt
```

---

## 🚀 Lancement

### Étape 1 — Entraîner les modèles (si pas déjà fait)
```bash
jupyter notebook senegal_price_forecasting_v2.ipynb
# Exécuter toutes les cellules → les modèles seront sauvegardés dans ./models/
```

### Étape 2 — Démarrer le backend FastAPI
```bash
cd backend
uvicorn main:app --reload --port 8000
```
→ API disponible sur http://localhost:8000  
→ Documentation interactive : http://localhost:8000/docs

### Étape 3 — Démarrer le frontend Streamlit
```bash
# Dans un nouveau terminal
cd frontend
streamlit run app.py
```
→ Interface disponible sur http://localhost:8501

---

## 🔌 Endpoints API

| Méthode | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/meta/markets` | Liste des marchés |
| GET | `/meta/commodities/{market}` | Produits d'un marché |
| GET | `/history/{commodity}/{market}` | Historique des prix |
| POST | `/predict` | Prévision pour 1 produit |
| POST | `/predict/batch` | Prévisions pour tous les produits d'un marché |

### Exemple d'appel API

```bash
# Prévision Millet / Thiès à +3 mois
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"commodity": "Millet", "market": "Thies", "horizon": 3}'
```

---

## ⚠️ Prérequis

- Le dossier `models/` doit être au même niveau que `backend/` et `frontend/`
- Le fichier `wfp_food_prices_sen.csv` doit être au même niveau
- Python 3.10+, TensorFlow 2.15+

---

## 🔧 Variables d'environnement (optionnel)

```bash
export MODELS_DIR=/chemin/vers/models
export DATA_PATH=/chemin/vers/wfp_food_prices_sen.csv
```
