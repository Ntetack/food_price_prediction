"""
FastAPI Backend — Senegal Food Price Forecasting
Charge les modèles LSTM et expose des endpoints REST.

Lancement :
    uvicorn main:app --reload --port 8000
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── TensorFlow (silencieux) ────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chemin vers les artefacts sauvegardés par le notebook
MODELS_DIR  = Path(os.getenv("MODELS_DIR", "../models"))
DATA_PATH   = Path(os.getenv("DATA_PATH",  "../wfp_food_prices_sen.csv"))
HORIZONS    = [1, 3, 6]
WINDOW_SIZE = 24

# ─────────────────────────────────────────────────────────────
# Chargement des artefacts au démarrage
# ─────────────────────────────────────────────────────────────
logger.info("Chargement des artefacts...")

with open(MODELS_DIR / "metadata.pkl", "rb") as f:
    META = pickle.load(f)

with open(MODELS_DIR / "scaler_lstm_features.pkl", "rb") as f:
    FEAT_SCALER = pickle.load(f)

LSTM_MODELS  = {}
Y_SCALERS    = {}

for h in HORIZONS:
    LSTM_MODELS[h] = load_model(MODELS_DIR / f"lstm_h{h}.keras")
    with open(MODELS_DIR / f"lstm_h{h}_scaler.pkl", "rb") as f:
        Y_SCALERS[h] = pickle.load(f)

# Données historiques (pour les features et l'affichage)
DF_RAW = pd.read_csv(DATA_PATH)
DF_RAW["date"] = pd.to_datetime(DF_RAW["date"])
DF_RAW = DF_RAW.sort_values(["commodity", "market", "date"]).reset_index(drop=True)

logger.info(f"Données chargées : {len(DF_RAW):,} lignes")
logger.info("Artefacts prêts ✓")


# ─────────────────────────────────────────────────────────────
# Feature engineering (identique au notebook)
# ─────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_values(["commodity", "market", "date"])
    grp = d.groupby(["commodity", "market"])["price"]

    d["year"]      = d["date"].dt.year
    d["month"]     = d["date"].dt.month
    d["quarter"]   = d["date"].dt.quarter
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)

    for lag in [1, 2, 3, 6, 12, 24]:
        d[f"lag_{lag}"] = grp.transform(lambda x, l=lag: x.shift(l))

    for w in [3, 6, 12]:
        d[f"roll{w}_mean"] = grp.transform(
            lambda x, w=w: x.shift(1).rolling(w, min_periods=max(2, w // 2)).mean())
        d[f"roll{w}_std"] = grp.transform(
            lambda x, w=w: x.shift(1).rolling(w, min_periods=max(2, w // 2)).std())

    d["mom_1"] = grp.transform(lambda x: x.shift(1).pct_change(1))
    d["mom_3"] = grp.transform(lambda x: x.shift(1).pct_change(3))
    d["yoy"]   = grp.transform(lambda x: x.shift(1).pct_change(12))

    d["commodity_id_enc"] = d.groupby("commodity").ngroup()
    d["market_id_enc"]    = d.groupby("market").ngroup()
    d["category_id_enc"]  = d.groupby("category").ngroup()
    d["region_id_enc"]    = d.groupby("admin1").ngroup()

    return d


DF_FEAT = build_features(DF_RAW)


# ─────────────────────────────────────────────────────────────
# Fonction de prédiction LSTM
# ─────────────────────────────────────────────────────────────
def predict_lstm(commodity: str, market: str, horizon: int) -> float:
    lstm_features = META["lstm_features"]

    series = DF_FEAT[
        (DF_FEAT["commodity"] == commodity) &
        (DF_FEAT["market"]    == market)
    ].sort_values("date")

    if len(series) < WINDOW_SIZE:
        raise ValueError(
            f"Pas assez d'historique pour {commodity} / {market} "
            f"({len(series)} obs, minimum {WINDOW_SIZE})"
        )

    feat_vals = (
        series[lstm_features]
        .tail(WINDOW_SIZE)
        .ffill()
        .bfill()
        .values
    )

    feat_scaled = FEAT_SCALER.transform(feat_vals)
    X_in = feat_scaled[np.newaxis, :, :].astype(np.float32)

    pred_scaled = LSTM_MODELS[horizon].predict(X_in, verbose=0)[0][0]
    pred_price  = Y_SCALERS[horizon].inverse_transform([[pred_scaled]])[0][0]

    return float(max(pred_price, 0))


# ─────────────────────────────────────────────────────────────
# App FastAPI
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Senegal Food Price Forecasting API",
    description="Prédiction des prix alimentaires sur les marchés sénégalais — Modèle LSTM",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Schémas Pydantic
# ─────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    commodity: str = Field(..., example="Millet")
    market:    str = Field(..., example="Thies")
    horizon:   int = Field(..., example=3, description="1, 3 ou 6 mois")


class PredictionResponse(BaseModel):
    commodity:      str
    market:         str
    horizon:        str
    current_price:  float
    predicted_price: float
    change_pct:     float
    signal:         str
    signal_emoji:   str
    history:        list   # [{date, price}]


class BatchRequest(BaseModel):
    market:  str  = Field(..., example="Thies")
    horizon: int  = Field(..., example=3)


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Senegal Food Price Forecasting API"}


@app.get("/meta/commodities", tags=["Metadata"])
def get_commodities():
    """Liste tous les produits disponibles."""
    return {"commodities": sorted(DF_RAW["commodity"].unique().tolist())}


@app.get("/meta/markets", tags=["Metadata"])
def get_markets():
    """Liste tous les marchés disponibles."""
    return {"markets": sorted(DF_RAW["market"].unique().tolist())}


@app.get("/meta/commodities/{market}", tags=["Metadata"])
def get_commodities_by_market(market: str):
    """Liste les produits disponibles pour un marché donné."""
    sub = DF_RAW[DF_RAW["market"] == market]
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"Marché inconnu : {market}")
    return {"market": market, "commodities": sorted(sub["commodity"].unique().tolist())}


@app.get("/history/{commodity}/{market}", tags=["Data"])
def get_history(commodity: str, market: str):
    """Retourne l'historique des prix pour un produit × marché."""
    sub = DF_RAW[
        (DF_RAW["commodity"] == commodity) &
        (DF_RAW["market"]    == market)
    ].sort_values("date")

    if sub.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Pas de données pour {commodity} / {market}"
        )

    history = sub[["date", "price"]].copy()
    history["date"] = history["date"].dt.strftime("%Y-%m-%d")
    return {
        "commodity": commodity,
        "market":    market,
        "n_obs":     len(history),
        "history":   history.to_dict(orient="records"),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Forecast"])
def predict(req: PredictionRequest):
    """Prédit le prix d'un produit sur un marché à h mois."""

    if req.horizon not in HORIZONS:
        raise HTTPException(
            status_code=422,
            detail=f"Horizon invalide : {req.horizon}. Valeurs acceptées : {HORIZONS}"
        )

    # Prix actuel
    sub = DF_RAW[
        (DF_RAW["commodity"] == req.commodity) &
        (DF_RAW["market"]    == req.market)
    ].sort_values("date")

    if sub.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Pas de données pour {req.commodity} / {req.market}"
        )

    current_price = float(sub["price"].iloc[-1])

    try:
        predicted_price = predict_lstm(req.commodity, req.market, req.horizon)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    change_pct = (predicted_price - current_price) / current_price * 100

    if change_pct > 5:
        signal, emoji = "Prix en HAUSSE — Acheter / Stocker maintenant", "📈"
    elif change_pct < -5:
        signal, emoji = "Prix en BAISSE — Attendre pour acheter", "📉"
    else:
        signal, emoji = "Prix STABLE — Pas de signal fort", "➡️"

    # Historique pour le graphique
    history = sub[["date", "price"]].tail(60).copy()
    history["date"] = history["date"].dt.strftime("%Y-%m-%d")

    return PredictionResponse(
        commodity=req.commodity,
        market=req.market,
        horizon=f"+{req.horizon} mois",
        current_price=round(current_price, 1),
        predicted_price=round(predicted_price, 1),
        change_pct=round(change_pct, 2),
        signal=signal,
        signal_emoji=emoji,
        history=history.to_dict(orient="records"),
    )


@app.post("/predict/batch", tags=["Forecast"])
def predict_batch(req: BatchRequest):
    """Prédit les prix de tous les produits d'un marché à h mois."""

    if req.horizon not in HORIZONS:
        raise HTTPException(
            status_code=422,
            detail=f"Horizon invalide : {req.horizon}. Valeurs acceptées : {HORIZONS}"
        )

    commodities = DF_RAW[DF_RAW["market"] == req.market]["commodity"].unique()
    if len(commodities) == 0:
        raise HTTPException(status_code=404, detail=f"Marché inconnu : {req.market}")

    results = []
    for commodity in commodities:
        try:
            sub           = DF_RAW[(DF_RAW["commodity"] == commodity) & (DF_RAW["market"] == req.market)]
            current_price = float(sub["price"].iloc[-1])
            pred_price    = predict_lstm(commodity, req.market, req.horizon)
            change_pct    = (pred_price - current_price) / current_price * 100

            if change_pct > 5:
                signal, emoji = "Acheter / Stocker", "📈"
            elif change_pct < -5:
                signal, emoji = "Attendre", "📉"
            else:
                signal, emoji = "Stable", "➡️"

            results.append({
                "commodity":       commodity,
                "current_price":   round(current_price, 1),
                "predicted_price": round(pred_price, 1),
                "change_pct":      round(change_pct, 2),
                "signal":          signal,
                "signal_emoji":    emoji,
            })
        except Exception:
            continue   # Skip si pas assez d'historique

    results.sort(key=lambda x: x["change_pct"], reverse=True)
    return {"market": req.market, "horizon": f"+{req.horizon} mois", "predictions": results}
