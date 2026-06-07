"""
Streamlit Frontend & Backend Unifié — Senegal Food Price Forecasting
Application autonome pour Streamlit Cloud (Sans FastAPI).

Lancement :
    streamlit run app.py
"""

import os
import pickle
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from pathlib import Path

# Silence TensorFlow au démarrage
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.models import load_model


# ─────────────────────────────────────────────────────────────
# CONFIGURATION & CONSTANTES
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HORIZONS = [1, 3, 6]
WINDOW_SIZE = 24

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Marchés Sénégal - Prévision des Prix",
    page_icon="🇸🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style CSS personnalisé pour l'interface
st.markdown("""
<style>
    .signal-card {
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin: 0.5rem 0;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
    }
    .signal-up   { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .signal-down { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .signal-flat { background: #fff3cd; color: #856404; border: 1px solid #ffeeba; }

    .metric-box {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-label { font-size: 0.8rem; color: #6c757d; margin-bottom: 4px; }
    .metric-value { font-size: 1.6rem; font-weight: 700; color: #212529; }
    .metric-sub   { font-size: 0.85rem; color: #6c757d; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# FONCTIONS BACKEND (Intégrées)
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_all_artifacts():
    """Charge les modèles, scalers et métadonnées une seule fois."""
    # Chemins relatifs par défaut pour Streamlit Cloud
    models_dir = Path("models")
    data_path = Path("wfp_food_prices_sen.csv")
    
    logger.info("Chargement des artefacts ML...")
    
    with open(models_dir / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)

    with open(models_dir / "scaler_lstm_features.pkl", "rb") as f:
        feat_scaler = pickle.load(f)

    lstm_models = {}
    y_scalers = {}

    for h in HORIZONS:
        lstm_models[h] = load_model(models_dir / f"lstm_h{h}.keras")
        with open(models_dir / f"lstm_h{h}_scaler.pkl", "rb") as f:
            y_scalers[h] = pickle.load(f)
            
    # Chargement du dataset brut
    df_raw = pd.read_csv(data_path)
    df_raw["date"] = pd.to_datetime(df_raw["date"])
    df_raw = df_raw.sort_values(["commodity", "market", "date"]).reset_index(drop=True)
    
    logger.info("Tous les artefacts ont été chargés avec succès ✓")
    return meta, feat_scaler, lstm_models, y_scalers, df_raw


# Initialisation et chargement global des données et modèles
try:
    META, FEAT_SCALER, LSTM_MODELS, Y_SCALERS, DF_RAW = load_all_artifacts()
except Exception as e:
    st.error(f"❌ Erreur lors du chargement des modèles ou du fichier CSV : {e}")
    st.info("Vérifiez que vos dossiers 'models/' et votre fichier CSV sont bien à la racine de votre dépôt Git.")
    st.stop()


@st.cache_data
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Génération de features temporelles et lags (identique au notebook)."""
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

# Génération globale des features
DF_FEAT = build_features(DF_RAW)


def predict_lstm(commodity: str, market: str, horizon: int) -> float:
    """Exécute la prédiction LSTM en local sur le serveur Streamlit."""
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
# INTERFACE SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://flagcdn.com/w80/sn.png", width=60)
    st.title("Marchés Sénégal")
    st.caption("Outil de prévision des prix alimentaires — Modèle LSTM")
    st.divider()

    markets = sorted(DF_RAW["market"].unique().tolist())
    market = st.selectbox("📍 Marché", markets, index=markets.index("Thies") if "Thies" in markets else 0)

    # Produits disponibles sur le marché sélectionné
    sub_commodities = sorted(DF_RAW[DF_RAW["market"] == market]["commodity"].unique().tolist())
    commodity = st.selectbox("🛒 Produit", sub_commodities) if sub_commodities else None

    horizon = st.select_slider(
        "⏱️ Horizon de prévision",
        options=HORIZONS,
        value=3,
        format_func=lambda x: f"+{x} mois"
    )

    st.divider()
    predict_btn = st.button("🔮 Lancer la prévision", type="primary", use_container_width=True)
    batch_btn   = st.button("📊 Voir tous les produits", use_container_width=True)

    st.divider()
    st.caption(f"Mode : Autonome (Streamlit Cloud)")
    st.caption(f"Modèle : LSTM (fenêtre 24 mois)")
    st.caption(f"Données : WFP 2000–2024")


# ─────────────────────────────────────────────────────────────
# PAGE PRINCIPALE
# ─────────────────────────────────────────────────────────────
st.title("Prévision des Prix Alimentaires — Sénégal")
st.caption(f"Dernière mise à jour : {datetime.now().strftime('%d/%m/%Y %H:%M')}")

tab_predict, tab_batch = st.tabs([
    "🔮 Prévision produit",
    "📊 Tableau de bord marché"
])


# ════════════════════════════════════════════════════════════
# ONGLETS 1 — Prévision produit
# ════════════════════════════════════════════════════════════
with tab_predict:
    if not commodity:
        st.info("Sélectionnez un marché et un produit dans la barre latérale.")
        st.stop()

    # Récupération de l'historique directement depuis le dataframe en mémoire
    hist_df = DF_RAW[(DF_RAW["commodity"] == commodity) & (DF_RAW["market"] == market)].sort_values("date")
    n_obs = len(hist_df)

    st.subheader(f"{commodity} — {market}")
    st.caption(f"{n_obs} observations disponibles · 2000–2024")

    # Graphique historique
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=hist_df["date"], y=hist_df["price"],
        mode="lines", name="Prix historique",
        line=dict(color="#1f77b4", width=1.5),
        fill="tozeroy", fillcolor="rgba(31,119,180,0.08)"
    ))

    events = {
        "2008-07-01": ("Crise alim. 2008", "#e74c3c"),
        "2020-03-01": ("COVID-19",         "#f39c12"),
        "2022-02-01": ("Guerre Ukraine",   "#9b59b6"),
    }
    for date_str, (label, color) in events.items():
        d = pd.to_datetime(date_str)
        if hist_df["date"].min() <= d <= hist_df["date"].max():
            fig_hist.add_vline(x=d, line_dash="dot", line_color=color, opacity=0.5)
            fig_hist.add_annotation(x=d, y=hist_df["price"].max(),
                                    text=label, showarrow=False,
                                    font=dict(size=9, color=color), yshift=5)

    fig_hist.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Date", yaxis_title="Prix (XOF/kg)",
        legend=dict(orientation="h", y=1.1),
        hovermode="x unified"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    if predict_btn:
        with st.spinner(f"Prévision LSTM à +{horizon} mois en cours..."):
            try:
                current_price = float(hist_df["price"].iloc[-1])
                predicted_price = predict_lstm(commodity, market, horizon)
                change_pct = (predicted_price - current_price) / current_price * 100
            except Exception as e:
                st.error(f"Erreur lors du calcul de la prévision : {e}")
                st.stop()

        if change_pct > 5:
            signal, emoji, css_class = "Prix en HAUSSE — Acheter / Stocker maintenant", "📈", "signal-up"
        elif change_pct < -5:
            signal, emoji, css_class = "Prix en BAISSE — Attendre pour acheter", "📉", "signal-down"
        else:
            signal, emoji, css_class = "Prix STABLE — Pas de signal fort", "➡️", "signal-flat"

        st.markdown(f'<div class="signal-card {css_class}">{emoji} {signal}</div>', unsafe_allow_html=True)

        # Boites métriques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-box"><div class="metric-label">Prix actuel</div><div class="metric-value">{round(current_price, 1):,.0f}</div><div class="metric-sub">XOF / kg</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-box"><div class="metric-label">Prix prédit (+{horizon} mois)</div><div class="metric-value">{round(predicted_price, 1):,.0f}</div><div class="metric-sub">XOF / kg</div></div>', unsafe_allow_html=True)
        with col3:
            color_val = "#155724" if change_pct > 5 else ("#721c24" if change_pct < -5 else "#856404")
            st.markdown(f'<div class="metric-box"><div class="metric-label">Variation attendue</div><div class="metric-value" style="color:{color_val}">{change_pct:+.2f}%</div><div class="metric-sub">par rapport au prix actuel</div></div>', unsafe_allow_html=True)

        st.divider()

        # Graphique d'horizon temporel futur
        hist_full = hist_df.tail(60).copy()
        last_date = hist_full["date"].max()
        pred_date = last_date + pd.DateOffset(months=horizon)

        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=hist_full["date"], y=hist_full["price"], mode="lines", name="Historique", line=dict(color="#1f77b4", width=2)))
        fig_pred.add_trace(go.Scatter(
            x=[last_date, pred_date], y=[current_price, predicted_price],
            mode="lines+markers", name=f"Prévision +{horizon} mois",
            line=dict(color="#E05A2B", width=2, dash="dash"),
            marker=dict(size=[6, 14], symbol=["circle", "diamond"], color=["#E05A2B", "#E05A2B"])
        ))
        fig_pred.add_annotation(x=pred_date, y=predicted_price, text=f"  {round(predicted_price, 1):,.0f} XOF", showarrow=False, font=dict(size=12, color="#E05A2B"), xanchor="left")
        fig_pred.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="Date", yaxis_title="Prix (XOF/kg)", legend=dict(orientation="h", y=1.1), hovermode="x unified")
        st.plotly_chart(fig_pred, use_container_width=True)

        # Prévisions multi-horizons interactives
        st.subheader("📅 Prévisions multi-horizons")
        cols = st.columns(3)
        for i, h in enumerate(HORIZONS):
            with cols[i]:
                try:
                    p_val = predict_lstm(commodity, market, h)
                    c_val = (p_val - current_price) / current_price * 100
                    emoji_h = "📈" if c_val > 5 else ("📉" if c_val < -5 else "➡️")
                    st.metric(label=f"+{h} mois", value=f"{round(p_val, 1):,.0f} XOF", delta=f"{c_val:+.1f}%")
                    st.caption(f"{emoji_h} Calculé avec succès")
                except Exception:
                    st.warning(f"Prévision +{h} mois indisponible")
    else:
        st.info("👈 Cliquez sur **Lancer la prévision** pour obtenir une prévision LSTM.")


# ════════════════════════════════════════════════════════════
# ONGLETS 2 — Tableau de bord marché (Batch)
# ════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader(f"📊 Tous les produits — {market} — +{horizon} mois")

    if batch_btn or st.button("Charger le tableau", key="load_batch"):
        with st.spinner(f"Calcul des prévisions pour tous les produits de {market}..."):
            commodities_market = DF_RAW[DF_RAW["market"] == market]["commodity"].unique()
            results = []

            for c in commodities_market:
                try:
                    sub_c = DF_RAW[(DF_RAW["commodity"] == c) & (DF_RAW["market"] == market)].sort_values("date")
                    c_price = float(sub_c["price"].iloc[-1])
                    p_price = predict_lstm(c, market, horizon)
                    pct = (p_price - c_price) / c_price * 100

                    if pct > 5:
                        sig, em = "Acheter / Stocker", "📈"
                    elif pct < -5:
                        sig, em = "Attendre", "📉"
                    else:
                        sig, em = "Stable", "➡️"

                    results.append({
                        "commodity":       c,
                        "current_price":   round(c_price, 1),
                        "predicted_price": round(p_price, 1),
                        "change_pct":      round(pct, 2),
                        "signal":          sig,
                        "signal_emoji":    em,
                    })
                except Exception:
                    continue  # Saute le produit si l'historique est insuffisant

            if not results:
                st.warning("Aucun résultat disponible pour ce marché.")
                st.stop()

            df_batch = pd.DataFrame(results).sort_values("change_pct", ascending=False)

        # Compteurs KPIs
        c1, c2, c3 = st.columns(3)
        n_up   = (df_batch["change_pct"] > 5).sum()
        n_down = (df_batch["change_pct"] < -5).sum()
        n_flat = len(df_batch) - n_up - n_down
        c1.metric("📈 Produits en hausse", n_up)
        c2.metric("📉 Produits en baisse", n_down)
        c3.metric("➡️  Produits stables",  n_flat)

        st.divider()

        # Graphique à barres horizontal
        df_sorted = df_batch.sort_values("change_pct", ascending=True)
        colors = ["#155724" if v > 5 else ("#721c24" if v < -5 else "#856404") for v in df_sorted["change_pct"]]

        fig_bar = go.Figure(go.Bar(
            x=df_sorted["change_pct"], y=df_sorted["commodity"],
            orientation="h", marker_color=colors,
            text=[f"{v:+.1f}%" for v in df_sorted["change_pct"]], textposition="outside",
        ))
        fig_bar.update_layout(height=max(400, len(df_batch) * 28), margin=dict(l=0, r=60, t=10, b=0), xaxis_title="Variation attendue (%)", yaxis_title="", xaxis=dict(zeroline=True, zerolinecolor="#aaa", zerolinewidth=1.5))
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # Tableau final mis en forme
        df_display = df_batch[[
            "signal_emoji", "commodity", "current_price", "predicted_price", "change_pct", "signal"
        ]].rename(columns={
            "signal_emoji": "Signal", "commodity": "Produit", "current_price": "Prix actuel (XOF)",
            "predicted_price": "Prix prédit (XOF)", "change_pct": "Variation (%)", "signal": "Recommandation",
        })

        def color_change(val):
            if val > 5: return "color: #155724; font-weight: bold"
            elif val < -5: return "color: #721c24; font-weight: bold"
            return "color: #856404"

        st.dataframe(df_display.style.applymap(color_change, subset=["Variation (%)"]), use_container_width=True, height=min(600, len(df_display) * 38 + 40))

        csv = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(label="⬇️ Télécharger en CSV", data=csv, file_name=f"previsions_{market}_h{horizon}.csv", mime="text/csv")
    else:
        st.info("👈 Cliquez sur **Voir tous les produits** dans la barre latérale.")
