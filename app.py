"""
Senegal Food Price Forecasting — Streamlit Cloud (standalone)
Logique LSTM intégrée directement, sans FastAPI.

Déploiement : share.streamlit.io
"""

import os
import io
import pickle
import warnings
import requests
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─────────────────────────────────────────────────────────────
# Configuration page
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Marchés Sénégal — Prévision des Prix",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .signal-card {
        border-radius: 12px; padding: 1.25rem 1.5rem;
        margin: 0.5rem 0; font-size: 1.1rem;
        font-weight: 600; text-align: center;
    }
    .signal-up   { background:#d4edda; color:#155724; border:1px solid #c3e6cb; }
    .signal-down { background:#f8d7da; color:#721c24; border:1px solid #f5c6cb; }
    .signal-flat { background:#fff3cd; color:#856404; border:1px solid #ffeeba; }
    .metric-box  {
        background:#f8f9fa; border-radius:10px; padding:1rem;
        text-align:center; border:1px solid #e9ecef;
    }
    .metric-label { font-size:0.8rem; color:#6c757d; margin-bottom:4px; }
    .metric-value { font-size:1.6rem; font-weight:700; color:#212529; }
    .metric-sub   { font-size:0.85rem; color:#6c757d; }
</style>
""", unsafe_allow_html=True)

HORIZONS    = [1, 3, 6]
WINDOW_SIZE = 24

# ─────────────────────────────────────────────────────────────
# Chargement des artefacts
# ─────────────────────────────────────────────────────────────
# Les modèles peuvent venir de 3 sources selon l'environnement :
#   1. Dossier local ./models/ (dev local)
#   2. Hugging Face Hub      (cloud recommandé)
#   3. URL directe           (Google Drive public, etc.)
#
# Configure la source dans st.secrets ou les variables d'env.
# ─────────────────────────────────────────────────────────────

def load_pickle_from_url(url: str):
    """Télécharge et désérialise un fichier pickle depuis une URL."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return pickle.loads(r.content)


def load_keras_from_url(url: str):
    """Télécharge un modèle .keras depuis une URL et le charge en mémoire."""
    from tensorflow.keras.models import load_model as keras_load
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    tmp_path = f"/tmp/lstm_model_{hash(url)}.keras"
    with open(tmp_path, "wb") as f:
        f.write(r.content)
    return keras_load(tmp_path)


@st.cache_resource(show_spinner="Chargement des modèles LSTM...")
def load_all_artifacts():
    """
    Charge tous les artefacts.
    Priorité : local → Hugging Face → URL custom.

    Pour Hugging Face, configure dans .streamlit/secrets.toml :
        HF_REPO = "ton-username/senegal-food-prices"

    Pour URL custom (Google Drive) :
        MODEL_BASE_URL = "https://..."
    """
    models_dir = Path("models")

    # ── Source 1 : local (dev / si modèles dans le repo) ──
    if (models_dir / "metadata.pkl").exists():
        st.toast("Modèles chargés depuis le dossier local ✓", icon="✅")
        from tensorflow.keras.models import load_model as keras_load

        with open(models_dir / "metadata.pkl", "rb") as f:
            meta = pickle.load(f)
        with open(models_dir / "scaler_lstm_features.pkl", "rb") as f:
            feat_scaler = pickle.load(f)

        lstm_models, y_scalers = {}, {}
        for h in HORIZONS:
            lstm_models[h] = keras_load(models_dir / f"lstm_h{h}.keras")
            with open(models_dir / f"lstm_h{h}_scaler.pkl", "rb") as f:
                y_scalers[h] = pickle.load(f)
        return meta, feat_scaler, lstm_models, y_scalers

    # ── Source 2 : Hugging Face Hub ───────────────────────
    hf_repo = st.secrets.get("HF_REPO", os.getenv("HF_REPO", ""))
    if hf_repo:
        try:
            from huggingface_hub import hf_hub_download
            from tensorflow.keras.models import load_model as keras_load

            def hf_pkl(filename):
                path = hf_hub_download(repo_id=hf_repo, filename=filename)
                with open(path, "rb") as f:
                    return pickle.load(f)

            meta        = hf_pkl("metadata.pkl")
            feat_scaler = hf_pkl("scaler_lstm_features.pkl")

            lstm_models, y_scalers = {}, {}
            for h in HORIZONS:
                model_path   = hf_hub_download(repo_id=hf_repo, filename=f"lstm_h{h}.keras")
                lstm_models[h] = keras_load(model_path)
                y_scalers[h]   = hf_pkl(f"lstm_h{h}_scaler.pkl")

            st.toast("Modèles chargés depuis Hugging Face ✓", icon="🤗")
            return meta, feat_scaler, lstm_models, y_scalers

        except Exception as e:
            st.warning(f"Hugging Face indisponible : {e}")

    # ── Source 3 : URL custom (Google Drive / S3 / etc.) ──
    base_url = st.secrets.get("MODEL_BASE_URL", os.getenv("MODEL_BASE_URL", ""))
    if base_url:
        try:
            meta        = load_pickle_from_url(f"{base_url}/metadata.pkl")
            feat_scaler = load_pickle_from_url(f"{base_url}/scaler_lstm_features.pkl")

            lstm_models, y_scalers = {}, {}
            for h in HORIZONS:
                lstm_models[h] = load_keras_from_url(f"{base_url}/lstm_h{h}.keras")
                y_scalers[h]   = load_pickle_from_url(f"{base_url}/lstm_h{h}_scaler.pkl")

            st.toast("Modèles chargés depuis URL ✓", icon="🌐")
            return meta, feat_scaler, lstm_models, y_scalers

        except Exception as e:
            st.warning(f"URL distante indisponible : {e}")

    st.error("""
    ❌ **Aucun modèle trouvé.**

    Configurez une des sources suivantes dans `.streamlit/secrets.toml` :

    ```toml
    # Option A — Hugging Face Hub (recommandé)
    HF_REPO = "votre-username/senegal-food-prices"

    # Option B — URL directe
    MODEL_BASE_URL = "https://votre-serveur.com/models"
    ```
    """)
    st.stop()


@st.cache_data(show_spinner="Chargement des données...")
def load_data():
    """Charge le CSV — local ou depuis GitHub raw."""
    local = Path("wfp_food_prices_sen.csv")
    if local.exists():
        df = pd.read_csv(local)
    else:
        # Fallback : GitHub raw (adapter l'URL à ton repo)
        raw_url = st.secrets.get(
            "DATA_URL",
            os.getenv("DATA_URL",
                      "https://raw.githubusercontent.com/TON_USERNAME/TON_REPO/main/wfp_food_prices_sen.csv")
        )
        df = pd.read_csv(raw_url)

    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["commodity", "market", "date"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# Feature engineering (identique au notebook)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Calcul des features...")
def build_features(_df: pd.DataFrame) -> pd.DataFrame:
    d = _df.copy().sort_values(["commodity", "market", "date"])
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
            lambda x, w=w: x.shift(1).rolling(w, min_periods=max(2, w//2)).mean())
        d[f"roll{w}_std"] = grp.transform(
            lambda x, w=w: x.shift(1).rolling(w, min_periods=max(2, w//2)).std())

    d["mom_1"] = grp.transform(lambda x: x.shift(1).pct_change(1))
    d["mom_3"] = grp.transform(lambda x: x.shift(1).pct_change(3))
    d["yoy"]   = grp.transform(lambda x: x.shift(1).pct_change(12))

    d["commodity_id_enc"] = d.groupby("commodity").ngroup()
    d["market_id_enc"]    = d.groupby("market").ngroup()
    d["category_id_enc"]  = d.groupby("category").ngroup()
    d["region_id_enc"]    = d.groupby("admin1").ngroup()

    return d


# ─────────────────────────────────────────────────────────────
# Prédiction LSTM
# ─────────────────────────────────────────────────────────────
def predict_lstm(commodity: str, market: str, horizon: int,
                 df_feat: pd.DataFrame, meta: dict,
                 feat_scaler, lstm_models: dict, y_scalers: dict) -> float:

    series = df_feat[
        (df_feat["commodity"] == commodity) &
        (df_feat["market"]    == market)
    ].sort_values("date")

    if len(series) < WINDOW_SIZE:
        raise ValueError(
            f"Historique insuffisant : {len(series)} observations "
            f"(minimum {WINDOW_SIZE})"
        )

    feat_vals   = series[meta["lstm_features"]].tail(WINDOW_SIZE).ffill().bfill().values
    feat_scaled = feat_scaler.transform(feat_vals)
    X_in        = feat_scaled[np.newaxis, :, :].astype(np.float32)

    pred_scaled = lstm_models[horizon].predict(X_in, verbose=0)[0][0]
    pred_price  = y_scalers[horizon].inverse_transform([[pred_scaled]])[0][0]

    return float(max(pred_price, 0))


def predict_successive(commodity: str, market: str, n_months: int,
                       df_feat: pd.DataFrame, meta: dict,
                       feat_scaler, lstm_models: dict, y_scalers: dict) -> list:
    """
    Prévisions successives mois par mois via le modèle h=1.
    Retourne une liste de (date, prix_prédit).
    """
    series = df_feat[
        (df_feat["commodity"] == commodity) &
        (df_feat["market"]    == market)
    ].sort_values("date")

    last_date   = series["date"].iloc[-1]
    feat_window = series[meta["lstm_features"]].tail(WINDOW_SIZE).ffill().bfill().values.copy()

    predictions = []
    for i in range(1, n_months + 1):
        feat_scaled = feat_scaler.transform(feat_window)
        X_in        = feat_scaled[np.newaxis, :, :].astype(np.float32)

        pred_scaled = lstm_models[1].predict(X_in, verbose=0)[0][0]
        pred_price  = float(max(
            y_scalers[1].inverse_transform([[pred_scaled]])[0][0], 0
        ))

        pred_date = last_date + pd.DateOffset(months=i)
        predictions.append({"date": pred_date, "price": round(pred_price, 1),
                             "type": "predicted"})

        # Glisser la fenêtre : on ajoute la prédiction comme nouvelle observation
        new_row         = feat_window[-1].copy()
        lag_1_idx       = meta["lstm_features"].index("lag_1")
        new_row[lag_1_idx] = pred_price   # lag_1 = prix prédit devient l'obs suivante
        feat_window     = np.vstack([feat_window[1:], new_row])

    return predictions


# ─────────────────────────────────────────────────────────────
# Chargement initial
# ─────────────────────────────────────────────────────────────
META, FEAT_SCALER, LSTM_MODELS, Y_SCALERS = load_all_artifacts()
DF_RAW  = load_data()
DF_FEAT = build_features(DF_RAW)

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://flagcdn.com/w80/sn.png", width=60)
    st.title("🌍 Marchés Sénégal")
    st.caption("Prévision des prix alimentaires — LSTM")
    st.divider()

    markets     = sorted(DF_RAW["market"].unique())
    market      = st.selectbox("📍 Marché", markets,
                                index=markets.index("Thies") if "Thies" in markets else 0)

    commodities = sorted(DF_RAW[DF_RAW["market"] == market]["commodity"].unique())
    commodity   = st.selectbox("🛒 Produit", commodities)

    st.divider()
    st.subheader("Mode de prévision")

    mode = st.radio(
        "Type",
        ["Point unique (h=1, 3 ou 6 mois)", "Successive (mois par mois)"],
        label_visibility="collapsed"
    )

    if mode == "Point unique (h=1, 3 ou 6 mois)":
        horizon = st.select_slider(
            "Horizon", options=HORIZONS, value=3,
            format_func=lambda x: f"+{x} mois"
        )
        n_months = None
    else:
        n_months = st.slider("Nombre de mois à prévoir", 1, 12, 6)
        horizon  = None

    st.divider()
    predict_btn = st.button("🔮 Lancer la prévision", type="primary",
                             use_container_width=True)
    batch_btn   = st.button("📊 Tous les produits", use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Page principale
# ─────────────────────────────────────────────────────────────
st.title("Prévision des Prix Alimentaires — Sénégal")
st.caption(f"Données WFP 2000–2024 · Modèle LSTM · {datetime.now().strftime('%d/%m/%Y %H:%M')}")

tab_pred, tab_batch, tab_about = st.tabs([
    "🔮 Prévision produit",
    "📊 Tableau de bord marché",
    "ℹ️ À propos"
])

# ════════════════════════════════════════════════════════════
# TAB 1 — Prévision produit
# ════════════════════════════════════════════════════════════
with tab_pred:
    sub = DF_RAW[(DF_RAW["commodity"] == commodity) & (DF_RAW["market"] == market)]
    hist_df = sub[["date", "price"]].sort_values("date")

    st.subheader(f"{commodity} — {market}")
    st.caption(f"{len(hist_df)} observations · {hist_df['date'].min().year}–{hist_df['date'].max().year}")

    # ── Graphique historique complet ──────────────────────────
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=hist_df["date"], y=hist_df["price"],
        mode="lines", name="Prix historique",
        line=dict(color="#1f77b4", width=1.5),
        fill="tozeroy", fillcolor="rgba(31,119,180,0.08)"
    ))
    for date_str, (label, color) in {
        "2008-07-01": ("Crise 2008",   "#e74c3c"),
        "2020-03-01": ("COVID-19",     "#f39c12"),
        "2022-02-01": ("Guerre Ukraine", "#9b59b6"),
    }.items():
        d = pd.to_datetime(date_str)
        if hist_df["date"].min() <= d <= hist_df["date"].max():
            fig_hist.add_vline(x=d, line_dash="dot", line_color=color, opacity=0.5)
            fig_hist.add_annotation(x=d, y=hist_df["price"].max(),
                                    text=label, showarrow=False,
                                    font=dict(size=9, color=color), yshift=5)
    fig_hist.update_layout(
        height=280, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Date", yaxis_title="Prix (XOF/kg)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.divider()

    # ── Prévision ─────────────────────────────────────────────
    if predict_btn:

        # ── Mode : Point unique ───────────────────────────────
        if mode == "Point unique (h=1, 3 ou 6 mois)":
            with st.spinner(f"Prévision LSTM à +{horizon} mois..."):
                try:
                    current_price = float(hist_df["price"].iloc[-1])
                    pred_price    = predict_lstm(
                        commodity, market, horizon,
                        DF_FEAT, META, FEAT_SCALER, LSTM_MODELS, Y_SCALERS
                    )
                    change_pct = (pred_price - current_price) / current_price * 100

                    if change_pct > 5:
                        css, emoji, signal = "signal-up",   "📈", "Prix en HAUSSE — Acheter / Stocker maintenant"
                    elif change_pct < -5:
                        css, emoji, signal = "signal-down", "📉", "Prix en BAISSE — Attendre pour acheter"
                    else:
                        css, emoji, signal = "signal-flat", "➡️", "Prix STABLE — Pas de signal fort"

                    st.markdown(
                        f'<div class="signal-card {css}">{emoji} {signal}</div>',
                        unsafe_allow_html=True
                    )

                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f'<div class="metric-box"><div class="metric-label">Prix actuel</div>'
                                f'<div class="metric-value">{current_price:,.0f}</div>'
                                f'<div class="metric-sub">XOF / kg</div></div>', unsafe_allow_html=True)
                    c2.markdown(f'<div class="metric-box"><div class="metric-label">Prix prédit (+{horizon} mois)</div>'
                                f'<div class="metric-value">{pred_price:,.0f}</div>'
                                f'<div class="metric-sub">XOF / kg</div></div>', unsafe_allow_html=True)
                    color_val = "#155724" if change_pct > 0 else ("#721c24" if change_pct < 0 else "#856404")
                    c3.markdown(f'<div class="metric-box"><div class="metric-label">Variation attendue</div>'
                                f'<div class="metric-value" style="color:{color_val}">{change_pct:+.1f}%</div>'
                                f'<div class="metric-sub">vs prix actuel</div></div>', unsafe_allow_html=True)

                    st.divider()

                    # Graphique historique + point de prévision
                    last_60 = hist_df.tail(60)
                    last_date = hist_df["date"].iloc[-1]
                    pred_date = last_date + pd.DateOffset(months=horizon)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=last_60["date"], y=last_60["price"],
                        mode="lines", name="Historique (5 ans)",
                        line=dict(color="#1f77b4", width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[last_date, pred_date],
                        y=[current_price, pred_price],
                        mode="lines+markers", name=f"Prévision +{horizon} mois",
                        line=dict(color="#E05A2B", width=2, dash="dash"),
                        marker=dict(size=[6, 14], color="#E05A2B",
                                    symbol=["circle", "diamond"])
                    ))
                    fig.add_annotation(
                        x=pred_date, y=pred_price,
                        text=f"  {pred_price:,.0f} XOF",
                        showarrow=False, font=dict(size=12, color="#E05A2B"),
                        xanchor="left"
                    )
                    fig.update_layout(
                        height=350, margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title="Date", yaxis_title="Prix (XOF/kg)",
                        hovermode="x unified",
                        legend=dict(orientation="h", y=1.1)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Multi-horizons
                    st.subheader("📅 Comparaison multi-horizons")
                    cols = st.columns(3)
                    for i, h in enumerate(HORIZONS):
                        with cols[i]:
                            p = predict_lstm(commodity, market, h,
                                             DF_FEAT, META, FEAT_SCALER, LSTM_MODELS, Y_SCALERS)
                            c = (p - current_price) / current_price * 100
                            st.metric(
                                label=f"+{h} mois",
                                value=f"{p:,.0f} XOF",
                                delta=f"{c:+.1f}%"
                            )

                except Exception as e:
                    st.error(f"Erreur : {e}")

        # ── Mode : Successive ─────────────────────────────────
        else:
            with st.spinner(f"Prévisions successives sur {n_months} mois..."):
                try:
                    current_price = float(hist_df["price"].iloc[-1])
                    last_date     = hist_df["date"].iloc[-1]

                    preds = predict_successive(
                        commodity, market, n_months,
                        DF_FEAT, META, FEAT_SCALER, LSTM_MODELS, Y_SCALERS
                    )
                    preds_df = pd.DataFrame(preds)

                    # Signal global basé sur la dernière prévision
                    final_price = preds_df["price"].iloc[-1]
                    change_pct  = (final_price - current_price) / current_price * 100

                    if change_pct > 5:
                        css, emoji, signal = "signal-up",   "📈", f"Tendance HAUSSE sur {n_months} mois — Stocker maintenant"
                    elif change_pct < -5:
                        css, emoji, signal = "signal-down", "📉", f"Tendance BAISSE sur {n_months} mois — Attendre pour acheter"
                    else:
                        css, emoji, signal = "signal-flat", "➡️", f"Prix STABLE sur {n_months} mois"

                    st.markdown(
                        f'<div class="signal-card {css}">{emoji} {signal}</div>',
                        unsafe_allow_html=True
                    )

                    # ── Graphique historique + courbe de prévision ────
                    last_60 = hist_df.tail(60)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=last_60["date"], y=last_60["price"],
                        mode="lines", name="Historique",
                        line=dict(color="#1f77b4", width=2)
                    ))

                    # Connecter le dernier point historique à la prévision
                    x_pred = [last_date] + preds_df["date"].tolist()
                    y_pred = [current_price] + preds_df["price"].tolist()

                    fig.add_trace(go.Scatter(
                        x=x_pred, y=y_pred,
                        mode="lines+markers",
                        name=f"Prévision ({n_months} mois successifs)",
                        line=dict(color="#E05A2B", width=2.5, dash="dash"),
                        marker=dict(size=8, color="#E05A2B"),
                        hovertemplate="<b>%{x|%b %Y}</b><br>Prix prédit: %{y:,.0f} XOF<extra></extra>"
                    ))

                    # Zone de prévision
                    fig.add_vrect(
                        x0=last_date, x1=preds_df["date"].iloc[-1],
                        fillcolor="#E05A2B", opacity=0.05,
                        annotation_text="Période prévue",
                        annotation_position="top left",
                        annotation_font_size=10
                    )

                    fig.update_layout(
                        height=400, margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title="Date", yaxis_title="Prix (XOF/kg)",
                        hovermode="x unified",
                        legend=dict(orientation="h", y=1.1)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # ── Tableau des prévisions ─────────────────────────
                    st.subheader("📋 Prévisions mois par mois")
                    display_df = preds_df.copy()
                    display_df["date"] = display_df["date"].dt.strftime("%B %Y")
                    display_df["variation"] = (
                        (display_df["price"] - current_price) / current_price * 100
                    ).round(1)
                    display_df["signal"] = display_df["variation"].apply(
                        lambda v: "📈 Stocker" if v > 5 else ("📉 Attendre" if v < -5 else "➡️ Stable")
                    )
                    display_df.columns = ["Mois", "Prix prédit (XOF)", "Type", "Variation (%)", "Signal"]
                    display_df = display_df.drop(columns=["Type"])

                    def color_var(val):
                        if val > 5:   return "color: #155724; font-weight:bold"
                        if val < -5:  return "color: #721c24; font-weight:bold"
                        return "color: #856404"

                    st.dataframe(
                        display_df.style.applymap(color_var, subset=["Variation (%)"]),
                        use_container_width=True, hide_index=True
                    )

                    # Téléchargement
                    csv = display_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Télécharger les prévisions (CSV)", csv,
                        f"previsions_{commodity}_{market}_{n_months}mois.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Erreur : {e}")

    else:
        st.info("👈 Configurez vos paramètres et cliquez sur **Lancer la prévision**.")


# ════════════════════════════════════════════════════════════
# TAB 2 — Tableau de bord marché
# ════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader(f"Tous les produits — {market}")

    h_batch = st.select_slider(
        "Horizon", options=HORIZONS, value=3,
        format_func=lambda x: f"+{x} mois", key="batch_horizon"
    )

    if batch_btn or st.button("Charger", key="load_batch"):
        results = []
        bar = st.progress(0, text="Calcul en cours...")
        prods = sorted(DF_RAW[DF_RAW["market"] == market]["commodity"].unique())

        for i, prod in enumerate(prods):
            try:
                cur = float(DF_RAW[(DF_RAW["commodity"] == prod) &
                                    (DF_RAW["market"] == market)]["price"].iloc[-1])
                pred = predict_lstm(prod, market, h_batch,
                                    DF_FEAT, META, FEAT_SCALER, LSTM_MODELS, Y_SCALERS)
                chg  = (pred - cur) / cur * 100
                results.append({
                    "signal":    "📈" if chg > 5 else ("📉" if chg < -5 else "➡️"),
                    "commodity": prod,
                    "current":   cur,
                    "predicted": pred,
                    "change":    chg,
                    "reco":      "Stocker" if chg > 5 else ("Attendre" if chg < -5 else "Stable"),
                })
            except Exception:
                pass
            bar.progress((i + 1) / len(prods), text=f"{prod}...")

        bar.empty()

        if not results:
            st.warning("Aucun résultat disponible.")
        else:
            df_b = pd.DataFrame(results).sort_values("change", ascending=False)

            c1, c2, c3 = st.columns(3)
            c1.metric("📈 En hausse",  (df_b["change"] >  5).sum())
            c2.metric("📉 En baisse",  (df_b["change"] < -5).sum())
            c3.metric("➡️  Stable",    ((df_b["change"] >= -5) & (df_b["change"] <= 5)).sum())

            # Barres horizontales
            colors = ["#155724" if v > 5 else ("#721c24" if v < -5 else "#856404")
                      for v in df_b["change"]]
            fig_bar = go.Figure(go.Bar(
                x=df_b["change"], y=df_b["commodity"], orientation="h",
                marker_color=colors,
                text=[f"{v:+.1f}%" for v in df_b["change"]],
                textposition="outside"
            ))
            fig_bar.update_layout(
                height=max(400, len(df_b) * 28),
                margin=dict(l=0, r=60, t=10, b=0),
                xaxis_title="Variation attendue (%)", yaxis_title="",
                xaxis=dict(zeroline=True, zerolinecolor="#aaa", zerolinewidth=1.5)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Tableau
            df_show = df_b.rename(columns={
                "signal": "", "commodity": "Produit",
                "current": "Prix actuel (XOF)", "predicted": "Prix prédit (XOF)",
                "change": "Variation (%)", "reco": "Recommandation"
            })
            st.dataframe(
                df_show.style.applymap(
                    lambda v: "color:#155724;font-weight:bold" if v > 5
                              else ("color:#721c24;font-weight:bold" if v < -5 else "color:#856404"),
                    subset=["Variation (%)"]
                ).format({"Prix actuel (XOF)": "{:,.0f}", "Prix prédit (XOF)": "{:,.0f}",
                           "Variation (%)": "{:+.1f}"}),
                use_container_width=True, hide_index=True
            )

            csv = df_show.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Télécharger CSV", csv,
                                f"dashboard_{market}_h{h_batch}.csv", mime="text/csv")
    else:
        st.info("👈 Cliquez sur **Tous les produits** dans la barre latérale.")


# ════════════════════════════════════════════════════════════
# TAB 3 — À propos
# ════════════════════════════════════════════════════════════
with tab_about:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 Objectif
        Prédire les prix alimentaires sur les marchés sénégalais
        à **1, 3 et 6 mois** (point unique) ou **mois par mois**
        (prévision successive) pour décider quand acheter, stocker ou vendre.

        ### 📦 Données
        - Source : **WFP Food Prices — Sénégal**
        - Période : **2000–2024**
        - **64 marchés**, **~30 produits**

        ### 🤖 Modèle LSTM
        | Paramètre | Valeur |
        |---|---|
        | Architecture | 2 couches LSTM (128→64) |
        | Fenêtre d'entrée | 24 mois |
        | Horizons | +1, +3, +6 mois |
        | Loss | Huber |
        """)
    with col2:
        st.markdown("""
        ### 📖 Signaux de décision
        | Signal | Condition | Action |
        |---|---|---|
        | 📈 Hausse | Variation > +5% | Acheter / Stocker |
        | 📉 Baisse | Variation < -5% | Attendre |
        | ➡️ Stable | Entre -5% et +5% | Pas de signal |

        ### 🔁 Modes de prévision
        **Point unique** : le modèle prédit directement le prix
        à +1, +3 ou +6 mois — plus précis.

        **Successive** : le modèle prédit mois par mois en
        réutilisant chaque prédiction comme entrée — permet de
        voir la trajectoire complète mais les erreurs s'accumulent.

        > ⚠️ Ces prévisions sont un outil d'aide à la décision,
        > pas une garantie.
        """)
