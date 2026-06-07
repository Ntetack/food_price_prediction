"""
Streamlit Frontend — Senegal Food Price Forecasting
Appelle le backend FastAPI et affiche les prévisions.

Lancement :
    streamlit run app.py
"""

import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
API_URL  = "http://localhost:8000"
HORIZONS = [1, 3, 6]

st.set_page_config(
    page_title="Marchés Sénégal - Prévision des Prix",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS personnalisé
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Carte signal */
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

    /* Métrique */
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
# Helpers API
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_markets():
    try:
        r = requests.get(f"{API_URL}/meta/markets", timeout=5)
        return r.json()["markets"]
    except Exception:
        st.error("❌ Impossible de contacter l'API. Assurez-vous que le backend est démarré.")
        return []


@st.cache_data(ttl=3600)
def get_commodities(market: str):
    try:
        r = requests.get(f"{API_URL}/meta/commodities/{market}", timeout=5)
        return r.json()["commodities"]
    except Exception:
        return []


@st.cache_data(ttl=300)
def get_history(commodity: str, market: str):
    r = requests.get(f"{API_URL}/history/{commodity}/{market}", timeout=10)
    return r.json()


@st.cache_data(ttl=300)
def get_prediction(commodity: str, market: str, horizon: int):
    payload = {"commodity": commodity, "market": market, "horizon": horizon}
    r = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
    if r.status_code != 200:
        raise Exception(r.json().get("detail", "Erreur API"))
    return r.json()


@st.cache_data(ttl=300)
def get_batch(market: str, horizon: int):
    payload = {"market": market, "horizon": horizon}
    r = requests.post(f"{API_URL}/predict/batch", json=payload, timeout=60)
    if r.status_code != 200:
        raise Exception(r.json().get("detail", "Erreur API"))
    return r.json()


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://flagcdn.com/w80/sn.png", width=60)
    st.title("Marchés Sénégal")
    st.caption("Outil de prévision des prix alimentaires — Modèle LSTM")
    st.divider()

    markets = get_markets()
    if not markets:
        st.stop()

    market = st.selectbox("📍 Marché", markets,
                           index=markets.index("Thies") if "Thies" in markets else 0)

    commodities = get_commodities(market)
    commodity   = st.selectbox("🛒 Produit", commodities) if commodities else None

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
    st.caption(f"API : `{API_URL}`")
    st.caption(f"Modèle : LSTM (fenêtre 24 mois)")
    st.caption(f"Données : WFP 2000–2024")


# ─────────────────────────────────────────────────────────────
# Page principale
# ─────────────────────────────────────────────────────────────
st.title("Prévision des Prix Alimentaires — Sénégal")
st.caption(f"Dernière mise à jour : {datetime.now().strftime('%d/%m/%Y %H:%M')}")

# ── Onglets ───────────────────────────────────────────────────
tab_predict, tab_batch = st.tabs([
    "🔮 Prévision produit",
    "📊 Tableau de bord marché"
])


# ════════════════════════════════════════════════════════════
# TAB 1 — Prévision produit
# ════════════════════════════════════════════════════════════
with tab_predict:
    if not commodity:
        st.info("Sélectionnez un marché et un produit dans la barre latérale.")
        st.stop()

    # Toujours afficher l'historique
    with st.spinner("Chargement de l'historique..."):
        hist_data = get_history(commodity, market)

    hist_df = pd.DataFrame(hist_data["history"])
    hist_df["date"] = pd.to_datetime(hist_df["date"])

    st.subheader(f"{commodity} — {market}")
    st.caption(f"{hist_data['n_obs']} observations disponibles · 2000–2024")

    # ── Graphique historique ──────────────────────────────────
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=hist_df["date"], y=hist_df["price"],
        mode="lines", name="Prix historique",
        line=dict(color="#1f77b4", width=1.5),
        fill="tozeroy", fillcolor="rgba(31,119,180,0.08)"
    ))

    # Événements clés
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

    # ── Prévision ─────────────────────────────────────────────
    if predict_btn:
        with st.spinner(f"Prévision LSTM à +{horizon} mois en cours..."):
            try:
                pred = get_prediction(commodity, market, horizon)
            except Exception as e:
                st.error(f"Erreur : {e}")
                st.stop()

        # Signal
        emoji = pred["signal_emoji"]
        change = pred["change_pct"]
        if emoji == "📈":
            css_class = "signal-up"
        elif emoji == "📉":
            css_class = "signal-down"
        else:
            css_class = "signal-flat"

        st.markdown(
            f'<div class="signal-card {css_class}">{emoji} {pred["signal"]}</div>',
            unsafe_allow_html=True
        )

        # Métriques
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Prix actuel</div>
                <div class="metric-value">{pred['current_price']:,.0f}</div>
                <div class="metric-sub">XOF / kg</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Prix prédit ({pred['horizon']})</div>
                <div class="metric-value">{pred['predicted_price']:,.0f}</div>
                <div class="metric-sub">XOF / kg</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            color_val = "#155724" if change > 0 else ("#721c24" if change < 0 else "#856404")
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-label">Variation attendue</div>
                <div class="metric-value" style="color:{color_val}">{change:+.1f}%</div>
                <div class="metric-sub">par rapport au prix actuel</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Graphique historique + prévision ──────────────────
        hist_full = pd.DataFrame(pred["history"])
        hist_full["date"] = pd.to_datetime(hist_full["date"])
        last_date = hist_full["date"].max()

        # Date de prévision
        pred_date = last_date + pd.DateOffset(months=horizon)

        fig_pred = go.Figure()

        # Historique (5 dernières années)
        fig_pred.add_trace(go.Scatter(
            x=hist_full["date"], y=hist_full["price"],
            mode="lines", name="Historique",
            line=dict(color="#1f77b4", width=2)
        ))

        # Ligne de connexion vers la prévision
        fig_pred.add_trace(go.Scatter(
            x=[last_date, pred_date],
            y=[pred["current_price"], pred["predicted_price"]],
            mode="lines+markers",
            name=f"Prévision +{horizon} mois",
            line=dict(color="#E05A2B", width=2, dash="dash"),
            marker=dict(size=[6, 14], symbol=["circle", "diamond"],
                        color=["#E05A2B", "#E05A2B"])
        ))

        # Annotation point de prévision
        fig_pred.add_annotation(
            x=pred_date, y=pred["predicted_price"],
            text=f"  {pred['predicted_price']:,.0f} XOF",
            showarrow=False, font=dict(size=12, color="#E05A2B"),
            xanchor="left"
        )

        fig_pred.update_layout(
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Date", yaxis_title="Prix (XOF/kg)",
            legend=dict(orientation="h", y=1.1),
            hovermode="x unified"
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # ── Prévisions sur les 3 horizons ─────────────────────
        st.subheader("📅 Prévisions multi-horizons")
        cols = st.columns(3)
        for i, h in enumerate(HORIZONS):
            with cols[i]:
                try:
                    p = get_prediction(commodity, market, h)
                    c = p["change_pct"]
                    color = "green" if c > 5 else ("red" if c < -5 else "orange")
                    st.metric(
                        label=f"+{h} mois",
                        value=f"{p['predicted_price']:,.0f} XOF",
                        delta=f"{c:+.1f}%"
                    )
                    st.caption(p["signal_emoji"] + " " + p["signal"])
                except Exception:
                    st.warning(f"Prévision +{h} mois indisponible")

    else:
        st.info("👈 Cliquez sur **Lancer la prévision** pour obtenir une prévision LSTM.")


# ════════════════════════════════════════════════════════════
# TAB 2 — Tableau de bord marché
# ════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader(f"📊 Tous les produits — {market} — +{horizon} mois")

    if batch_btn or st.button("Charger le tableau", key="load_batch"):
        with st.spinner(f"Calcul des prévisions pour tous les produits de {market}..."):
            try:
                batch = get_batch(market, horizon)
            except Exception as e:
                st.error(f"Erreur : {e}")
                st.stop()

        preds = batch["predictions"]
        if not preds:
            st.warning("Aucun résultat disponible.")
            st.stop()

        df_batch = pd.DataFrame(preds)

        # KPIs résumés
        c1, c2, c3 = st.columns(3)
        n_up   = (df_batch["change_pct"] >  5).sum()
        n_down = (df_batch["change_pct"] < -5).sum()
        n_flat = len(df_batch) - n_up - n_down

        c1.metric("📈 Produits en hausse",  n_up)
        c2.metric("📉 Produits en baisse",  n_down)
        c3.metric("➡️  Produits stables",   n_flat)

        st.divider()

        # ── Graphique barres — variation attendue ─────────────
        df_sorted = df_batch.sort_values("change_pct", ascending=True)
        colors = [
            "#155724" if v > 5 else ("#721c24" if v < -5 else "#856404")
            for v in df_sorted["change_pct"]
        ]

        fig_bar = go.Figure(go.Bar(
            x=df_sorted["change_pct"],
            y=df_sorted["commodity"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in df_sorted["change_pct"]],
            textposition="outside",
        ))
        fig_bar.update_layout(
            height=max(400, len(df_batch) * 28),
            margin=dict(l=0, r=60, t=10, b=0),
            xaxis_title="Variation attendue (%)",
            yaxis_title="",
            xaxis=dict(zeroline=True, zerolinecolor="#aaa", zerolinewidth=1.5),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # ── Tableau détaillé ──────────────────────────────────
        st.subheader("Tableau détaillé")

        df_display = df_batch[[
            "signal_emoji", "commodity", "current_price",
            "predicted_price", "change_pct", "signal"
        ]].rename(columns={
            "signal_emoji":    "Signal",
            "commodity":       "Produit",
            "current_price":   "Prix actuel (XOF)",
            "predicted_price": "Prix prédit (XOF)",
            "change_pct":      "Variation (%)",
            "signal":          "Recommandation",
        })

        def color_change(val):
            if val > 5:
                return "color: #155724; font-weight: bold"
            elif val < -5:
                return "color: #721c24; font-weight: bold"
            return "color: #856404"

        st.dataframe(
            df_display.style.applymap(color_change, subset=["Variation (%)"]),
            use_container_width=True,
            height=min(600, len(df_display) * 38 + 40),
        )

        # Téléchargement CSV
        csv = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Télécharger en CSV",
            data=csv,
            file_name=f"previsions_{market}_h{horizon}.csv",
            mime="text/csv",
        )
    else:
        st.info("👈 Cliquez sur **Voir tous les produits** dans la barre latérale.")


