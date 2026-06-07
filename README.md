# 🌍 Déploiement sur Streamlit Cloud

## Structure du repo GitHub

```
mon-repo/
├── app.py                        ← application Streamlit
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── secrets.toml              ← NE PAS committer (dans .gitignore)
├── models/
│   ├── metadata.pkl              ← léger, peut aller sur GitHub
│   ├── scaler_lstm_features.pkl  ← léger
│   ├── lstm_h1_scaler.pkl        ← léger
│   ├── lstm_h3_scaler.pkl        ← léger
│   ├── lstm_h6_scaler.pkl        ← léger
│   ├── lstm_h1.keras             ← LOURD → Hugging Face
│   ├── lstm_h3.keras             ← LOURD → Hugging Face
│   └── lstm_h6.keras             ← LOURD → Hugging Face
└── wfp_food_prices_sen.csv       ← données (~20MB)
```

---

## Étape 1 — Héberger les modèles LSTM sur Hugging Face

Les fichiers `.keras` sont trop lourds pour GitHub (limite 100MB).

```bash
# 1. Installer le CLI Hugging Face
pip install huggingface_hub

# 2. Se connecter
huggingface-cli login   # entrer votre token depuis huggingface.co/settings/tokens

# 3. Créer un repo Dataset sur huggingface.co/new-dataset
#    Nom suggéré : senegal-food-prices
#    Visibilité  : Public

# 4. Uploader les fichiers modèles
python - <<'EOF'
from huggingface_hub import HfApi

api    = HfApi()
repo   = "VOTRE_USERNAME/senegal-food-prices"   # ← modifier
folder = "./models"

import os
for fname in os.listdir(folder):
    api.upload_file(
        path_or_fileobj=f"{folder}/{fname}",
        path_in_repo=fname,
        repo_id=repo,
        repo_type="dataset"
    )
    print(f"Uploadé : {fname}")
EOF
```

---

## Étape 2 — Préparer le repo GitHub

```bash
# 1. Créer le repo GitHub (github.com/new)

# 2. Initialiser et pousser
git init
git add app.py requirements.txt .gitignore README.md
git add models/metadata.pkl models/scaler_lstm_features.pkl
git add models/lstm_h1_scaler.pkl models/lstm_h3_scaler.pkl models/lstm_h6_scaler.pkl
# NE PAS ajouter les .keras ni secrets.toml
git add wfp_food_prices_sen.csv

git commit -m "Initial commit — Senegal Food Price Forecasting"
git remote add origin https://github.com/VOTRE_USERNAME/VOTRE_REPO.git
git push -u origin main
```

---

## Étape 3 — Déployer sur Streamlit Cloud

1. Aller sur **[share.streamlit.io](https://share.streamlit.io)**
2. Se connecter avec GitHub
3. Cliquer **New app**
4. Sélectionner votre repo → branche `main` → fichier `app.py`
5. Cliquer **Advanced settings** → **Secrets** et coller :

```toml
HF_REPO  = "VOTRE_USERNAME/senegal-food-prices"
DATA_URL = "https://raw.githubusercontent.com/VOTRE_USERNAME/VOTRE_REPO/main/wfp_food_prices_sen.csv"
```

6. Cliquer **Deploy** ✅

---

## Étape 4 — Vérifier

- L'app est disponible sur : `https://VOTRE_USERNAME-VOTRE_REPO-app.streamlit.app`
- Premier chargement : ~2-3 min (téléchargement des modèles depuis HF)
- Chargements suivants : rapides (cache Streamlit)

---

## Résolution de problèmes

| Problème | Solution |
|---|---|
| `ModuleNotFoundError` | Vérifier `requirements.txt` |
| `Model not found` | Vérifier `HF_REPO` dans Secrets |
| App lente au démarrage | Normal — les modèles sont en cache après le 1er chargement |
| `Memory Error` | Utiliser `tensorflow-cpu` (déjà dans requirements.txt) |
| CSV non trouvé | Vérifier `DATA_URL` dans Secrets |
