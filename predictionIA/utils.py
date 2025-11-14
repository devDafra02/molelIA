from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from django.conf import settings
import re

BASE = Path(settings.BASE_DIR)

# chemins vers fichiers sauvegardés
CLASS_MODEL_PATH = BASE / "model_class.pkl"
REG_MODEL_PATH   = BASE / "model_reg.pkl"
SCALER_PATH      = BASE / "scaler.pkl"
LABEL_ENCODER_PATH = BASE / "label_encoder.pkl"
FEATURES_PATH    = BASE / "features.pkl"

# chargement
MODEL_CLASS = joblib.load(str(CLASS_MODEL_PATH))
MODEL_REG   = joblib.load(str(REG_MODEL_PATH))
SCALER = joblib.load(str(SCALER_PATH))
LABEL_ENCODER = joblib.load(str(LABEL_ENCODER_PATH))
FEATURE_COLUMNS = list(joblib.load(str(FEATURES_PATH)))

# helpers
def sanitize_colname(s: str) -> str:
    return re.sub(r'[^0-9a-zA-Z_]', '_', str(s))

def sanitize_input_dict(input_dict: dict) -> dict:
    return { sanitize_colname(k): v for k,v in input_dict.items() }

def prepare_input(input_dict: dict):
    clean = sanitize_input_dict(input_dict)
    df = pd.DataFrame([clean])
    X_enc = pd.get_dummies(df, drop_first=True)
    X_aligned = X_enc.reindex(columns=FEATURE_COLUMNS, fill_value=0)
    return SCALER.transform(X_aligned.values)

def predict_from_dict(input_dict: dict):
    X = prepare_input(input_dict)

    # classification
    y_class = MODEL_CLASS.predict(X)[0]
    proba = None
    if hasattr(MODEL_CLASS, "predict_proba"):
        proba = float(np.max(MODEL_CLASS.predict_proba(X)[0]))
    etat_label = LABEL_ENCODER.inverse_transform([int(y_class)])[0]

    # régression
    duree = float(MODEL_REG.predict(X)[0])

    return {
        "etat_label": etat_label,
        "etat_proba": proba,
        "duree_restant": duree
    }
