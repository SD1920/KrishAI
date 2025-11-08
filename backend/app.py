# backend/app.py
import os
import math
import joblib
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -------------------------
# Paths & artifacts
# -------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ML_DIR = os.path.join(BASE_DIR, "ml_artifacts")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Load artifacts (assumes you placed the files as instructed)
MODEL = joblib.load(os.path.join(ML_DIR, "crop_yield_model.pkl"))
FEATURE_COLS: List[str] = joblib.load(os.path.join(ML_DIR, "feature_cols.pkl"))
ENCODERS: Dict[str, Any] = joblib.load(os.path.join(ML_DIR, "encoders.pkl"))

FERT = pd.read_csv(os.path.join(DATA_DIR, "fert_products_clean.csv"))
PEST = pd.read_csv(os.path.join(DATA_DIR, "pesticide_clean2.csv"))
MASTER = pd.read_csv(os.path.join(DATA_DIR, "merged_ready3.csv"))

# crop recommendation artifacts
CROP_MODEL = joblib.load(os.path.join(ML_DIR, "crop_recommendation_model.pkl"))
CROP_LABEL = joblib.load(os.path.join(ML_DIR, "crop_label_encoder.pkl"))
CROP_SCALER = joblib.load(os.path.join(ML_DIR, "crop_scaler.pkl"))

# Optional ensemble artifacts (commented out unless you created them)
# RF = joblib.load(os.path.join(ML_DIR, "crop_rf.pkl"))
# GB = joblib.load(os.path.join(ML_DIR, "crop_gb.pkl"))

# -------------------------
# FastAPI app + CORS
# -------------------------
app = FastAPI(title="Agri AI Backend", version="1.0")

origins = [
    "http://localhost:8001",
    "http://127.0.0.1:8001",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request models
# -------------------------
class Req(BaseModel):
    district: str
    crop: str
    area_ha: float = 1.0
    lat: Optional[float] = None
    lon: Optional[float] = None
    sow_date: Optional[str] = None
    soil_type: Optional[str] = None

class CropReq(BaseModel):
    N: Optional[float] = None
    P: Optional[float] = None
    K: Optional[float] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    ph: Optional[float] = None
    rainfall: Optional[float] = None

# -------------------------
# Helpers (yield model)
# -------------------------
def build_row(base: Dict[str, Any]) -> pd.DataFrame:
    Xref = MASTER.drop(columns=[c for c in MASTER.columns if 'yield' in c.lower()], errors='ignore')
    row: Dict[str, Any] = {}

    for col in FEATURE_COLS:
        if col in Xref.columns:
            if np.issubdtype(Xref[col].dtype, np.number):
                row[col] = float(Xref[col].median())
            else:
                row[col] = str(Xref[col].mode().iloc[0])
        else:
            row[col] = 0.0

    for k, v in base.items():
        for fc in FEATURE_COLS:
            if fc.lower().replace(" ", "") == k.lower().replace(" ", ""):
                row[fc] = v
                break

    df = pd.DataFrame([row], columns=FEATURE_COLS)

    for col, le in ENCODERS.items():
        if col in df.columns:
            val = df.at[0, col]
            try:
                df[col] = le.transform([str(val)])[0]
            except Exception:
                df[col] = le.transform([le.classes_[0]])[0]

    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass

    return df

def choose_product(district: str, target=(19, 19, 19)) -> Dict[str, Any]:
    d = str(district).lower()
    cand = FERT[FERT['District'].astype(str).str.lower().str.contains(d, na=False)].copy()
    if cand.empty:
        cand = FERT.copy()

    # ensure numeric
    cand['Product_N'] = pd.to_numeric(cand['Product_N'], errors='coerce').fillna(0.0)
    cand['Product_P'] = pd.to_numeric(cand['Product_P'], errors='coerce').fillna(0.0)
    cand['Product_K'] = pd.to_numeric(cand['Product_K'], errors='coerce').fillna(0.0)

    def dist(a, b):
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    cand['dist'] = cand.apply(
        lambda r: dist((float(r['Product_N']), float(r['Product_P']), float(r['Product_K'])), target),
        axis=1
    )
    b = cand.sort_values('dist').iloc[0]
    return {
        "product": b.get('Product_Name', ''),
        "company": b.get('Company', ''),
        "npk": (int(round(float(b['Product_N']))), int(round(float(b['Product_P']))), int(round(float(b['Product_K']))))
    }

def pesticide_text(crop: str) -> str:
    df = PEST[PEST['Crop'].astype(str).str.lower().str.strip() == str(crop).lower().strip()]
    if df.empty:
        return "No pesticide data."
    blocks = []
    for (disease, pest), g in df.groupby(['Major_Disease', 'Major_Pest']):
        products = ", ".join(sorted(set(g['Recommended_Pesticide'].astype(str).tolist())))
        sol = "; ".join(sorted(set(g['Disease_Solution'].astype(str).tolist())))
        blocks.append(f"{pest} ({disease}): {products}. Solutions: {sol}.")
    return "\n".join(blocks)

# -------------------------
# Yield recommendation endpoint
# -------------------------
@app.post("/recommend")
def recommend(r: Req):
    base = {"Crop Type": r.crop, "District": r.district, "AREA(ha)": r.area_ha}
    if r.soil_type:
        base["Soil_Type"] = r.soil_type

    row = build_row(base)
    pred = float(MODEL.predict(row)[0])

    prod = choose_product(r.district)
    delivered_total = 120.0
    n, p, k = prod['npk']
    delivered = {
        "total_kg_per_ha": delivered_total,
        "N_kg": round(delivered_total * (n / 100.0), 1),
        "P_kg": round(delivered_total * (p / 100.0), 1),
        "K_kg": round(delivered_total * (k / 100.0), 1),
    }
    pest_block = pesticide_text(r.crop)

    return {
        "predicted_yield_kg_per_ha": round(pred, 2),
        "fertilizer": prod,
        "delivered": delivered,
        "pesticide_advisory": pest_block
    }

# -------------------------
# Auto features (NPK, weather, pH)
# -------------------------
@app.post("/auto_features")
def auto_features(payload: dict):
    lat = payload.get("lat")
    lon = payload.get("lon")
    district = payload.get("district", "")

    soil_row = None
    if district:
        mask = MASTER['District'].astype(str).str.lower().str.contains(str(district).lower(), na=False)
        if mask.any():
            soil_row = MASTER[mask].median(numeric_only=True)
    if soil_row is None:
        soil_row = MASTER.median(numeric_only=True)

    soil_n = float(_safe_get(soil_row, ['Soil N (kg/ha)', 'N(kg/ha)'], 90.0))
    soil_p = float(_safe_get(soil_row, ['Soill P (kg/ha)', 'Soil P (kg/ha)', 'P(kg/ha)'], 40.0))
    soil_k = float(_safe_get(soil_row, ['Soil K (kg/ha)', 'K(kg/ha)'], 40.0))
    soil_ph = float(_safe_get(soil_row, ['Soil_PH', 'ph(0 to 14)'], 6.5))

    temp = None
    hum = None
    rain = None
    if lat is not None and lon is not None:
        try:
            end = datetime.utcnow().date()
            start = end - timedelta(days=6)
            url = (
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                "&hourly=temperature_2m,relativehumidity_2m,precipitation"
                f"&start_date={start.isoformat()}&end_date={end.isoformat()}&timezone=UTC"
            )
            r = requests.get(url, timeout=10)
            j = r.json()
            hourly = j.get("hourly", {})
            temps = hourly.get("temperature_2m", []) or []
            hums = hourly.get("relativehumidity_2m", []) or []
            prec = hourly.get("precipitation", []) or []
            if temps:
                temp = float(sum(temps) / len(temps))
            if hums:
                hum = float(sum(hums) / len(hums))
            if prec:
                rain = float(sum(prec))
        except Exception:
            pass

    temp = temp if temp is not None else 25.0
    hum = hum if hum is not None else 70.0
    rain = rain if rain is not None else 100.0

    return {
        "N": round(soil_n, 1),
        "P": round(soil_p, 1),
        "K": round(soil_k, 1),
        "ph": round(soil_ph, 2),
        "temperature": round(temp, 2),
        "humidity": round(hum, 1),
        "rainfall": round(rain, 1),
        "source": "soil-dataset + open-meteo (7d)"
    }

# small helper used above: safe median extraction
def _safe_get(ser, candidates: List[str], default: float):
    for n in candidates:
        if n in ser.index:
            try:
                v = ser.get(n)
                if pd.isna(v):
                    continue
                return float(v)
            except Exception:
                continue
    return float(default)

# -------------------------
# Helpers for crop recommendation
# -------------------------
def _col_median(df: pd.DataFrame, names: List[str], default: float) -> float:
    for n in names:
        if n in df.columns:
            try:
                return float(pd.to_numeric(df[n], errors="coerce").median())
            except Exception:
                continue
    return float(default)

def _clamp_series(s: pd.Series, low: float, high: float) -> pd.Series:
    return s.clip(lower=low, upper=high)

# robust rainfall median calculation
def _robust_rainfall_median(df: pd.DataFrame) -> float:
    if "rainfall(in mm)" in df.columns:
        try:
            val = pd.to_numeric(df["rainfall(in mm)"], errors="coerce").median()
            return float(val) if not np.isnan(val) else 120.0
        except Exception:
            return 120.0
    rf_cols = [c for c in df.columns if "Rainfall" in c]
    if not rf_cols:
        return 120.0
    numeric = df[rf_cols].apply(lambda col: pd.to_numeric(col, errors="coerce"))
    try:
        s = numeric.sum(axis=1)
        val = s.median()
        return float(val) if not np.isnan(val) else 120.0
    except Exception:
        return 120.0

# -------------------------
# Crop recommendation endpoint (clamp + top-3 + confidence)
# -------------------------
@app.post("/recommend_crop")
def recommend_crop(req: CropReq):
    cols = [
        "N(kg/ha)",
        "P(kg/ha)",
        "K(kg/ha)",
        "temperature(in °C)",
        "humidity(in %)",
        "ph(0 to 14)",
        "rainfall(in mm)"
    ]

    n_med = _col_median(MASTER, ["Soil N (kg/ha)", "N(kg/ha)"], 100.0)
    p_med = _col_median(MASTER, ["Soil P (kg/ha)", "Soill P (kg/ha)", "P(kg/ha)"], 60.0)
    k_med = _col_median(MASTER, ["Soil K (kg/ha)", "K(kg/ha)"], 100.0)
    t_med = _col_median(MASTER, ["temperature(in °C)", "Temperature (°C)"], 25.0)
    ph_med = _col_median(MASTER, ["Soil_PH", "ph(0 to 14)"], 6.5)
    r_med = _robust_rainfall_median(MASTER)

    X = pd.DataFrame([{
        cols[0]: req.N if req.N is not None else n_med,
        cols[1]: req.P if req.P is not None else p_med,
        cols[2]: req.K if req.K is not None else k_med,
        cols[3]: req.temperature if req.temperature is not None else t_med,
        cols[4]: req.humidity if req.humidity is not None else 70.0,
        cols[5]: req.ph if req.ph is not None else ph_med,
        cols[6]: req.rainfall if req.rainfall is not None else r_med
    }], columns=cols)

    # clamp before scaling
    X_raw = X.copy()
    X_raw["temperature(in °C)"] = _clamp_series(X_raw["temperature(in °C)"], -5, 45)
    X_raw["humidity(in %)"] = _clamp_series(X_raw["humidity(in %)"], 0, 100)
    X_raw["ph(0 to 14)"] = _clamp_series(X_raw["ph(0 to 14)"], 3.5, 8.5)
    X_raw["rainfall(in mm)"] = _clamp_series(X_raw["rainfall(in mm)"], 0, 1000)
    X_raw["N(kg/ha)"] = _clamp_series(X_raw["N(kg/ha)"], 0, 300)
    X_raw["P(kg/ha)"] = _clamp_series(X_raw["P(kg/ha)"], 0, 200)
    X_raw["K(kg/ha)"] = _clamp_series(X_raw["K(kg/ha)"], 0, 300)

    try:
        Xs = CROP_SCALER.transform(X_raw)
    except Exception as e:
        return {"error": f"scaler error: {e}", "X_raw": X_raw.to_dict(orient="records")}

    # If you trained/used an ensemble, average probs here; otherwise use single model
    try:
        probs = CROP_MODEL.predict_proba(Xs)[0]
    except Exception:
        probs = None

    pred_idx = int(CROP_MODEL.predict(Xs)[0])
    pred_crop = CROP_LABEL.inverse_transform([pred_idx])[0]

    top_list: List[Dict[str, Any]] = []
    top_conf_raw: Optional[float] = None
    if probs is not None:
        classes = list(CROP_LABEL.classes_)
        pairs_sorted = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)
        top_list = [{"crop": c, "prob": round(float(p) * 100, 1)} for c, p in pairs_sorted[:5]]
        top_conf_raw = float(pairs_sorted[0][1]) if pairs_sorted else None
    else:
        top_conf_raw = 1.0

    if top_conf_raw is not None and top_conf_raw < 0.30:
        return {
            "recommended_crop": pred_crop,
            "confidence_percent": round(top_conf_raw * 100, 1),
            "top_candidates": top_list[:3],
            "note": "Ambiguous: model confidence low; showing top candidates"
        }

    return {
        "recommended_crop": pred_crop,
        "confidence_percent": round((top_conf_raw * 100) if top_conf_raw is not None else 100.0, 1),
        "top_candidates": top_list
    }

# -------------------------
# Debug endpoint (optional)
# -------------------------
@app.post("/debug_predict")
def debug_predict(req: CropReq):
    cols = ["N(kg/ha)","P(kg/ha)","K(kg/ha)","temperature(in °C)","humidity(in %)","ph(0 to 14)","rainfall(in mm)"]
    X = pd.DataFrame([{
        cols[0]: req.N if req.N is not None else 0,
        cols[1]: req.P if req.P is not None else 0,
        cols[2]: req.K if req.K is not None else 0,
        cols[3]: req.temperature if req.temperature is not None else 0,
        cols[4]: req.humidity if req.humidity is not None else 0,
        cols[5]: req.ph if req.ph is not None else 0,
        cols[6]: req.rainfall if req.rainfall is not None else 0
    }], columns=cols)

    try:
        X_scaled = CROP_SCALER.transform(X)
    except Exception as e:
        return {"error": "scaler.transform failed", "detail": str(e), "X_raw": X.to_dict(orient="records")}

    try:
        probs = CROP_MODEL.predict_proba(X_scaled)[0].tolist()
    except Exception:
        probs = None

    pred_idx = int(CROP_MODEL.predict(X_scaled)[0])
    classes = list(CROP_LABEL.classes_)

    mapping = []
    if probs is not None and len(probs) == len(classes):
        for cls, p in zip(classes, probs):
            mapping.append({"class": cls, "prob": round(float(p), 6)})
    else:
        mapping = [{"class": cls, "prob": None} for cls in classes]

    return {
        "X_raw": X.to_dict(orient="records")[0],
        "X_scaled": X_scaled.tolist()[0],
        "pred_index": pred_idx,
        "pred_class": CROP_LABEL.inverse_transform([pred_idx])[0],
        "probs_top10": sorted(mapping, key=lambda x: (x['prob'] is not None, x['prob']), reverse=True)[:10],
        "all_probs_count": len(probs) if probs is not None else None,
        "classes_count": len(classes)
    }
