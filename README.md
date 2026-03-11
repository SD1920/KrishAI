# KrishAI — AI-Powered Farm Intelligence System

> Predict crop yield · Recommend fertiliser · Advise on pesticides · Suggest best crop  
> **Live demo:** https://krishai-lrx0.onrender.com

---

## What it does

KrishAI takes a farmer's location, crop, area, and soil type and returns:

- **Yield prediction** — kg/ha and total yield for their area
- **Fertiliser recommendation** — best-matching NPK product from a district-level catalogue
- **Pesticide advisory** — pest and disease-specific chemical recommendations for the crop
- **Crop recommendation** — top crop candidates ranked by probability given local soil and weather conditions

All inputs can be auto-filled from device GPS + live weather (Open-Meteo API) + soil dataset medians. Frontend works on 3G (18KB JS bundle, no framework).

---

## Dataset

| Source | Rows | Coverage |
|---|---|---|
| Odisha district yield records | 3,472 | 28 districts, 1993–2017, 23 features |
| National crop recommendation | 2,200 | NPK · temperature · humidity · pH · rainfall |
| Fertiliser product catalogue | — | District-level NPK products |
| Pesticide advisory | — | Crop × pest × disease mappings |

**Key EDA findings:**
- Median soil pH = 5.25 (strongly acidic — below optimal 6.0–7.5 range)
- 58% of records are Red Soil
- Yield improved ~45% over the 30-year period (620 → 900 kg/ha median)
- Rice dominates by cultivated area; Groundnut leads by median yield (~1,500 kg/ha)
- Year (r=0.27) and Area (r=0.29) are the strongest yield predictors

---

## Models

| Task | Model | Metric |
|---|---|---|
| Yield prediction | ExtraTrees Regressor | R² = 0.89 |
| Crop recommendation | Calibrated Random Forest | 94.2% accuracy, 22 classes |
| Fertiliser | NPK grid search over yield model | — |

Benchmarked against RF, XGBoost, LightGBM, CatBoost — ExtraTrees selected on R² + inference speed. Fertiliser optimisation runs a grid search (~150 NPK combinations) through the yield model rather than a static lookup table.

---

## Stack

**Backend:** Python · FastAPI · scikit-learn · pandas · joblib  
**Frontend:** Vanilla JS · CSS (no framework, 18KB bundle)  
**Deployment:** Render — single service, API + static frontend  
**Weather:** Open-Meteo API (7-day rolling average)  
**Geocoding:** OpenStreetMap Nominatim

---

## Project structure

```
KrishAI/
├── backend/
│   ├── app.py              # FastAPI — /recommend, /recommend_crop, /auto_features
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── styles.css
├── ml_artifacts/
│   ├── crop_yield_model.pkl
│   ├── crop_recommendation_model.pkl
│   ├── crop_label_encoder.pkl
│   ├── crop_scaler.pkl
│   ├── encoders.pkl
│   └── feature_cols.pkl
├── data/
│   ├── merged_ready3.csv
│   ├── Crop_recommendation2.csv
│   ├── fert_products_clean.csv
│   └── pesticide_clean2.csv
└── render.yaml
```

---

## Run locally

```bash
git clone https://github.com/SD1920/KrishAI
cd KrishAI
pip install -r backend/requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` — frontend is served from the same process.

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/recommend` | Yield + fertiliser + pesticide |
| POST | `/recommend_crop` | Crop recommendation from soil/weather |
| POST | `/auto_features` | Auto-fill from GPS location |
| GET | `/docs` | Swagger UI |

---

## My role

ML pipeline, data preprocessing, model training and evaluation, FastAPI backend, Render deployment. Frontend built collaboratively with teammates.

---

## Limitations

- Yield model trained on Odisha data only — predictions outside Odisha are extrapolations
- NPK features have near-zero variance in training data — fertiliser recs driven by district + crop matching
- Crop recommender trained on national dataset — may not reflect Odisha microclimates
- Free Render tier spins down after inactivity — first request takes ~50s after idle