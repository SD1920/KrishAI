# 🌾 KrishAI - AI-Powered Farm Intelligence System

Hey there! Welcome to KrishAI, our college project that's trying to make farming a bit smarter using AI. We built this because honestly, farmers deserve better tools than just guessing what to plant or how much fertilizer to use.

## What's This All About?

So basically, KrishAI is a web app that helps farmers make better decisions about their crops. You tell it where you are, what you want to grow, and it gives you:
- How much yield you can expect
- What fertilizers to use (with actual product names!)
- Pesticide recommendations
- Or if you're not sure what to plant, it'll suggest the best crop for your soil and weather

## Why We Built This

We're from Odisha, and agriculture is huge here. But most farmers still rely on traditional methods or advice from neighbors. We thought - why not use machine learning to predict yields and recommend crops based on actual data? Plus it was a fun way to learn ML and web development lol.

## What It Can Do

### For Farmers:
- **Auto-detect your location** - just click a button and it figures out where you are (uses your phone's GPS)
- **Get yield predictions** - tell us what crop and how much land, we'll estimate your harvest
- **Fertilizer recommendations** - not just NPK numbers, but actual products you can buy locally
- **Crop suggestions** - if you're unsure what to plant, we'll recommend based on your soil and weather
- **Works on phones** - because most farmers use smartphones, not laptops

### Technical Stuff (for nerds like us):
- Frontend: Plain vanilla JavaScript (no React/Vue bloat - needed it fast on slow connections)
- Backend: FastAPI (Python) - super fast API responses
- ML Models: Random Forest for crop recommendation, Gradient Boosting for yield prediction
- Data: 6000+ records of soil data, weather patterns, and crop yields from Odisha
- APIs: OpenStreetMap for location, Open-Meteo for weather forecasts

## How to Run This Thing

### Prerequisites
You'll need:
- Python 3.8+ (we used 3.10)
- pip (comes with Python)
- A browser (obviously)
- Internet connection (for weather APIs)

### Setup

1. **Clone this repo**
```bash
git clone https://github.com/SD1920/KrishAI.git
cd KrishAI
```

2. **Set up virtual environment** (recommended)
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Mac/Linux
```

3. **Install dependencies**
```bash
pip install -r backend/requirements.txt
```

4. **Run the backend**
```bash
cd backend
python app.py
```
This starts the FastAPI server at http://localhost:8000

5. **Run the frontend**
Open a new terminal:
```bash
cd frontend
python -m http.server 8001
```
Or just open `index.html` in your browser

6. **Open it up**
Go to http://localhost:8001 in your browser and you're good to go!

## Project Structure

```
KrishAI/
├── backend/
│   ├── app.py                  # Main FastAPI application (15KB)
│   ├── requirements.txt        # Python dependencies
│   └── __pycache__/           # Python cache files
├── frontend/
│   ├── index.html              # Main page (4KB)
│   ├── styles.css              # Styling (2KB)
│   ├── app.js                  # Frontend logic (7KB)
│   └── .gitignore             # Git ignore file
├── data/                       # All datasets
│   ├── merged_ready3.csv       # Main agricultural data (600KB, 6000+ records)
│   ├── Crop_recommendation2.csv # Crop training data (144KB)
│   ├── fert_products_clean.csv # Fertilizer products (3KB)
│   └── pesticide_clean2.csv    # Pesticide recommendations (2KB)
├── ml_artifacts/               # Trained models and notebooks
│   ├── crop_recommendation_model.pkl  # Random Forest model (19MB)
│   ├── crop_yield_model.pkl          # Gradient Boosting model (19MB)
│   ├── crop_label_encoder.pkl        # Label encoder for crops
│   ├── crop_scaler.pkl              # Feature scaler
│   ├── encoders.pkl                 # Other encoders
│   ├── feature_cols.pkl             # Feature column names
│   ├── AI_PROJECT.ipynb            # Main training notebook
│   └── crop_recommender.ipynb      # Crop recommendation notebook
├── scripts/                    # Utility scripts
├── venv/                       # Virtual environment (not in git)
└── README.md                   # You are here!
```

## How We Built It

### Data Collection (the boring part)
- Spent 2 weeks collecting data from government agriculture sites
- Got district-wise soil data (NPK levels, pH) for 28 districts in Odisha
- Compiled fertilizer products available locally (brand names, NPK ratios)
- Gathered pesticide recommendations for different crops and pests
- Historical yield data from past 50 years

### Machine Learning (the fun part)
- Cleaned data, handled missing values (used KNN imputation)
- Tried multiple models - Random Forest worked best for classification
- Gradient Boosting gave us the best R² score for yield prediction
- Used GridSearchCV for hyperparameter tuning (took forever to run)
- Final accuracy: 94.2% for crop recommendation, R² of 0.89 for yield
- Model training done in Jupyter notebooks (check `ml_artifacts/` folder)

### Frontend (keeping it simple)
- No frameworks! Just vanilla JS because we needed it to load fast on 2G/3G
- Made it mobile-first since most farmers use phones
- Added auto-location feature so farmers don't have to type their district
- Color-coded results (green for fertilizer, yellow for pesticide) for easy reading
- Total bundle size: 18KB - loads in like 1 second even on slow connections

### Backend (making it work)
- FastAPI in `app.py` because it's fast and has automatic API docs
- Three main endpoints: `/recommend`, `/recommend_crop`, `/auto_features`
- Integrated weather API for real-time temperature/humidity/rainfall
- Loads pre-trained models from `ml_artifacts/` folder on startup
- Added CORS middleware so frontend and backend can talk

## Challenges We Faced

1. **Data quality** - Government datasets had tons of missing values and inconsistencies. Spent days cleaning it.
2. **API rate limits** - OpenStreetMap was throttling us. Added caching to fix it.
3. **Model size** - Our first model was 50MB. Had to optimize it down to 19MB (still big but manageable)
4. **CORS issues** - Spent an entire evening figuring out why frontend couldn't talk to backend. Classic web dev problem.

## What We Learned

- Data cleaning takes 80% of the time in ML projects (not kidding)
- Vanilla JS is actually pretty powerful, don't always need a framework
- FastAPI is amazing for quick API development
- Testing on real devices >>> browser DevTools
- Git is a lifesaver when you accidentally break everything

## Future Plans (if we continue this)

- [ ] Add multilingual support (Odia, Hindi)
- [ ] Mobile app (React Native)
- [ ] IoT integration for automatic soil testing
- [ ] Weather alerts for farmers
- [ ] Market price predictions
- [ ] Government scheme recommendations
- [ ] Community forum for farmers to share tips

## Team

**Aritro Halder** - Machine Learning, Data Collection, Data Preprocessing :

Collected the data, preprocessed the data, and built "Crop_Recommendation" model.

**Uddipt Shankar** - Machine Learning, Data Preprocessing, Data Analytics :

Built the ML models, Analyzed the data, built "Yield_Prediction_and_Optimization" model.

**Vinayaka Voleti** - Frontend, UX Design :

Handled all the data gathering, built the entire frontend, integrated APIs, and made sure it works on phones.

**Sanjam Das** - Machine Learning, Backend :

Collected & cleaned the data, trained the ML models, built the FastAPI backend, and got everything deployed.

## Tech Stack

**Frontend:**
- HTML5, CSS3, Vanilla JavaScript
- OpenStreetMap Nominatim API

**Backend:**
- Python 3.10
- FastAPI
- Scikit-learn (Random Forest, Gradient Boosting)
- Pandas, NumPy
- Open-Meteo Weather API

**Development:**
- Google Colab Notebook for model training
- Git for version control

## Acknowledgments

- KIIT University for the resources
- Odisha Agriculture Department and Indian Gov Datasets for the data
- Stack Overflow & ChatGPT, Claude for fixing our bugs.
  
## Contact

Got questions? Found a bug? Want to roast our code?

- Sanjam: sanjamdas2@gmail.com

Or just open an issue on GitHub and we'll get back to you... eventually.

---

Made with ☕ and 🌾 in Bhubaneswar, Odisha

*P.S. - If you're a farmer and actually using this, please let us know! Would love to hear feedback from real users.*
