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
git clone https://github.com/yourusername/krishai-project.git
cd krishai-project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the backend**
```bash
cd backend
uvicorn main:app --reload --port 8000
```
This starts the FastAPI server at http://localhost:8000

4. **Run the frontend**
```bash
cd frontend
python -m http.server 8001
```
Or just open `index.html` in your browser

5. **Open it up**
Go to http://localhost:8001 in your browser and you're good to go!

## Project Structure

```
krishai-project/
├── backend/
│   ├── main.py              # FastAPI app, all the endpoints
│   ├── models/              # Trained ML models (.pkl files)
│   ├── data/                # CSV files with fertilizer/pesticide info
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── index.html           # Main page
│   ├── styles.css           # Styling
│   └── app.js              # All the frontend logic
├── datasets/                # Original datasets we collected
│   ├── merged_ready3.csv    # Main agricultural data
│   ├── fert_products_clean.csv
│   └── pesticide_clean2.csv
└── README.md               # You are here!
```

## How We Built It

### Data Collection (the boring part)
- Spent 2 weeks collecting data from government agriculture sites
- Got district-wise soil data (NPK levels, pH) for 28 districts in Odisha
- Compiled fertilizer products available locally (brand names, NPK ratios)
- Gathered pesticide recommendations for different crops and pests
- Historical yield data from past 5 years

### Machine Learning (the fun part)
- Cleaned data, handled missing values (used KNN imputation)
- Tried multiple models - Random Forest worked best for classification
- Gradient Boosting gave us the best R² score for yield prediction
- Used GridSearchCV for hyperparameter tuning (took forever to run)
- Final accuracy: 94.2% for crop recommendation, R² of 0.89 for yield

### Frontend (keeping it simple)
- No frameworks! Just vanilla JS because we needed it to load fast on 2G/3G
- Made it mobile-first since most farmers use phones
- Added auto-location feature so farmers don't have to type their district
- Color-coded results (green for fertilizer, yellow for pesticide) for easy reading
- Total bundle size: 18KB - loads in like 1 second even on slow connections

### Backend (making it work)
- FastAPI because it's fast and has automatic API docs
- Three main endpoints: `/recommend`, `/recommend_crop`, `/auto_features`
- Integrated weather API for real-time temperature/humidity/rainfall
- SQLite for storing fertilizer and pesticide data (could've used CSV but DB is cleaner)
- Added CORS middleware so frontend and backend can talk

## Challenges We Faced

1. **Data quality** - Government datasets had tons of missing values and inconsistencies. Spent days cleaning it.
2. **API rate limits** - OpenStreetMap was throttling us. Added caching to fix it.
3. **Mobile optimization** - Had to test on actual phones because Chrome DevTools mobile view isn't the same as real devices
4. **Model size** - Our first model was 50MB. Had to optimize it down to 5MB so it loads faster
5. **CORS issues** - Spent an entire evening figuring out why frontend couldn't talk to backend. Classic web dev problem.

## What We Learned

- Data cleaning takes 80% of the time in ML projects (not kidding)
- Vanilla JS is actually pretty powerful, don't always need a framework
- FastAPI is amazing for quick API development
- Testing on real devices >>> browser DevTools
- Git is a lifesaver when you accidentally break everything

## Future Plans (if we continue this)

- [ ] Add multilingual support (Odia, Hindi)
- [ ] Mobile app (React Native maybe?)
- [ ] IoT integration for automatic soil testing
- [ ] Weather alerts for farmers
- [ ] Market price predictions
- [ ] Government scheme recommendations
- [ ] Community forum for farmers to share tips

## Team

**Yashraj Singh (2328058)** - Frontend, Data Collection, UX Design  
Handled all the data gathering, built the entire frontend, integrated APIs, and made sure it works on phones.

**Sanjam Das (XXXX)** - Machine Learning, Backend, Data Preprocessing  
Cleaned the data, trained the ML models, built the FastAPI backend, and got everything deployed.

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

**Database:**
- SQLite (for now - might migrate to PostgreSQL later)

## Contributing

This is a college project, so we're not really accepting contributions right now. But if you want to fork it and make your own version, go ahead! Just give us credit somewhere.

## License

MIT License - do whatever you want with this code, just don't sue us if your crops fail 😅

## Acknowledgments

- Our project guide for not killing us during deadlines
- KIIT University for the resources (and WiFi)
- Odisha Agriculture Department for the data
- Stack Overflow for fixing our bugs at 2 AM
- Coffee, lots of coffee

## Contact

Got questions? Found a bug? Want to roast our code?

- Yashraj: yashraj.singh@example.com
- Sanjam: sanjam.das@example.com

Or just open an issue on GitHub and we'll get back to you... eventually.

---

Made with ☕ and 🌾 in Bhubaneswar, Odisha

*P.S. - If you're a farmer and actually using this, please let us know! Would love to hear feedback from real users.*
