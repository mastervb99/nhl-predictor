# NHL Prediction Model - Revision Plan

**Last Updated:** 2024-12-18
**Status:** V1.0 Complete - Deployed on Streamlit Cloud

---

## Section 1: Client's Complete Instructions

### Original Job Posting
NHL betting model using Poisson & Monte Carlo, 3-season weighting, auto-loaded games, goalie dropdown, data from NST/MoneyPuck/Hockey-Reference.

### Additional Requirements from Tyler
- Card-style expandable UI (like NFL reference screenshot)
- B2B detection with badges
- L5 games and H2H history
- 1st period O/U predictions
- GIFT prop (goal in first 10 mins)
- 1+ SOG in first 2 mins prop
- PDF export

---

## Section 2: Implementation Status

### Completed (V1.0)

| Phase | Feature | Status | File(s) |
|-------|---------|--------|---------|
| 1 | Live Data Integration | ✅ Complete | `agents/data_ingestor.py` |
| 2 | Card-Style UI | ✅ Complete | `streamlit_app.py` |
| 3 | B2B Detection | ✅ Complete | `agents/data_ingestor.py`, `streamlit_app.py` |
| 4 | 1st Period Model | ✅ Complete | `agents/period_model.py` |
| 5 | L5/H2H Data | ✅ Complete | `agents/data_ingestor.py`, `streamlit_app.py` |
| 6 | GIFT Prop | ✅ Complete | `agents/props_model.py` |
| 7 | SOG Prop | ✅ Complete | `agents/props_model.py` |
| 8 | PDF Export | ✅ Complete | `utils/pdf_export.py` |

### Pending (V1.1)

| Feature | Status | Notes |
|---------|--------|-------|
| Live Odds API | Pending | Requires the-odds-api.com key |
| Goalie Dropdown | Pending | Data ready, UI needs wiring |
| NST Scraping | Pending | Using fallback stats |
| ML Training | Pending | XGBoost/LightGBM ensemble |

---

## Section 3: Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     STREAMLIT FRONTEND                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Card-style dark theme UI with expandable game cards    │    │
│  │  - Win probabilities, expected goals, score prediction  │    │
│  │  - 1st period O/U, GIFT/SOG props, B2B badges          │    │
│  │  - L5 games, H2H history, PDF download                  │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    PREDICTION ENGINE                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Poisson      │  │ Monte Carlo  │  │ Period Model │          │
│  │ Engine       │  │ Simulator    │  │ (1st Period) │          │
│  │ ✅ Complete  │  │ ✅ Complete  │  │ ✅ Complete  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Props Model  │  │ Edge Calc    │  │ Feature Eng  │          │
│  │ GIFT + SOG   │  │ Kelly/EV     │  │ 3-Season Wt  │          │
│  │ ✅ Complete  │  │ ✅ Complete  │  │ ✅ Complete  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                      DATA LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ NHL API      │  │ MoneyPuck    │  │ Odds API     │          │
│  │ Schedule/Logs│  │ Goalie Stats │  │ (Optional)   │          │
│  │ ✅ Live      │  │ ✅ CSV       │  │ 🔜 V1.1      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐                            │
│  │ Fallback     │  │ Cache Layer  │                            │
│  │ Team Stats   │  │ 5-min TTL    │                            │
│  │ ✅ Ready     │  │ ✅ Active    │                            │
│  └──────────────┘  └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Section 4: File Reference

### Core Agents

| File | Purpose | Key Methods |
|------|---------|-------------|
| `agents/poisson_engine.py` | Expected goals calculation | `predict()`, `calculate_lambdas()` |
| `agents/monte_carlo.py` | 10K game simulations | `simulate_games()`, `calculate_over_under()` |
| `agents/period_model.py` | 1st period predictions | `predict_first_period()` |
| `agents/props_model.py` | GIFT and SOG props | `calculate_gift_probability()`, `calculate_sog_probability()` |
| `agents/edge_calculator.py` | Betting edge analysis | `full_game_analysis()`, `calculate_kelly()` |
| `agents/data_ingestor.py` | Live data fetching | `fetch_schedule()`, `is_back_to_back()`, `fetch_last_n_games()` |
| `agents/feature_engineer.py` | 3-season weighting | `calculate_weighted_stats()` |

### Utilities

| File | Purpose |
|------|---------|
| `utils/pdf_export.py` | PDF report generation |
| `models/schemas.py` | Pydantic data models |

### Configuration

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Streamlit theme |

---

## Section 5: Data Sources

| Source | Data | Method | Status |
|--------|------|--------|--------|
| NHL API (nhle.com) | Schedule, game logs, standings | REST API | ✅ Live |
| MoneyPuck | Goalie GSAX, SV%, xG | CSV download | ✅ Ready |
| the-odds-api.com | Live betting lines | REST API | 🔜 Optional |
| Fallback stats | Team GF/GA/Shots per 60 | Hardcoded | ✅ Active |

---

## Section 6: Deployment

### Current: Streamlit Cloud
- **URL:** https://mastervb99-nhl-predictor.streamlit.app
- **Repo:** https://github.com/mastervb99/nhl-predictor
- **Branch:** main
- **Main file:** streamlit_app.py

### Optional: Vercel (Future)
If client requires Vercel deployment:
- Rebuild frontend in Next.js/React
- Deploy FastAPI backend as serverless functions
- Additional scope and timeline required

---

## Section 7: Future Enhancements (V1.1+)

1. **Live Odds Integration**
   - Integrate the-odds-api.com
   - Auto-populate betting lines
   - Real-time edge updates

2. **Goalie Selection**
   - Dropdown with goalie roster
   - Stats display (GSAX, SV%, GP)
   - Auto-adjust predictions

3. **ML Ensemble**
   - Train XGBoost/LightGBM on historical data
   - Blend with Poisson/MC output
   - Feature importance display

4. **Natural Stat Trick**
   - Scrape advanced metrics (CF%, xGF, xGA)
   - Replace fallback stats
   - Add to feature engineering

5. **Expanded Props**
   - Team totals
   - Period spreads
   - Player props (if data available)

---

## Section 8: Resuming Development

To continue development:

1. **Clone repo:**
   ```bash
   git clone https://github.com/mastervb99/nhl-predictor.git
   cd nhl-predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run locally:**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Key files to modify:**
   - `streamlit_app.py` - UI changes
   - `agents/data_ingestor.py` - Data source changes
   - `agents/*.py` - Model changes

5. **Deploy changes:**
   ```bash
   git add . && git commit -m "description" && git push
   ```
   Streamlit Cloud auto-deploys on push to main.

---

## Section 9: Contact

- **GitHub:** https://github.com/mastervb99/nhl-predictor
- **Client:** Tyler (Upwork)
