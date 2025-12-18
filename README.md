# NHL Game Prediction Model

Poisson + Monte Carlo prediction system for NHL games with prop bets, deployed on Streamlit Cloud.

## Live Demo

**URL:** https://mastervb99-nhl-predictor.streamlit.app

## Features

### Implemented (V1.0)
- **Core Prediction Engine:** Poisson distribution + 10K Monte Carlo simulations
- **Win Probabilities:** Full game and 60-minute regulation
- **Over/Under Analysis:** Multiple lines with edge calculation
- **1st Period Model:** O/U for 0.5, 1.5, 2.5 lines
- **GIFT Prop:** Goal In First 10 Minutes probability
- **SOG Prop:** 1+ Shot on Goal in First 2 Minutes
- **B2B Detection:** Back-to-back badges with schedule lookup
- **Team Form:** Last 5 games per team
- **H2H History:** Head-to-head matchup history
- **PDF Export:** Downloadable game reports
- **Dark Theme UI:** Card-style expandable game cards

### Data Sources
- **NHL API:** Schedule, game logs, standings (live)
- **MoneyPuck:** Goalie stats - GSAX, SV% (CSV)
- **Fallback Stats:** Team metrics when APIs unavailable

### Pending (V1.1)
- Live odds integration (the-odds-api.com)
- Goalie dropdown selection
- Natural Stat Trick scraping
- ML ensemble (XGBoost/LightGBM) training

## Project Structure

```
nhl-predictor/
├── agents/
│   ├── __init__.py
│   ├── data_ingestor.py      # Live NHL API, MoneyPuck integration
│   ├── poisson_engine.py     # Expected goals calculation
│   ├── monte_carlo.py        # 10K game simulations
│   ├── period_model.py       # 1st period predictions
│   ├── props_model.py        # GIFT and SOG props
│   ├── edge_calculator.py    # Kelly criterion, EV analysis
│   └── feature_engineer.py   # 3-season weighted features
├── utils/
│   ├── __init__.py
│   └── pdf_export.py         # PDF report generation
├── api/
│   └── main.py               # FastAPI backend (optional)
├── models/
│   └── schemas.py            # Pydantic models
├── streamlit_app.py          # Main Streamlit UI
├── requirements.txt
├── ARCHITECTURE.md           # System design docs
├── REVISION_PLAN.md          # Implementation plan
└── README.md
```

## Local Development

```bash
# Clone
git clone https://github.com/mastervb99/nhl-predictor.git
cd nhl-predictor

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_app.py
```

## Deployment

Deployed on Streamlit Cloud:
1. Connect GitHub repo at share.streamlit.io
2. Main file: `streamlit_app.py`
3. Branch: `main`

## Configuration

### Optional: Live Odds API
To enable automatic odds fetching, set environment variable:
```
ODDS_API_KEY=your_key_from_the-odds-api.com
```
Free tier: 500 requests/month.

Without this, users can manually enter odds in the sidebar.

## Model Details

### Poisson Engine
- Calculates expected goals (λ) per team
- Factors: GF/60, GA/60, home ice advantage (6%), goalie GSAX

### Monte Carlo Simulator
- 10,000 game simulations
- Overtime/shootout resolution
- 60-min regulation probabilities

### 1st Period Model
- ~32% of full game goals (historical average)
- O/U probabilities for 0.5, 1.5, 2.5 lines

### Props Models
- **GIFT:** Poisson-based, ~58% league average
- **SOG:** Shot pace extrapolation

### Edge Calculator
- American odds to implied probability
- Kelly criterion bet sizing (1/4 Kelly)
- Confidence tiers: STRONG/LEAN/PASS

## API Endpoints (Optional FastAPI)

```
GET  /api/games      - Today's schedule
POST /api/predict    - Run prediction
GET  /api/goalies    - Goalie roster
GET  /api/odds       - Betting lines
```

## Contributing

1. Fork the repo
2. Create feature branch
3. Submit PR

## License

MIT
