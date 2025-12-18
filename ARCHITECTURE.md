# NHL Game Prediction Model - System Architecture

**Last Updated:** 2024-12-18
**Version:** 1.0

## Overview

Multi-agent prediction system combining Poisson regression, Monte Carlo simulation, and prop bet models for NHL game outcome and betting analysis. Deployed on Streamlit Cloud.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT CLOUD DEPLOYMENT                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      streamlit_app.py                                │   │
│  │  ─────────────────────────────────────────────────────────────────  │   │
│  │  • Dark theme card-style UI                                         │   │
│  │  • Expandable game cards with tabs                                  │   │
│  │  • B2B badges, L5 games, H2H history                               │   │
│  │  • 1st Period, Props, Recent Form tabs                             │   │
│  │  • PDF download button                                              │   │
│  │  • Manual game entry sidebar                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PREDICTION ENGINE (Python)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │   POISSON    │   │ MONTE CARLO  │   │   PERIOD     │   │    PROPS     │ │
│  │   ENGINE     │──▶│  SIMULATOR   │──▶│   MODEL      │   │    MODEL     │ │
│  │              │   │              │   │              │   │              │ │
│  │ • λ_home     │   │ • 10K sims   │   │ • 1st period │   │ • GIFT prob  │ │
│  │ • λ_away     │   │ • Win prob   │   │ • O/U lines  │   │ • SOG prob   │ │
│  │ • Goalie adj │   │ • O/U prob   │   │ • 32% factor │   │ • Vs league  │ │
│  │ • Home ice   │   │ • Score dist │   │              │   │              │ │
│  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘ │
│         │                                                         │         │
│         ▼                                                         ▼         │
│  ┌──────────────┐                                         ┌──────────────┐ │
│  │   FEATURE    │                                         │    EDGE      │ │
│  │  ENGINEER    │                                         │  CALCULATOR  │ │
│  │              │                                         │              │ │
│  │ • 3yr weight │                                         │ • Kelly %    │ │
│  │ • Home/Away  │                                         │ • EV calc    │ │
│  │ • Rolling    │                                         │ • Line value │ │
│  └──────────────┘                                         └──────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │   NHL API       │  │   MoneyPuck     │  │      Fallback Stats         │ │
│  │   (nhle.com)    │  │                 │  │                             │ │
│  │  ─────────────  │  │  • Goalie GSAX  │  │  • Team GF/60, GA/60       │ │
│  │  • Schedule     │  │  • Save %       │  │  • Shots/60                 │ │
│  │  • Game logs    │  │  • xG stats     │  │  • All 32 teams             │ │
│  │  • Standings    │  │  • CSV format   │  │                             │ │
│  │  • B2B detect   │  │                 │  │                             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐                                   │
│  │   Odds API      │  │   PDF Export    │                                   │
│  │   (Optional)    │  │                 │                                   │
│  │  ─────────────  │  │  • ReportLab    │                                   │
│  │  • Live lines   │  │  • Game reports │                                   │
│  │  • the-odds-api │  │  • Overview PDF │                                   │
│  └─────────────────┘  └─────────────────┘                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Agent Specifications

### 1. DataIngestor (`agents/data_ingestor.py`)

**Role:** Fetch, validate, and normalize data from multiple NHL sources

**Methods:**
| Method | Purpose |
|--------|---------|
| `fetch_schedule()` | Get today's games from NHL API |
| `fetch_standings()` | Get current standings |
| `fetch_game_logs()` | Get team game history |
| `is_back_to_back()` | Detect B2B situations |
| `fetch_last_n_games()` | Get L5 games for team |
| `fetch_h2h_games()` | Get head-to-head history |
| `fetch_moneypuck_goalies()` | Download goalie stats CSV |

**Data Sources:**
| Source | Stats | Method |
|--------|-------|--------|
| NHL API (nhle.com) | Schedule, game logs, standings | REST API |
| MoneyPuck | GSAX, xG, goalie metrics | CSV download |
| The Odds API | Live lines (optional) | REST API |

### 2. PoissonEngine (`agents/poisson_engine.py`)

**Role:** Calculate expected goals using Poisson regression

**Model:**
```
λ_home = (home_attack × away_defense × home_ice × goalie_adj) × league_avg
λ_away = (away_attack × home_defense × goalie_adj) × league_avg
```

**Parameters:**
- `home_ice_advantage`: 1.06 (6% boost)
- `league_avg_goals`: ~3.0 goals/team/game
- `goalie_adjustment`: Based on GSAX and SV% vs league average

### 3. MonteCarloSimulator (`agents/monte_carlo.py`)

**Role:** Simulate games to generate probability distributions

**Configuration:**
```python
N_SIMULATIONS = 10_000
OVERTIME_PROB = 0.23  # ~23% go to OT
SHOOTOUT_PROB = 0.50  # Of OT games
```

**Outputs:**
- Win probability distribution
- Score probability matrix
- Over/Under probability for any total
- 60-minute result probabilities

### 4. PeriodModel (`agents/period_model.py`)

**Role:** Calculate 1st period predictions

**Model:**
```python
FIRST_PERIOD_FACTOR = 0.32  # ~32% of goals in 1st period
p1_lambda = full_game_lambda * 0.32
```

**Outputs:**
- 1st period expected goals (home/away)
- O/U probabilities for 0.5, 1.5, 2.5 lines
- Most likely 1st period score

### 5. PropsModel (`agents/props_model.py`)

**Role:** Calculate prop bet probabilities

**GIFT (Goal In First Ten):**
```python
LEAGUE_GIFT_RATE = 0.58  # ~58% of games
FIRST_10_MIN_FACTOR = 0.167  # 10/60 mins
```

**SOG (1+ Shot in First 2 Min):**
```python
LEAGUE_SOG_2MIN_RATE = 0.85  # ~85% of games
shots_factor = 2/60  # 2 mins of 60 min game
```

### 6. EdgeCalculator (`agents/edge_calculator.py`)

**Role:** Compare model predictions to betting lines

**Calculations:**
```python
# American odds to implied probability
def american_to_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)

# Kelly criterion (1/4 Kelly for safety)
kelly = (bp - q) / b * 0.25
```

**Outputs:**
- Edge % vs market line
- Kelly-optimal bet size
- EV per $100 wagered
- Confidence tier (STRONG/LEAN/PASS)

### 7. FeatureEngineer (`agents/feature_engineer.py`)

**Role:** Transform raw stats into model-ready features

**3-Year Weighting:**
```python
SEASON_WEIGHTS = {
    'current': 0.50,
    'prior_1': 0.30,
    'prior_2': 0.20
}
```

### 8. PDFExporter (`utils/pdf_export.py`)

**Role:** Generate downloadable PDF reports

**Methods:**
- `generate_game_pdf()` - Single game report
- `generate_overview_pdf()` - All daily games summary

## Project Structure

```
nhl-predictor/
├── agents/
│   ├── __init__.py
│   ├── data_ingestor.py      # NHL API, MoneyPuck integration
│   ├── poisson_engine.py     # Expected goals model
│   ├── monte_carlo.py        # 10K simulations
│   ├── period_model.py       # 1st period predictions
│   ├── props_model.py        # GIFT, SOG props
│   ├── edge_calculator.py    # Betting edge analysis
│   └── feature_engineer.py   # Feature weighting
├── utils/
│   ├── __init__.py
│   └── pdf_export.py         # PDF generation
├── models/
│   ├── __init__.py
│   └── schemas.py            # Pydantic models
├── api/
│   └── main.py               # FastAPI backend (optional)
├── streamlit_app.py          # Main UI
├── requirements.txt
├── ARCHITECTURE.md
├── REVISION_PLAN.md
└── README.md
```

## Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Frontend | Streamlit | Rapid development, Python-native |
| Hosting | Streamlit Cloud | Free, auto-deploy from GitHub |
| ML | SciPy, NumPy | Statistical computing |
| Data | Pandas | Data manipulation |
| PDF | ReportLab | PDF generation |
| HTTP | Requests | API calls |

## Caching Strategy

```python
@st.cache_data(ttl=300)   # 5 min - schedule
@st.cache_data(ttl=600)   # 10 min - L5, H2H
@st.cache_data(ttl=3600)  # 1 hour - team stats
```

## API Endpoints (Optional FastAPI)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/games` | GET | Today's schedule |
| `/api/predict` | POST | Run prediction |
| `/api/goalies` | GET | Goalie roster |
| `/api/odds` | GET | Betting lines |

## Future Enhancements

1. **ML Ensemble** - XGBoost/LightGBM training
2. **Live Odds** - the-odds-api.com integration
3. **Goalie Dropdown** - Selection with stats
4. **NST Scraping** - Advanced metrics
5. **Vercel Deployment** - React frontend option
