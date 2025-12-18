# NHL Game Prediction Model - System Architecture

## Overview

Multi-agent prediction system combining Poisson regression, Monte Carlo simulation, and gradient boosting ML for NHL game outcome and betting analysis. Deployed on Vercel with FastAPI backend.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VERCEL DEPLOYMENT                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐    ┌─────────────────────────────────────────────┐ │
│  │   Next.js Frontend  │    │          FastAPI Backend (Serverless)       │ │
│  │  ─────────────────  │    │  ─────────────────────────────────────────  │ │
│  │  • Game Dashboard   │◄──►│  /api/games      - Today's matchups         │ │
│  │  • Goalie Selector  │    │  /api/predict    - Run predictions          │ │
│  │  • Prediction Cards │    │  /api/odds       - Current betting lines    │ │
│  │  • Historical View  │    │  /api/teams      - Team stats lookup        │ │
│  └─────────────────────┘    │  /api/goalies    - Goalie dropdown data     │ │
│                             └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PREDICTION ENGINE (Python)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐ │
│  │    DATA      │   │   POISSON    │   │ MONTE CARLO  │   │     ML       │ │
│  │  INGESTOR    │──►│   ENGINE     │──►│  SIMULATOR   │──►│  ENSEMBLE    │ │
│  │              │   │              │   │              │   │              │ │
│  │ • NST API    │   │ • λ_home     │   │ • 10K sims   │   │ • XGBoost    │ │
│  │ • MoneyPuck  │   │ • λ_away     │   │ • Win prob   │   │ • LightGBM   │ │
│  │ • Hockey-Ref │   │ • Goalie adj │   │ • O/U prob   │   │ • Stacking   │ │
│  │ • Odds API   │   │ • Home ice   │   │ • Score dist │   │              │ │
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
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │  Supabase/      │  │   Redis Cache   │  │      S3/Vercel Blob         │ │
│  │  PostgreSQL     │  │                 │  │                             │ │
│  │  ─────────────  │  │  • Daily games  │  │  • Pickled models           │ │
│  │  • Team stats   │  │  • Live odds    │  │  • Historical predictions   │ │
│  │  • Goalie stats │  │  • Team cache   │  │  • Training datasets        │ │
│  │  • Game history │  │                 │  │                             │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Agent Specifications

### 1. DataIngestorAgent

**Role:** Fetch, validate, and normalize data from multiple NHL stat sources

**Inputs:**
- Date range (current season + 2 prior seasons)
- Team identifiers
- Goalie identifiers

**Outputs:**
- Normalized team stats DataFrame
- Goalie performance DataFrame
- Schedule/games DataFrame

**Data Sources:**
| Source | Stats | Method |
|--------|-------|--------|
| Natural Stat Trick | CF%, xGF, xGA, shots, scoring chances | Web scrape + CSV export |
| MoneyPuck | GSAX, xG, 60-min stats, goalie metrics | JSON API |
| Hockey-Reference | Historical standings, basic stats | Web scrape |
| The Odds API | Live lines, O/U, moneylines | REST API |

**Weighting Schema (3-year):**
```python
SEASON_WEIGHTS = {
    'current': 0.50,    # 2024-25 season
    'prior_1': 0.30,    # 2023-24 season
    'prior_2': 0.20     # 2022-23 season
}
```

### 2. FeatureEngineerAgent

**Role:** Transform raw stats into model-ready features

**Core Features (per team):**
| Feature | Description | Source |
|---------|-------------|--------|
| `adj_gf_60` | Adjusted goals for per 60 | NST |
| `adj_ga_60` | Adjusted goals against per 60 | NST |
| `xgf_pct` | Expected goals for % | MoneyPuck |
| `cf_pct` | Corsi for % (shot attempts) | NST |
| `scf_pct` | Scoring chances for % | NST |
| `pp_pct` | Power play conversion % | Hockey-Ref |
| `pk_pct` | Penalty kill % | Hockey-Ref |
| `sh_pct` | Shooting % (regression target) | NST |
| `sv_pct` | Team save % | MoneyPuck |
| `fo_pct` | Faceoff win % | NST |
| `gsax` | Goals saved above expected (goalie) | MoneyPuck |
| `es_sv_pct` | Even strength save % (goalie) | MoneyPuck |

**Derived Features:**
```python
# Goal differential metrics
'goal_diff_60': adj_gf_60 - adj_ga_60
'xg_diff_60': xgf_60 - xga_60

# Form indicators (last 10 games)
'l10_win_pct': wins in last 10 / 10
'l10_goal_diff': goal differential in last 10

# Home/Away splits
'home_gf_60': home-only goals for per 60
'away_gf_60': away-only goals for per 60

# Goalie-adjusted expected goals
'goalie_adj_xga': xga * (1 - gsax_factor)
```

### 3. PoissonEngineAgent

**Role:** Calculate expected goals using Poisson regression

**Model:**
```
λ_home = (home_attack × away_defense × home_ice_advantage × goalie_adj) × league_avg_goals
λ_away = (away_attack × home_defense × goalie_adj) × league_avg_goals
```

**Parameters:**
- `home_ice_advantage`: ~1.06 (6% boost historically)
- `league_avg_goals`: ~3.0 goals/team/game (2024-25)
- `goalie_adjustment`: Based on GSAX and SV% vs league average

**Outputs:**
- λ_home: Expected goals for home team
- λ_away: Expected goals for away team
- P(home win), P(away win), P(tie in regulation)

### 4. MonteCarloAgent

**Role:** Simulate games to generate probability distributions

**Configuration:**
```python
N_SIMULATIONS = 10_000
OVERTIME_PROB = 0.23  # ~23% of games go to OT historically
SHOOTOUT_PROB = 0.50  # Of OT games, ~50% go to shootout
```

**Simulation Logic:**
1. Draw home_goals from Poisson(λ_home)
2. Draw away_goals from Poisson(λ_away)
3. If tied, simulate OT/SO with adjusted probabilities
4. Track: final scores, total goals, winning team

**Outputs:**
- Win probability distribution
- Score probability matrix
- Over/Under probability for any total
- Most likely final scores
- 60-minute result probabilities (for 60-min moneyline)

### 5. MLEnsembleAgent

**Role:** Gradient boosting models for enhanced predictions

**Models:**
1. **XGBoost Classifier** - Win/Loss prediction
2. **LightGBM Regressor** - Total goals prediction
3. **Stacking Ensemble** - Meta-learner combining Poisson + ML

**Features for ML:**
```python
ML_FEATURES = [
    # Team strength
    'home_xgf_pct', 'away_xgf_pct',
    'home_cf_pct', 'away_cf_pct',

    # Special teams
    'home_pp_pct', 'away_pp_pct',
    'home_pk_pct', 'away_pk_pct',

    # Goalie
    'home_goalie_gsax', 'away_goalie_gsax',
    'home_goalie_sv_pct', 'away_goalie_sv_pct',

    # Form
    'home_l10_win_pct', 'away_l10_win_pct',

    # Rest/schedule
    'home_rest_days', 'away_rest_days',
    'home_b2b', 'away_b2b',  # back-to-back flag

    # Poisson baseline
    'poisson_home_win_prob',
    'poisson_expected_total'
]
```

### 6. EdgeCalculatorAgent

**Role:** Compare model predictions to betting lines for edge detection

**Calculations:**
```python
# Convert American odds to implied probability
def american_to_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)

# Expected value
ev = (win_prob * payout) - (loss_prob * stake)

# Kelly criterion for bet sizing
kelly_fraction = (bp - q) / b
# where b = decimal odds - 1, p = win prob, q = 1 - p
```

**Outputs:**
- Edge % vs market line
- Kelly-optimal bet size
- EV per $100 wagered
- Confidence tier (Strong/Moderate/Weak/No edge)

## API Endpoints

### GET /api/games
Returns today's NHL games with basic info.

### POST /api/predict
```json
{
  "home_team": "TOR",
  "away_team": "MTL",
  "home_goalie": "joseph-woll",
  "away_goalie": "sam-montembeault",
  "include_ml": true
}
```

Response:
```json
{
  "prediction": {
    "home_win_prob": 0.58,
    "away_win_prob": 0.42,
    "expected_total": 6.2,
    "over_prob": 0.54,
    "most_likely_score": "4-2",
    "sixty_min_home_prob": 0.52
  },
  "edge_analysis": {
    "moneyline_edge": 0.04,
    "over_edge": -0.02,
    "kelly_home_ml": 0.08,
    "recommendation": "LEAN HOME ML"
  },
  "model_details": {
    "poisson_lambda_home": 3.4,
    "poisson_lambda_away": 2.8,
    "ml_ensemble_prob": 0.61,
    "confidence": "MODERATE"
  }
}
```

### GET /api/goalies?team=TOR
Returns goalie roster with stats for dropdown selection.

### GET /api/odds
Returns current betting lines from The Odds API.

## Project Structure

```
nhl-predictor/
├── api/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app
│   ├── routes/
│   │   ├── games.py
│   │   ├── predict.py
│   │   ├── odds.py
│   │   └── teams.py
│   └── deps.py                 # Dependencies
├── agents/
│   ├── __init__.py
│   ├── data_ingestor.py
│   ├── feature_engineer.py
│   ├── poisson_engine.py
│   ├── monte_carlo.py
│   ├── ml_ensemble.py
│   └── edge_calculator.py
├── models/
│   ├── __init__.py
│   ├── schemas.py              # Pydantic models
│   └── trained/                # Pickled models
├── data/
│   ├── raw/                    # Scraped data
│   ├── processed/              # Feature-engineered
│   └── cache/                  # Daily cache
├── scrapers/
│   ├── natural_stat_trick.py
│   ├── moneypuck.py
│   ├── hockey_reference.py
│   └── odds_api.py
├── frontend/                   # Next.js app
│   ├── pages/
│   ├── components/
│   └── ...
├── scripts/
│   ├── daily_update.py         # Cron job for data refresh
│   └── train_models.py         # ML training pipeline
├── tests/
├── requirements.txt
├── vercel.json
└── README.md
```

## Technology Stack

| Layer | Technology | Rationale |
|-------|------------|-----------|
| Frontend | Next.js 14 | Vercel-native, SSR, React |
| Backend | FastAPI | Async, auto-docs, Pydantic |
| ML | XGBoost, LightGBM, scikit-learn | Industry standard, fast |
| Data | Pandas, NumPy, SciPy | Statistical computing |
| Database | Supabase PostgreSQL | Free tier, Vercel integration |
| Cache | Upstash Redis | Serverless Redis |
| Storage | Vercel Blob | Model pickle storage |
| Scraping | requests, BeautifulSoup, httpx | Reliable extraction |
| Deployment | Vercel | Serverless, edge functions |

## Data Refresh Strategy

**Daily Cron (via Vercel Cron or external):**
1. 10:00 AM ET - Fetch today's schedule
2. 10:05 AM ET - Update team stats (season-to-date)
3. 10:10 AM ET - Refresh goalie stats
4. 10:15 AM ET - Pull current betting odds
5. 10:20 AM ET - Pre-compute predictions for all games

**Real-time:**
- Odds refresh every 15 minutes during game day
- Goalie confirmations (morning skate ~11 AM ET)

## Model Validation

**Backtesting Protocol:**
1. Train on 2022-23 + 2023-24 seasons
2. Validate on 2024-25 season (out-of-sample)
3. Track: Brier score, log loss, calibration, ROI

**Key Metrics:**
- Win prediction accuracy (target: >55%)
- O/U prediction accuracy (target: >52%)
- Calibration: predicted prob vs actual frequency
- Simulated betting ROI with Kelly sizing
