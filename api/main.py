"""
NHL Prediction Model - FastAPI Backend

Main API server for Vercel deployment.
Exposes endpoints for game predictions, team stats, and betting analysis.
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import date
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.poisson_engine import PoissonEngine
from agents.monte_carlo import MonteCarloSimulator, MonteCarloConfig
from agents.feature_engineer import FeatureEngineer, SeasonWeights
from agents.edge_calculator import EdgeCalculator
from agents.data_ingestor import DataIngestor
from agents.ml_ensemble import quick_ml_adjustment

# Initialize FastAPI app
app = FastAPI(
    title="NHL Prediction Model API",
    description="Poisson + Monte Carlo + ML ensemble for NHL game predictions",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize agents
poisson = PoissonEngine()
monte_carlo = MonteCarloSimulator(MonteCarloConfig(n_simulations=10000))
feature_eng = FeatureEngineer(SeasonWeights())
edge_calc = EdgeCalculator()
data_ingestor = DataIngestor()


# ============================================================================
# Request/Response Models
# ============================================================================

class PredictionRequest(BaseModel):
    home_team: str
    away_team: str
    home_goalie: Optional[str] = None
    away_goalie: Optional[str] = None
    include_ml: bool = True


class PredictionResponse(BaseModel):
    game: Dict
    poisson: Dict
    monte_carlo: Dict
    edge_analysis: Optional[Dict] = None
    ml_adjustment: Optional[Dict] = None
    confidence: str


class TeamStatsResponse(BaseModel):
    team: str
    stats: Dict
    goalies: List[Dict]


class GamesResponse(BaseModel):
    date: str
    games: List[Dict]


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "NHL Prediction Model",
        "version": "1.0.0",
    }


@app.get("/api/games", response_model=GamesResponse)
async def get_todays_games():
    """Get today's NHL games."""
    games = data_ingestor.fetch_todays_games()
    return {
        "date": date.today().isoformat(),
        "games": games,
    }


@app.get("/api/teams/{team_code}")
async def get_team_stats(team_code: str):
    """Get team statistics."""
    team_code = team_code.upper()

    # Validate team code
    if team_code not in data_ingestor.TEAM_MAPPING:
        raise HTTPException(status_code=404, detail=f"Team {team_code} not found")

    # Fetch team data
    team_data = data_ingestor.get_team_data(team_code)

    return {
        "team": team_code,
        "stats": team_data,
    }


@app.get("/api/goalies/{team_code}")
async def get_team_goalies(team_code: str):
    """Get goalies for a team (for dropdown selection)."""
    team_code = team_code.upper()

    goalie_df = data_ingestor.fetch_moneypuck_goalie_stats()
    team_goalies = goalie_df[goalie_df['team'] == team_code]

    if team_goalies.empty:
        # Return sample data if no real data
        return {
            "team": team_code,
            "goalies": [
                {"name": "Starter", "sv_pct": 0.910, "gsax": 5.0},
                {"name": "Backup", "sv_pct": 0.895, "gsax": -2.0},
            ]
        }

    return {
        "team": team_code,
        "goalies": team_goalies.to_dict('records'),
    }


@app.get("/api/odds")
async def get_betting_odds():
    """Get current betting odds for today's games."""
    odds = data_ingestor.fetch_betting_odds()
    return {"odds": odds}


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_game(request: PredictionRequest):
    """
    Run full prediction for a game.

    Returns Poisson model output, Monte Carlo simulation results,
    ML adjustments, and betting edge analysis.
    """
    home_team = request.home_team.upper()
    away_team = request.away_team.upper()

    # Fetch all game data
    game_data = data_ingestor.get_full_game_data(
        home_team, away_team,
        request.home_goalie, request.away_goalie
    )

    # Extract stats (use current season or default)
    home_stats = game_data['home_team']['stats'].get('2024-25', {})
    away_stats = game_data['away_team']['stats'].get('2024-25', {})
    home_goalie = game_data['home_team']['goalie']
    away_goalie = game_data['away_team']['goalie']
    odds = game_data['odds']

    # Default values if data missing
    home_gf = home_stats.get('gf_60', 3.1)
    home_ga = home_stats.get('ga_60', 2.9)
    away_gf = away_stats.get('gf_60', 3.0)
    away_ga = away_stats.get('ga_60', 3.0)

    # Run Poisson model
    poisson_result = poisson.predict(
        home_stats={'gf_60': home_gf, 'ga_60': home_ga},
        away_stats={'gf_60': away_gf, 'ga_60': away_ga},
        home_goalie={
            'gsax': home_goalie.get('gsax', 0),
            'sv_pct': home_goalie.get('sv_pct', 0.905),
        } if home_goalie else None,
        away_goalie={
            'gsax': away_goalie.get('gsax', 0),
            'sv_pct': away_goalie.get('sv_pct', 0.905),
        } if away_goalie else None,
        over_under_line=odds.get('over_under', 6.0),
    )

    # Run Monte Carlo simulation
    mc_result = monte_carlo.simulate_games(
        poisson_result['lambda_home'],
        poisson_result['lambda_away']
    )

    # Calculate over/under for different lines
    ou_analysis = monte_carlo.calculate_over_under(
        poisson_result['lambda_home'],
        poisson_result['lambda_away'],
        [5.5, 6.0, 6.5, 7.0]
    )
    mc_result['over_under_analysis'] = ou_analysis

    # ML adjustment (if requested)
    ml_result = None
    if request.include_ml:
        # Quick heuristic adjustment (replace with trained model in production)
        home_xgf = home_stats.get('xgf_60', home_gf)
        away_xgf = away_stats.get('xgf_60', away_gf)
        home_cf = home_stats.get('cf_pct', 50)
        away_cf = away_stats.get('cf_pct', 50)

        adjusted_prob = quick_ml_adjustment(
            mc_result['final']['home_win_prob'],
            home_xgf, away_xgf,
            home_cf, away_cf
        )

        ml_result = {
            'adjusted_home_win_prob': round(adjusted_prob, 4),
            'adjustment': round(adjusted_prob - mc_result['final']['home_win_prob'], 4),
            'note': 'Heuristic adjustment based on xG and Corsi differentials',
        }

    # Edge analysis
    edge_result = None
    if odds.get('home_ml') and odds.get('away_ml'):
        final_prob = ml_result['adjusted_home_win_prob'] if ml_result else mc_result['final']['home_win_prob']
        over_prob = mc_result['final'].get('home_win_prob', 0.5)  # Placeholder

        # Get actual over prob from O/U analysis
        ou_line = odds.get('over_under', 6.0)
        if ou_line in ou_analysis:
            over_prob = ou_analysis[ou_line]['over_prob']
        else:
            over_prob = ou_analysis.get(6.0, {}).get('over_prob', 0.5)

        edge_result = edge_calc.full_game_analysis(
            home_win_prob=final_prob,
            over_prob=over_prob,
            home_ml=odds['home_ml'],
            away_ml=odds['away_ml'],
            total_line=ou_line,
            over_odds=odds.get('over_odds', -110),
            under_odds=odds.get('under_odds', -110),
        )

    # Determine confidence
    prob_diff = abs(mc_result['final']['home_win_prob'] - 0.5)
    if prob_diff > 0.15:
        confidence = "HIGH"
    elif prob_diff > 0.08:
        confidence = "MODERATE"
    else:
        confidence = "LOW"

    return {
        "game": {
            "home_team": home_team,
            "away_team": away_team,
            "home_goalie": request.home_goalie or "Starter",
            "away_goalie": request.away_goalie or "Starter",
            "date": date.today().isoformat(),
        },
        "poisson": poisson_result,
        "monte_carlo": mc_result,
        "ml_adjustment": ml_result,
        "edge_analysis": edge_result,
        "confidence": confidence,
    }


@app.get("/api/predict/quick")
async def quick_predict(
    home: str = Query(..., description="Home team code (e.g., TOR)"),
    away: str = Query(..., description="Away team code (e.g., MTL)"),
):
    """
    Quick prediction with minimal input.

    Uses default stats and no goalie adjustments for fast results.
    """
    # Use sample/cached data for speed
    team_stats = data_ingestor._get_sample_team_stats()

    home_row = team_stats[team_stats['team'] == home.upper()]
    away_row = team_stats[team_stats['team'] == away.upper()]

    if home_row.empty or away_row.empty:
        raise HTTPException(status_code=404, detail="Team not found")

    home_stats = home_row.iloc[0]
    away_stats = away_row.iloc[0]

    # Run quick Poisson prediction
    poisson_result = poisson.predict(
        home_stats={'gf_60': home_stats['gf_60'], 'ga_60': home_stats['ga_60']},
        away_stats={'gf_60': away_stats['gf_60'], 'ga_60': away_stats['ga_60']},
    )

    return {
        "home_team": home.upper(),
        "away_team": away.upper(),
        "home_win_prob": poisson_result['home_win_prob'],
        "away_win_prob": poisson_result['away_win_prob'],
        "expected_total": poisson_result['expected_total'],
        "most_likely_score": poisson_result['most_likely_score'],
    }


# ============================================================================
# Vercel serverless handler
# ============================================================================

# For Vercel deployment
handler = app
