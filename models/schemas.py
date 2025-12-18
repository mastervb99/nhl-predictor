"""
Pydantic schemas for NHL Prediction Model
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import date
from enum import Enum


class TeamCode(str, Enum):
    """NHL Team abbreviations"""
    ANA = "ANA"  # Anaheim Ducks
    ARI = "ARI"  # Arizona Coyotes (Utah now)
    BOS = "BOS"  # Boston Bruins
    BUF = "BUF"  # Buffalo Sabres
    CGY = "CGY"  # Calgary Flames
    CAR = "CAR"  # Carolina Hurricanes
    CHI = "CHI"  # Chicago Blackhawks
    COL = "COL"  # Colorado Avalanche
    CBJ = "CBJ"  # Columbus Blue Jackets
    DAL = "DAL"  # Dallas Stars
    DET = "DET"  # Detroit Red Wings
    EDM = "EDM"  # Edmonton Oilers
    FLA = "FLA"  # Florida Panthers
    LAK = "LAK"  # Los Angeles Kings
    MIN = "MIN"  # Minnesota Wild
    MTL = "MTL"  # Montreal Canadiens
    NSH = "NSH"  # Nashville Predators
    NJD = "NJD"  # New Jersey Devils
    NYI = "NYI"  # New York Islanders
    NYR = "NYR"  # New York Rangers
    OTT = "OTT"  # Ottawa Senators
    PHI = "PHI"  # Philadelphia Flyers
    PIT = "PIT"  # Pittsburgh Penguins
    SJS = "SJS"  # San Jose Sharks
    SEA = "SEA"  # Seattle Kraken
    STL = "STL"  # St. Louis Blues
    TBL = "TBL"  # Tampa Bay Lightning
    TOR = "TOR"  # Toronto Maple Leafs
    UTA = "UTA"  # Utah Hockey Club
    VAN = "VAN"  # Vancouver Canucks
    VGK = "VGK"  # Vegas Golden Knights
    WSH = "WSH"  # Washington Capitals
    WPG = "WPG"  # Winnipeg Jets


class TeamStats(BaseModel):
    """Team statistics for a season"""
    team: str
    season: str
    games_played: int

    # Scoring
    goals_for: float
    goals_against: float
    goals_for_60: float
    goals_against_60: float

    # Advanced metrics
    xgf: float = Field(description="Expected goals for")
    xga: float = Field(description="Expected goals against")
    xgf_pct: float = Field(description="Expected goals for %")
    cf_pct: float = Field(description="Corsi for %")
    scf_pct: float = Field(description="Scoring chances for %")

    # Special teams
    pp_pct: float = Field(description="Power play %")
    pk_pct: float = Field(description="Penalty kill %")

    # Other
    sh_pct: float = Field(description="Shooting %")
    sv_pct: float = Field(description="Save %")
    fo_pct: float = Field(description="Faceoff win %")

    # Home/Away splits (optional)
    home_gf_60: Optional[float] = None
    home_ga_60: Optional[float] = None
    away_gf_60: Optional[float] = None
    away_ga_60: Optional[float] = None


class GoalieStats(BaseModel):
    """Goalie statistics"""
    name: str
    team: str
    season: str
    games_played: int

    sv_pct: float = Field(description="Save percentage")
    gaa: float = Field(description="Goals against average")
    gsax: float = Field(description="Goals saved above expected")
    es_sv_pct: float = Field(description="Even strength save %")

    # MoneyPuck specific
    gsax_60: Optional[float] = None
    xga: Optional[float] = None


class GameInfo(BaseModel):
    """Game information"""
    game_id: str
    date: date
    home_team: str
    away_team: str
    start_time: Optional[str] = None
    venue: Optional[str] = None


class BettingOdds(BaseModel):
    """Betting odds for a game"""
    game_id: str
    home_ml: int = Field(description="Home moneyline (American odds)")
    away_ml: int = Field(description="Away moneyline (American odds)")
    over_under: float = Field(description="Total goals line")
    over_odds: int = Field(default=-110)
    under_odds: int = Field(default=-110)
    home_puck_line: float = Field(default=-1.5)
    away_puck_line: float = Field(default=1.5)
    sixty_min_home_ml: Optional[int] = None
    sixty_min_away_ml: Optional[int] = None


class PredictionRequest(BaseModel):
    """Request for game prediction"""
    home_team: str
    away_team: str
    home_goalie: Optional[str] = None
    away_goalie: Optional[str] = None
    include_ml: bool = Field(default=True, description="Include ML ensemble prediction")


class PoissonResult(BaseModel):
    """Poisson model output"""
    lambda_home: float
    lambda_away: float
    home_win_prob: float
    away_win_prob: float
    tie_prob: float


class MonteCarloResult(BaseModel):
    """Monte Carlo simulation output"""
    n_simulations: int
    home_win_prob: float
    away_win_prob: float
    expected_total: float
    over_prob: float
    under_prob: float
    most_likely_score: str
    score_distribution: dict
    sixty_min_home_prob: float
    sixty_min_away_prob: float
    sixty_min_tie_prob: float


class EdgeAnalysis(BaseModel):
    """Betting edge analysis"""
    moneyline_edge: float = Field(description="Edge vs market home ML")
    over_edge: float = Field(description="Edge vs market O/U")
    kelly_home_ml: float = Field(description="Kelly fraction for home ML")
    kelly_over: float = Field(description="Kelly fraction for over")
    ev_home_ml: float = Field(description="EV per $100 on home ML")
    ev_over: float = Field(description="EV per $100 on over")
    recommendation: str


class PredictionResponse(BaseModel):
    """Full prediction response"""
    game: GameInfo
    prediction: MonteCarloResult
    poisson: PoissonResult
    ml_prediction: Optional[dict] = None
    edge_analysis: Optional[EdgeAnalysis] = None
    confidence: str = Field(description="STRONG/MODERATE/WEAK")
