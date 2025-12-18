"""
NHL Game Prediction Model - Streamlit App
Card-Style UI with Live Data Integration
"""
import streamlit as st
import pandas as pd
from datetime import date, datetime
from typing import Dict, List, Optional

# Import prediction agents
from agents.poisson_engine import PoissonEngine
from agents.monte_carlo import MonteCarloSimulator, MonteCarloConfig
from agents.edge_calculator import EdgeCalculator
from agents.period_model import PeriodModel
from agents.props_model import PropsModel
from agents.data_ingestor import DataIngestor

# PDF export (optional - will work without reportlab)
try:
    from utils.pdf_export import export_game_pdf, export_overview_pdf
    HAS_PDF_EXPORT = True
except ImportError:
    HAS_PDF_EXPORT = False

# Page config - Dark theme
st.set_page_config(
    page_title="NHL Predictions",
    page_icon="🏒",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for card-style dark theme
st.markdown("""
<style>
    /* Dark theme base */
    .stApp {
        background-color: #0e1117;
    }

    /* Game card styling */
    .game-card {
        background: linear-gradient(145deg, #1a1f2e 0%, #151922 100%);
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #2d3748;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .game-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }

    /* Team info */
    .team-name {
        font-size: 18px;
        font-weight: 600;
        color: #ffffff;
    }
    .team-code {
        font-size: 14px;
        color: #8b949e;
    }

    /* Win probability display */
    .win-prob {
        font-size: 32px;
        font-weight: 700;
        color: #58a6ff;
    }
    .win-prob-label {
        font-size: 11px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-b2b {
        background: #f85149;
        color: white;
    }
    .badge-home {
        background: #238636;
        color: white;
    }
    .badge-strong {
        background: #238636;
        color: white;
    }
    .badge-lean {
        background: #d29922;
        color: white;
    }
    .badge-pass {
        background: #484f58;
        color: #8b949e;
    }

    /* Section headers */
    .section-header {
        font-size: 13px;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 15px 0 10px 0;
        border-bottom: 1px solid #30363d;
        padding-bottom: 8px;
    }

    /* Metric cards */
    .metric-card {
        background: #21262d;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 11px;
        color: #8b949e;
        text-transform: uppercase;
    }

    /* Positive/negative indicators */
    .positive { color: #3fb950; }
    .negative { color: #f85149; }
    .neutral { color: #8b949e; }

    /* Best bet highlight */
    .best-bet {
        background: linear-gradient(145deg, #238636 0%, #1a7f32 100%);
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .best-bet-label {
        font-size: 11px;
        color: rgba(255,255,255,0.8);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .best-bet-value {
        font-size: 18px;
        font-weight: 700;
        color: white;
    }

    /* Game time */
    .game-time {
        font-size: 13px;
        color: #58a6ff;
        font-weight: 500;
    }

    /* Props section */
    .prop-item {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #30363d;
    }
    .prop-name {
        color: #c9d1d9;
        font-size: 14px;
    }
    .prop-value {
        color: #58a6ff;
        font-weight: 600;
        font-size: 14px;
    }

    /* Recent games table */
    .recent-game {
        display: flex;
        justify-content: space-between;
        padding: 6px 0;
        border-bottom: 1px solid #21262d;
        font-size: 13px;
    }
    .game-result-w { color: #3fb950; font-weight: 600; }
    .game-result-l { color: #f85149; font-weight: 600; }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #21262d !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# Team name mapping
TEAM_NAMES = {
    "ANA": "Ducks", "BOS": "Bruins", "BUF": "Sabres", "CGY": "Flames",
    "CAR": "Hurricanes", "CHI": "Blackhawks", "COL": "Avalanche", "CBJ": "Blue Jackets",
    "DAL": "Stars", "DET": "Red Wings", "EDM": "Oilers", "FLA": "Panthers",
    "LAK": "Kings", "MIN": "Wild", "MTL": "Canadiens", "NSH": "Predators",
    "NJD": "Devils", "NYI": "Islanders", "NYR": "Rangers", "OTT": "Senators",
    "PHI": "Flyers", "PIT": "Penguins", "SJS": "Sharks", "SEA": "Kraken",
    "STL": "Blues", "TBL": "Lightning", "TOR": "Maple Leafs", "UTA": "Utah HC",
    "VAN": "Canucks", "VGK": "Golden Knights", "WSH": "Capitals", "WPG": "Jets",
}

# Fallback team stats (used when API unavailable)
FALLBACK_STATS = {
    "ANA": {"gf_60": 2.65, "ga_60": 3.35, "shots_60": 28.5},
    "BOS": {"gf_60": 3.15, "ga_60": 2.75, "shots_60": 32.0},
    "BUF": {"gf_60": 2.95, "ga_60": 3.20, "shots_60": 30.5},
    "CGY": {"gf_60": 2.80, "ga_60": 2.95, "shots_60": 29.5},
    "CAR": {"gf_60": 3.35, "ga_60": 2.65, "shots_60": 34.0},
    "CHI": {"gf_60": 2.55, "ga_60": 3.45, "shots_60": 27.0},
    "COL": {"gf_60": 3.45, "ga_60": 2.85, "shots_60": 33.5},
    "CBJ": {"gf_60": 2.70, "ga_60": 3.40, "shots_60": 28.0},
    "DAL": {"gf_60": 3.25, "ga_60": 2.70, "shots_60": 31.5},
    "DET": {"gf_60": 2.90, "ga_60": 3.10, "shots_60": 29.0},
    "EDM": {"gf_60": 3.55, "ga_60": 3.00, "shots_60": 32.5},
    "FLA": {"gf_60": 3.40, "ga_60": 2.75, "shots_60": 33.0},
    "LAK": {"gf_60": 3.05, "ga_60": 2.80, "shots_60": 31.0},
    "MIN": {"gf_60": 3.00, "ga_60": 2.90, "shots_60": 30.0},
    "MTL": {"gf_60": 2.85, "ga_60": 3.25, "shots_60": 29.5},
    "NSH": {"gf_60": 2.75, "ga_60": 3.05, "shots_60": 28.5},
    "NJD": {"gf_60": 3.20, "ga_60": 2.95, "shots_60": 31.5},
    "NYI": {"gf_60": 2.80, "ga_60": 2.85, "shots_60": 29.0},
    "NYR": {"gf_60": 3.10, "ga_60": 2.70, "shots_60": 30.5},
    "OTT": {"gf_60": 2.95, "ga_60": 3.15, "shots_60": 30.0},
    "PHI": {"gf_60": 2.85, "ga_60": 3.20, "shots_60": 29.5},
    "PIT": {"gf_60": 3.00, "ga_60": 3.10, "shots_60": 30.5},
    "SJS": {"gf_60": 2.50, "ga_60": 3.50, "shots_60": 27.5},
    "SEA": {"gf_60": 2.90, "ga_60": 3.00, "shots_60": 29.5},
    "STL": {"gf_60": 2.85, "ga_60": 3.15, "shots_60": 29.0},
    "TBL": {"gf_60": 3.30, "ga_60": 2.90, "shots_60": 32.0},
    "TOR": {"gf_60": 3.35, "ga_60": 2.85, "shots_60": 33.0},
    "UTA": {"gf_60": 2.80, "ga_60": 3.10, "shots_60": 29.0},
    "VAN": {"gf_60": 3.20, "ga_60": 2.80, "shots_60": 31.5},
    "VGK": {"gf_60": 3.25, "ga_60": 2.75, "shots_60": 32.0},
    "WSH": {"gf_60": 3.15, "ga_60": 2.90, "shots_60": 30.5},
    "WPG": {"gf_60": 3.40, "ga_60": 2.65, "shots_60": 32.5},
}


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_todays_games():
    """Fetch today's games from NHL API."""
    try:
        ingestor = DataIngestor()
        games = ingestor.fetch_schedule()
        return games
    except Exception as e:
        st.warning(f"Could not fetch live schedule: {e}")
        return []


@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_team_stats(team_code: str) -> Dict:
    """Get team stats (live or fallback)."""
    return FALLBACK_STATS.get(team_code, {"gf_60": 3.0, "ga_60": 3.0, "shots_60": 30.0})


@st.cache_data(ttl=300)
def check_back_to_back(team_code: str) -> tuple:
    """Check if team is on back-to-back."""
    try:
        ingestor = DataIngestor()
        is_b2b, prev_opponent = ingestor.is_back_to_back(team_code)
        return is_b2b, prev_opponent
    except Exception:
        return False, None


@st.cache_data(ttl=600)
def fetch_last_5_games(team_code: str) -> List[Dict]:
    """Fetch last 5 games for a team."""
    try:
        ingestor = DataIngestor()
        return ingestor.fetch_last_n_games(team_code, n=5)
    except Exception:
        return []


@st.cache_data(ttl=600)
def fetch_h2h_games(team1: str, team2: str) -> List[Dict]:
    """Fetch head-to-head history."""
    try:
        ingestor = DataIngestor()
        return ingestor.fetch_h2h_games(team1, team2, n=5)
    except Exception:
        return []


def run_full_prediction(home_code: str, away_code: str, ou_line: float = 6.0) -> Dict:
    """Run complete prediction pipeline for a game."""
    home_stats = get_team_stats(home_code)
    away_stats = get_team_stats(away_code)

    # Initialize models
    poisson = PoissonEngine()
    mc = MonteCarloSimulator(MonteCarloConfig(n_simulations=10000))
    edge_calc = EdgeCalculator()
    period_model = PeriodModel()
    props_model = PropsModel()

    # Run Poisson model
    poisson_result = poisson.predict(
        home_stats={'gf_60': home_stats['gf_60'], 'ga_60': home_stats['ga_60']},
        away_stats={'gf_60': away_stats['gf_60'], 'ga_60': away_stats['ga_60']},
        over_under_line=ou_line,
    )

    # Run Monte Carlo
    mc_result = mc.simulate_games(
        poisson_result['lambda_home'],
        poisson_result['lambda_away']
    )

    # O/U analysis
    ou_result = mc.calculate_over_under(
        poisson_result['lambda_home'],
        poisson_result['lambda_away'],
        [ou_line - 0.5, ou_line, ou_line + 0.5]
    )

    # First period prediction
    p1_result = period_model.predict_first_period(
        poisson_result['lambda_home'],
        poisson_result['lambda_away']
    )

    # Props calculation
    props_result = props_model.calculate_all_props(
        home_stats={'gf_60': home_stats['gf_60'], 'ga_60': home_stats['ga_60'],
                    'shots_60': home_stats.get('shots_60', 30.0), 'team': home_code},
        away_stats={'gf_60': away_stats['gf_60'], 'ga_60': away_stats['ga_60'],
                    'shots_60': away_stats.get('shots_60', 30.0), 'team': away_code},
    )

    # Edge analysis (using placeholder odds)
    edge_result = edge_calc.full_game_analysis(
        home_win_prob=mc_result['final']['home_win_prob'],
        over_prob=ou_result.get(ou_line, {}).get('over_prob', 0.5),
        home_ml=-150,  # Placeholder
        away_ml=130,   # Placeholder
        total_line=ou_line,
    )

    return {
        'home_team': {'code': home_code, 'name': TEAM_NAMES.get(home_code, home_code)},
        'away_team': {'code': away_code, 'name': TEAM_NAMES.get(away_code, away_code)},
        'poisson': poisson_result,
        'monte_carlo': mc_result,
        'over_under': ou_result,
        'first_period': p1_result,
        'props': props_result,
        'edge': edge_result,
        'odds': {'over_under': ou_line},
    }


def render_game_card(game: Dict, prediction: Dict, expanded: bool = False):
    """Render a single game card."""
    home = prediction['home_team']
    away = prediction['away_team']
    mc = prediction['monte_carlo']['final']

    # Check for B2B using DataIngestor
    home_b2b, home_prev = check_back_to_back(home['code'])
    away_b2b, away_prev = check_back_to_back(away['code'])

    # Main card container
    with st.container():
        # Header row: Teams and time
        cols = st.columns([3, 2, 3])

        with cols[0]:
            st.markdown(f"""
                <div style='text-align: center;'>
                    <div class='team-code'>{away['code']}</div>
                    <div class='team-name'>{away['name']}</div>
                    {'<span class="badge badge-b2b">B2B</span>' if away_b2b else ''}
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
                <div style='text-align: center;'>
                    <div class='win-prob'>{mc['away_win_prob']*100:.0f}%</div>
                    <div class='win-prob-label'>Win Prob</div>
                </div>
            """, unsafe_allow_html=True)

        with cols[1]:
            st.markdown("""
                <div style='text-align: center; padding-top: 20px;'>
                    <div style='font-size: 24px; color: #8b949e;'>@</div>
                </div>
            """, unsafe_allow_html=True)
            # Expected total
            exp_total = prediction['poisson']['expected_total']
            st.markdown(f"""
                <div style='text-align: center; margin-top: 10px;'>
                    <div style='font-size: 14px; color: #8b949e;'>Total: {exp_total:.1f}</div>
                </div>
            """, unsafe_allow_html=True)

        with cols[2]:
            st.markdown(f"""
                <div style='text-align: center;'>
                    <div class='team-code'>{home['code']}</div>
                    <div class='team-name'>{home['name']}</div>
                    <span class="badge badge-home">HOME</span>
                    {'<span class="badge badge-b2b">B2B</span>' if home_b2b else ''}
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
                <div style='text-align: center;'>
                    <div class='win-prob'>{mc['home_win_prob']*100:.0f}%</div>
                    <div class='win-prob-label'>Win Prob</div>
                </div>
            """, unsafe_allow_html=True)

        # Best bet highlight
        top_plays = prediction['edge'].get('top_plays', [])
        if top_plays:
            best = top_plays[0]
            edge_pct = best['edge'] * 100
            st.markdown(f"""
                <div class='best-bet'>
                    <div class='best-bet-label'>Best Bet</div>
                    <div class='best-bet-value'>{best['type'].upper()} {best['side'].upper()} ({edge_pct:+.1f}% edge)</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")


def render_game_details(prediction: Dict):
    """Render expanded game details."""
    home = prediction['home_team']
    away = prediction['away_team']

    # PDF export button at top
    if HAS_PDF_EXPORT:
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            try:
                pdf_bytes = export_game_pdf(prediction)
                filename = f"{away['code']}_at_{home['code']}_{date.today().isoformat()}.pdf"
                st.download_button(
                    label="Download PDF",
                    data=pdf_bytes,
                    file_name=filename,
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"PDF export unavailable: {e}")

    # Tabs for different analysis sections
    tabs = st.tabs(["Prediction", "1st Period", "Props", "Recent Form"])

    with tabs[0]:
        # Main prediction details
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Expected Goals**")
            st.metric(f"{home['code']} (Home)", f"{prediction['poisson']['lambda_home']:.2f}")
            st.metric(f"{away['code']} (Away)", f"{prediction['poisson']['lambda_away']:.2f}")
            st.metric("Total", f"{prediction['poisson']['expected_total']:.1f}")

        with col2:
            st.markdown("**Score Prediction**")
            st.write(f"Most Likely: **{prediction['monte_carlo']['scores']['most_likely']}**")
            st.write(f"OT Probability: {prediction['monte_carlo']['meta']['overtime_games']*100:.1f}%")

            st.markdown("**60-Min Result:**")
            reg = prediction['monte_carlo']['sixty_min']
            st.write(f"- {home['code']} Win: {reg['home_win_prob']*100:.1f}%")
            st.write(f"- {away['code']} Win: {reg['away_win_prob']*100:.1f}%")
            st.write(f"- Tie: {reg['tie_prob']*100:.1f}%")

    with tabs[1]:
        # First period analysis
        p1 = prediction.get('first_period', {})
        if p1:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**1st Period Expected Goals**")
                st.metric(f"{home['code']}", f"{p1.get('lambda_home', 0):.2f}")
                st.metric(f"{away['code']}", f"{p1.get('lambda_away', 0):.2f}")
                st.metric("Total", f"{p1.get('expected_total', 0):.2f}")

            with col2:
                st.markdown("**1st Period O/U**")
                ou = p1.get('over_under', {})
                for line in [0.5, 1.5, 2.5]:
                    line_data = ou.get(line, {})
                    over_prob = line_data.get('over_prob', 0) * 100
                    st.write(f"Over {line}: **{over_prob:.1f}%**")

                st.markdown("**Most Likely Score**")
                st.write(f"**{p1.get('most_likely_score', 'N/A')}**")

    with tabs[2]:
        # Props analysis
        props = prediction.get('props', {})
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**GIFT (Goal In First 10)**")
            gift = props.get('gift', {})
            gift_prob = gift.get('gift_prob', 0) * 100
            vs_league = gift.get('vs_league', 0) * 100

            st.metric("Probability", f"{gift_prob:.1f}%")
            st.write(f"League Avg: 58%")
            color = "positive" if vs_league > 0 else "negative"
            st.markdown(f"vs League: <span class='{color}'>{vs_league:+.1f}%</span>", unsafe_allow_html=True)

        with col2:
            st.markdown("**1+ SOG in First 2 Min**")
            sog = props.get('sog_2min', {})
            sog_prob = sog.get('either_team_sog_prob', 0) * 100

            st.metric("Either Team", f"{sog_prob:.1f}%")
            st.write(f"{home['code']}: {sog.get('home_sog_prob', 0)*100:.1f}%")
            st.write(f"{away['code']}: {sog.get('away_sog_prob', 0)*100:.1f}%")

    with tabs[3]:
        # Recent form with live data
        st.markdown("**Last 5 Games**")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{home['code']}**")
            home_l5 = fetch_last_5_games(home['code'])
            if home_l5:
                wins = sum(1 for g in home_l5 if g.get('result', '').startswith('W'))
                losses = len(home_l5) - wins
                results_str = ' '.join([g.get('result', '?')[0] for g in home_l5])
                st.write(f"{results_str} ({wins}-{losses})")
                for g in home_l5[:3]:
                    opponent = g.get('opponent', '???')
                    score = g.get('score', '0-0')
                    result = g.get('result', '?')
                    color = "game-result-w" if result.startswith('W') else "game-result-l"
                    st.markdown(f"<span class='{color}'>{result}</span> vs {opponent} ({score})", unsafe_allow_html=True)
            else:
                st.write("No recent games available")

        with col2:
            st.markdown(f"**{away['code']}**")
            away_l5 = fetch_last_5_games(away['code'])
            if away_l5:
                wins = sum(1 for g in away_l5 if g.get('result', '').startswith('W'))
                losses = len(away_l5) - wins
                results_str = ' '.join([g.get('result', '?')[0] for g in away_l5])
                st.write(f"{results_str} ({wins}-{losses})")
                for g in away_l5[:3]:
                    opponent = g.get('opponent', '???')
                    score = g.get('score', '0-0')
                    result = g.get('result', '?')
                    color = "game-result-w" if result.startswith('W') else "game-result-l"
                    st.markdown(f"<span class='{color}'>{result}</span> vs {opponent} ({score})", unsafe_allow_html=True)
            else:
                st.write("No recent games available")

        # Head-to-Head
        st.markdown("---")
        st.markdown("**Head-to-Head (Last 5)**")
        h2h = fetch_h2h_games(home['code'], away['code'])
        if h2h:
            for g in h2h[:5]:
                game_date = g.get('date', 'Unknown')
                score = g.get('score', '0-0')
                winner = g.get('winner', '???')
                st.write(f"{game_date}: {score} - Winner: **{winner}**")
        else:
            st.write("No H2H data available")


# =============================================================================
# MAIN APP
# =============================================================================

st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #58a6ff; margin: 0;'>NHL Predictions</h1>
        <p style='color: #8b949e; font-size: 14px;'>Poisson + Monte Carlo + ML Ensemble</p>
    </div>
""", unsafe_allow_html=True)

# Date selector
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    selected_date = st.date_input("Select Date", value=date.today())

st.markdown("---")

# Fetch games for selected date
games = fetch_todays_games()

if not games:
    st.info("No games scheduled for today. Showing demo predictions.")
    # Demo games
    demo_matchups = [
        ("TOR", "MTL"),
        ("EDM", "VGK"),
        ("COL", "DAL"),
    ]

    for home, away in demo_matchups:
        prediction = run_full_prediction(home, away)

        with st.expander(f"{away} @ {home}", expanded=False):
            render_game_card({}, prediction)
            render_game_details(prediction)

else:
    st.markdown(f"### {len(games)} Games Today")

    for game in games:
        home_code = game.get('home_team', 'UNK')
        away_code = game.get('away_team', 'UNK')
        game_time = game.get('start_time', '')

        # Run prediction
        prediction = run_full_prediction(home_code, away_code)

        # Add game time to prediction
        prediction['game_time'] = game_time

        with st.expander(f"{away_code} @ {home_code} - {game_time}", expanded=False):
            render_game_card(game, prediction)
            render_game_details(prediction)

# Sidebar for manual game entry
with st.sidebar:
    st.markdown("### Manual Game Entry")

    teams = list(TEAM_NAMES.keys())

    away_team = st.selectbox("Away Team", teams, index=teams.index("MTL"))
    home_team = st.selectbox("Home Team", teams, index=teams.index("TOR"))
    ou_line = st.number_input("O/U Line", value=6.0, step=0.5)

    if st.button("Run Prediction", type="primary", use_container_width=True):
        with st.spinner("Running 10,000 simulations..."):
            manual_pred = run_full_prediction(home_team, away_team, ou_line)
            st.session_state['manual_prediction'] = manual_pred

    if 'manual_prediction' in st.session_state:
        st.markdown("---")
        pred = st.session_state['manual_prediction']
        mc = pred['monte_carlo']['final']

        st.markdown(f"**{pred['away_team']['code']} @ {pred['home_team']['code']}**")
        st.write(f"Home Win: {mc['home_win_prob']*100:.1f}%")
        st.write(f"Away Win: {mc['away_win_prob']*100:.1f}%")
        st.write(f"Expected Total: {pred['poisson']['expected_total']:.1f}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #8b949e; font-size: 12px; padding: 20px 0;'>
        NHL Prediction Model v2.0 | Data: NHL API, MoneyPuck, Natural Stat Trick
    </div>
""", unsafe_allow_html=True)
