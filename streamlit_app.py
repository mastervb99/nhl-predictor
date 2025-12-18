"""
NHL Game Prediction Model - Streamlit App
Poisson + Monte Carlo + ML Ensemble
"""
import streamlit as st
import numpy as np
import pandas as pd
from datetime import date

# Import prediction agents
from agents.poisson_engine import PoissonEngine
from agents.monte_carlo import MonteCarloSimulator, MonteCarloConfig
from agents.edge_calculator import EdgeCalculator

# Page config
st.set_page_config(
    page_title="NHL Prediction Model",
    page_icon="🏒",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .big-number {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
        text-align: center;
    }
    .edge-positive {
        color: #28a745;
        font-weight: bold;
    }
    .edge-negative {
        color: #dc3545;
    }
    .recommendation-strong {
        background-color: #28a745;
        color: white;
        padding: 8px 16px;
        border-radius: 4px;
        font-weight: bold;
    }
    .recommendation-lean {
        background-color: #ffc107;
        color: black;
        padding: 8px 16px;
        border-radius: 4px;
    }
    .recommendation-pass {
        background-color: #6c757d;
        color: white;
        padding: 8px 16px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Team data
TEAMS = {
    "ANA": {"name": "Anaheim Ducks", "gf_60": 2.65, "ga_60": 3.35, "xgf_60": 2.70, "cf_pct": 47.2},
    "BOS": {"name": "Boston Bruins", "gf_60": 3.15, "ga_60": 2.75, "xgf_60": 3.20, "cf_pct": 52.1},
    "BUF": {"name": "Buffalo Sabres", "gf_60": 2.95, "ga_60": 3.20, "xgf_60": 2.90, "cf_pct": 48.5},
    "CGY": {"name": "Calgary Flames", "gf_60": 2.80, "ga_60": 2.95, "xgf_60": 2.85, "cf_pct": 49.8},
    "CAR": {"name": "Carolina Hurricanes", "gf_60": 3.35, "ga_60": 2.65, "xgf_60": 3.40, "cf_pct": 54.2},
    "CHI": {"name": "Chicago Blackhawks", "gf_60": 2.55, "ga_60": 3.45, "xgf_60": 2.60, "cf_pct": 46.1},
    "COL": {"name": "Colorado Avalanche", "gf_60": 3.45, "ga_60": 2.85, "xgf_60": 3.50, "cf_pct": 53.5},
    "CBJ": {"name": "Columbus Blue Jackets", "gf_60": 2.70, "ga_60": 3.40, "xgf_60": 2.75, "cf_pct": 47.0},
    "DAL": {"name": "Dallas Stars", "gf_60": 3.25, "ga_60": 2.70, "xgf_60": 3.30, "cf_pct": 52.8},
    "DET": {"name": "Detroit Red Wings", "gf_60": 2.90, "ga_60": 3.10, "xgf_60": 2.95, "cf_pct": 49.2},
    "EDM": {"name": "Edmonton Oilers", "gf_60": 3.55, "ga_60": 3.00, "xgf_60": 3.45, "cf_pct": 51.5},
    "FLA": {"name": "Florida Panthers", "gf_60": 3.40, "ga_60": 2.75, "xgf_60": 3.35, "cf_pct": 53.0},
    "LAK": {"name": "Los Angeles Kings", "gf_60": 3.05, "ga_60": 2.80, "xgf_60": 3.10, "cf_pct": 51.2},
    "MIN": {"name": "Minnesota Wild", "gf_60": 3.00, "ga_60": 2.90, "xgf_60": 3.05, "cf_pct": 50.5},
    "MTL": {"name": "Montreal Canadiens", "gf_60": 2.85, "ga_60": 3.25, "xgf_60": 2.80, "cf_pct": 48.0},
    "NSH": {"name": "Nashville Predators", "gf_60": 2.75, "ga_60": 3.05, "xgf_60": 2.80, "cf_pct": 48.8},
    "NJD": {"name": "New Jersey Devils", "gf_60": 3.20, "ga_60": 2.95, "xgf_60": 3.25, "cf_pct": 51.8},
    "NYI": {"name": "New York Islanders", "gf_60": 2.80, "ga_60": 2.85, "xgf_60": 2.75, "cf_pct": 49.5},
    "NYR": {"name": "New York Rangers", "gf_60": 3.10, "ga_60": 2.70, "xgf_60": 3.15, "cf_pct": 51.0},
    "OTT": {"name": "Ottawa Senators", "gf_60": 2.95, "ga_60": 3.15, "xgf_60": 3.00, "cf_pct": 49.0},
    "PHI": {"name": "Philadelphia Flyers", "gf_60": 2.85, "ga_60": 3.20, "xgf_60": 2.90, "cf_pct": 48.2},
    "PIT": {"name": "Pittsburgh Penguins", "gf_60": 3.00, "ga_60": 3.10, "xgf_60": 3.05, "cf_pct": 50.0},
    "SJS": {"name": "San Jose Sharks", "gf_60": 2.50, "ga_60": 3.50, "xgf_60": 2.55, "cf_pct": 45.5},
    "SEA": {"name": "Seattle Kraken", "gf_60": 2.90, "ga_60": 3.00, "xgf_60": 2.95, "cf_pct": 49.5},
    "STL": {"name": "St. Louis Blues", "gf_60": 2.85, "ga_60": 3.15, "xgf_60": 2.90, "cf_pct": 48.5},
    "TBL": {"name": "Tampa Bay Lightning", "gf_60": 3.30, "ga_60": 2.90, "xgf_60": 3.25, "cf_pct": 52.0},
    "TOR": {"name": "Toronto Maple Leafs", "gf_60": 3.35, "ga_60": 2.85, "xgf_60": 3.30, "cf_pct": 52.5},
    "UTA": {"name": "Utah Hockey Club", "gf_60": 2.80, "ga_60": 3.10, "xgf_60": 2.85, "cf_pct": 48.5},
    "VAN": {"name": "Vancouver Canucks", "gf_60": 3.20, "ga_60": 2.80, "xgf_60": 3.15, "cf_pct": 51.5},
    "VGK": {"name": "Vegas Golden Knights", "gf_60": 3.25, "ga_60": 2.75, "xgf_60": 3.20, "cf_pct": 52.2},
    "WSH": {"name": "Washington Capitals", "gf_60": 3.15, "ga_60": 2.90, "xgf_60": 3.10, "cf_pct": 50.8},
    "WPG": {"name": "Winnipeg Jets", "gf_60": 3.40, "ga_60": 2.65, "xgf_60": 3.35, "cf_pct": 53.2},
}

# Goalie data (sample - would be fetched from API in production)
GOALIES = {
    "ANA": [{"name": "John Gibson", "sv_pct": 0.903, "gsax": -5.2}, {"name": "Lukas Dostal", "sv_pct": 0.910, "gsax": 2.1}],
    "BOS": [{"name": "Jeremy Swayman", "sv_pct": 0.918, "gsax": 8.5}, {"name": "Joonas Korpisalo", "sv_pct": 0.895, "gsax": -4.2}],
    "BUF": [{"name": "Ukko-Pekka Luukkonen", "sv_pct": 0.908, "gsax": 1.2}, {"name": "Devon Levi", "sv_pct": 0.902, "gsax": -1.5}],
    "CGY": [{"name": "Dustin Wolf", "sv_pct": 0.915, "gsax": 6.8}, {"name": "Dan Vladar", "sv_pct": 0.898, "gsax": -2.1}],
    "CAR": [{"name": "Pyotr Kochetkov", "sv_pct": 0.912, "gsax": 5.5}, {"name": "Frederik Andersen", "sv_pct": 0.920, "gsax": 9.2}],
    "CHI": [{"name": "Petr Mrazek", "sv_pct": 0.901, "gsax": -3.5}, {"name": "Arvid Soderblom", "sv_pct": 0.895, "gsax": -5.0}],
    "COL": [{"name": "Alexandar Georgiev", "sv_pct": 0.905, "gsax": -1.8}, {"name": "Justus Annunen", "sv_pct": 0.908, "gsax": 1.5}],
    "CBJ": [{"name": "Elvis Merzlikins", "sv_pct": 0.898, "gsax": -4.5}, {"name": "Daniil Tarasov", "sv_pct": 0.905, "gsax": 0.5}],
    "DAL": [{"name": "Jake Oettinger", "sv_pct": 0.915, "gsax": 10.2}, {"name": "Casey DeSmith", "sv_pct": 0.902, "gsax": -1.0}],
    "DET": [{"name": "Ville Husso", "sv_pct": 0.898, "gsax": -3.8}, {"name": "Alex Lyon", "sv_pct": 0.910, "gsax": 2.5}],
    "EDM": [{"name": "Stuart Skinner", "sv_pct": 0.905, "gsax": -2.5}, {"name": "Calvin Pickard", "sv_pct": 0.908, "gsax": 1.2}],
    "FLA": [{"name": "Sergei Bobrovsky", "sv_pct": 0.915, "gsax": 12.5}, {"name": "Spencer Knight", "sv_pct": 0.900, "gsax": -2.0}],
    "LAK": [{"name": "Cam Talbot", "sv_pct": 0.908, "gsax": 2.0}, {"name": "David Rittich", "sv_pct": 0.902, "gsax": -1.5}],
    "MIN": [{"name": "Filip Gustavsson", "sv_pct": 0.912, "gsax": 6.5}, {"name": "Marc-Andre Fleury", "sv_pct": 0.900, "gsax": -2.8}],
    "MTL": [{"name": "Sam Montembeault", "sv_pct": 0.905, "gsax": 1.5}, {"name": "Cayden Primeau", "sv_pct": 0.895, "gsax": -4.0}],
    "NSH": [{"name": "Juuse Saros", "sv_pct": 0.918, "gsax": 15.2}, {"name": "Scott Wedgewood", "sv_pct": 0.895, "gsax": -3.5}],
    "NJD": [{"name": "Jacob Markstrom", "sv_pct": 0.908, "gsax": 3.5}, {"name": "Jake Allen", "sv_pct": 0.902, "gsax": -1.2}],
    "NYI": [{"name": "Ilya Sorokin", "sv_pct": 0.915, "gsax": 11.0}, {"name": "Semyon Varlamov", "sv_pct": 0.905, "gsax": 1.8}],
    "NYR": [{"name": "Igor Shesterkin", "sv_pct": 0.925, "gsax": 22.5}, {"name": "Jonathan Quick", "sv_pct": 0.900, "gsax": -2.5}],
    "OTT": [{"name": "Linus Ullmark", "sv_pct": 0.915, "gsax": 9.8}, {"name": "Anton Forsberg", "sv_pct": 0.898, "gsax": -3.2}],
    "PHI": [{"name": "Samuel Ersson", "sv_pct": 0.905, "gsax": 1.0}, {"name": "Ivan Fedotov", "sv_pct": 0.900, "gsax": -1.5}],
    "PIT": [{"name": "Tristan Jarry", "sv_pct": 0.902, "gsax": -2.5}, {"name": "Alex Nedeljkovic", "sv_pct": 0.898, "gsax": -4.0}],
    "SJS": [{"name": "Mackenzie Blackwood", "sv_pct": 0.908, "gsax": 5.5}, {"name": "Vitek Vanecek", "sv_pct": 0.895, "gsax": -5.5}],
    "SEA": [{"name": "Joey Daccord", "sv_pct": 0.912, "gsax": 7.2}, {"name": "Philipp Grubauer", "sv_pct": 0.898, "gsax": -4.5}],
    "STL": [{"name": "Jordan Binnington", "sv_pct": 0.905, "gsax": 0.5}, {"name": "Joel Hofer", "sv_pct": 0.908, "gsax": 2.0}],
    "TBL": [{"name": "Andrei Vasilevskiy", "sv_pct": 0.915, "gsax": 13.5}, {"name": "Jonas Johansson", "sv_pct": 0.898, "gsax": -2.8}],
    "TOR": [{"name": "Joseph Woll", "sv_pct": 0.912, "gsax": 8.0}, {"name": "Anthony Stolarz", "sv_pct": 0.918, "gsax": 10.5}],
    "UTA": [{"name": "Karel Vejmelka", "sv_pct": 0.905, "gsax": 2.5}, {"name": "Connor Ingram", "sv_pct": 0.902, "gsax": 0.5}],
    "VAN": [{"name": "Thatcher Demko", "sv_pct": 0.918, "gsax": 12.0}, {"name": "Kevin Lankinen", "sv_pct": 0.910, "gsax": 5.5}],
    "VGK": [{"name": "Adin Hill", "sv_pct": 0.910, "gsax": 5.0}, {"name": "Ilya Samsonov", "sv_pct": 0.902, "gsax": -1.5}],
    "WSH": [{"name": "Charlie Lindgren", "sv_pct": 0.908, "gsax": 3.5}, {"name": "Logan Thompson", "sv_pct": 0.912, "gsax": 6.0}],
    "WPG": [{"name": "Connor Hellebuyck", "sv_pct": 0.925, "gsax": 28.5}, {"name": "Eric Comrie", "sv_pct": 0.895, "gsax": -3.5}],
}


def run_prediction(home_team, away_team, home_goalie, away_goalie, home_ml, away_ml, ou_line):
    """Run full prediction pipeline."""

    # Get team stats
    home_stats = TEAMS[home_team]
    away_stats = TEAMS[away_team]

    # Initialize agents
    poisson = PoissonEngine()
    mc = MonteCarloSimulator(MonteCarloConfig(n_simulations=10000))
    edge = EdgeCalculator()

    # Run Poisson model
    poisson_result = poisson.predict(
        home_stats={'gf_60': home_stats['gf_60'], 'ga_60': home_stats['ga_60']},
        away_stats={'gf_60': away_stats['gf_60'], 'ga_60': away_stats['ga_60']},
        home_goalie={'gsax': home_goalie['gsax'], 'sv_pct': home_goalie['sv_pct']},
        away_goalie={'gsax': away_goalie['gsax'], 'sv_pct': away_goalie['sv_pct']},
        over_under_line=ou_line,
    )

    # Run Monte Carlo simulation
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

    # Edge analysis
    edge_result = edge.full_game_analysis(
        home_win_prob=mc_result['final']['home_win_prob'],
        over_prob=ou_result.get(ou_line, {}).get('over_prob', 0.5),
        home_ml=home_ml,
        away_ml=away_ml,
        total_line=ou_line,
    )

    return {
        'poisson': poisson_result,
        'monte_carlo': mc_result,
        'over_under': ou_result,
        'edge': edge_result,
    }


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.title("🏒 NHL Game Prediction Model")
st.markdown("**Poisson Distribution + Monte Carlo Simulation + ML Ensemble**")
st.markdown("---")

# Sidebar - Game Setup
st.sidebar.header("Game Setup")

col1, col2 = st.sidebar.columns(2)

with col1:
    away_team = st.selectbox(
        "Away Team",
        options=list(TEAMS.keys()),
        format_func=lambda x: f"{x} - {TEAMS[x]['name']}",
        index=list(TEAMS.keys()).index("MTL"),
    )

with col2:
    home_team = st.selectbox(
        "Home Team",
        options=list(TEAMS.keys()),
        format_func=lambda x: f"{x} - {TEAMS[x]['name']}",
        index=list(TEAMS.keys()).index("TOR"),
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Goalie Selection")

# Goalie dropdowns
home_goalie_options = GOALIES.get(home_team, [{"name": "Starter", "sv_pct": 0.905, "gsax": 0}])
away_goalie_options = GOALIES.get(away_team, [{"name": "Starter", "sv_pct": 0.905, "gsax": 0}])

home_goalie_name = st.sidebar.selectbox(
    f"{home_team} Goalie",
    options=[g['name'] for g in home_goalie_options],
)
home_goalie = next(g for g in home_goalie_options if g['name'] == home_goalie_name)

away_goalie_name = st.sidebar.selectbox(
    f"{away_team} Goalie",
    options=[g['name'] for g in away_goalie_options],
)
away_goalie = next(g for g in away_goalie_options if g['name'] == away_goalie_name)

st.sidebar.markdown("---")
st.sidebar.subheader("Betting Lines")

home_ml = st.sidebar.number_input("Home Moneyline", value=-150, step=5)
away_ml = st.sidebar.number_input("Away Moneyline", value=130, step=5)
ou_line = st.sidebar.number_input("Over/Under Line", value=6.5, step=0.5)

# Run prediction button
if st.sidebar.button("🎯 Run Prediction", type="primary", use_container_width=True):

    with st.spinner("Running 10,000 Monte Carlo simulations..."):
        results = run_prediction(
            home_team, away_team,
            home_goalie, away_goalie,
            home_ml, away_ml, ou_line
        )

    # Store in session state
    st.session_state['results'] = results
    st.session_state['home_team'] = home_team
    st.session_state['away_team'] = away_team

# Display results if available
if 'results' in st.session_state:
    results = st.session_state['results']
    home_team = st.session_state['home_team']
    away_team = st.session_state['away_team']

    # Header
    st.markdown(f"### {TEAMS[away_team]['name']} @ {TEAMS[home_team]['name']}")
    st.markdown(f"*{date.today().strftime('%B %d, %Y')}*")

    # Main prediction display
    st.markdown("---")

    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        away_prob = results['monte_carlo']['final']['away_win_prob']
        st.markdown(f"<div class='big-number'>{away_prob*100:.1f}%</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>{TEAMS[away_team]['name']}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div style='text-align: center; font-size: 24px; padding-top: 20px;'>VS</div>", unsafe_allow_html=True)

    with col3:
        home_prob = results['monte_carlo']['final']['home_win_prob']
        st.markdown(f"<div class='big-number'>{home_prob*100:.1f}%</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-label'>{TEAMS[home_team]['name']}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Detailed results in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Prediction Details", "🎰 Betting Edge", "📈 Score Distribution", "⚙️ Model Details"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Expected Goals")
            st.metric(f"{home_team} (Home)", f"{results['poisson']['lambda_home']:.2f}")
            st.metric(f"{away_team} (Away)", f"{results['poisson']['lambda_away']:.2f}")
            st.metric("Expected Total", f"{results['poisson']['expected_total']:.1f}")

        with col2:
            st.markdown("#### Most Likely Outcomes")
            st.metric("Predicted Score", results['monte_carlo']['scores']['most_likely'])
            st.metric("OT Probability", f"{results['monte_carlo']['meta']['overtime_games']*100:.1f}%")

            # 60-min results
            st.markdown("**60-Minute Result:**")
            reg = results['monte_carlo']['sixty_min']
            st.write(f"- {home_team} Win: {reg['home_win_prob']*100:.1f}%")
            st.write(f"- {away_team} Win: {reg['away_win_prob']*100:.1f}%")
            st.write(f"- Tie (→OT): {reg['tie_prob']*100:.1f}%")

    with tab2:
        st.markdown("#### Moneyline Analysis")

        edge_ml = results['edge']['moneyline']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{home_team} ({home_ml:+d})**")
            edge_val = edge_ml['home']['edge'] * 100
            edge_class = "edge-positive" if edge_val > 0 else "edge-negative"
            st.markdown(f"Edge: <span class='{edge_class}'>{edge_val:+.1f}%</span>", unsafe_allow_html=True)
            st.write(f"Kelly: {edge_ml['home']['kelly']*100:.1f}% of bankroll")
            st.write(f"EV/$100: ${edge_ml['home']['ev_per_100']:.2f}")

            rec = edge_ml['home']['recommendation']
            if "STRONG" in rec:
                st.markdown(f"<span class='recommendation-strong'>{rec}</span>", unsafe_allow_html=True)
            elif "LEAN" in rec:
                st.markdown(f"<span class='recommendation-lean'>{rec}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='recommendation-pass'>{rec}</span>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"**{away_team} ({away_ml:+d})**")
            edge_val = edge_ml['away']['edge'] * 100
            edge_class = "edge-positive" if edge_val > 0 else "edge-negative"
            st.markdown(f"Edge: <span class='{edge_class}'>{edge_val:+.1f}%</span>", unsafe_allow_html=True)
            st.write(f"Kelly: {edge_ml['away']['kelly']*100:.1f}% of bankroll")
            st.write(f"EV/$100: ${edge_ml['away']['ev_per_100']:.2f}")

            rec = edge_ml['away']['recommendation']
            if "STRONG" in rec:
                st.markdown(f"<span class='recommendation-strong'>{rec}</span>", unsafe_allow_html=True)
            elif "LEAN" in rec:
                st.markdown(f"<span class='recommendation-lean'>{rec}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='recommendation-pass'>{rec}</span>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Over/Under Analysis")

        edge_ou = results['edge']['over_under']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Over {ou_line}**")
            over_prob = results['over_under'].get(ou_line, {}).get('over_prob', 0.5) * 100
            st.write(f"Model Probability: {over_prob:.1f}%")
            edge_val = edge_ou['over']['edge'] * 100
            edge_class = "edge-positive" if edge_val > 0 else "edge-negative"
            st.markdown(f"Edge: <span class='{edge_class}'>{edge_val:+.1f}%</span>", unsafe_allow_html=True)

            rec = edge_ou['over']['recommendation']
            if "STRONG" in rec:
                st.markdown(f"<span class='recommendation-strong'>{rec}</span>", unsafe_allow_html=True)
            elif "LEAN" in rec:
                st.markdown(f"<span class='recommendation-lean'>{rec}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='recommendation-pass'>{rec}</span>", unsafe_allow_html=True)

        with col2:
            st.markdown(f"**Under {ou_line}**")
            under_prob = results['over_under'].get(ou_line, {}).get('under_prob', 0.5) * 100
            st.write(f"Model Probability: {under_prob:.1f}%")
            edge_val = edge_ou['under']['edge'] * 100
            edge_class = "edge-positive" if edge_val > 0 else "edge-negative"
            st.markdown(f"Edge: <span class='{edge_class}'>{edge_val:+.1f}%</span>", unsafe_allow_html=True)

            rec = edge_ou['under']['recommendation']
            if "STRONG" in rec:
                st.markdown(f"<span class='recommendation-strong'>{rec}</span>", unsafe_allow_html=True)
            elif "LEAN" in rec:
                st.markdown(f"<span class='recommendation-lean'>{rec}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span class='recommendation-pass'>{rec}</span>", unsafe_allow_html=True)

        # Top plays summary
        if results['edge'].get('top_plays'):
            st.markdown("---")
            st.markdown("#### 🎯 Top Plays")
            for i, play in enumerate(results['edge']['top_plays'], 1):
                edge_pct = play['edge'] * 100
                st.write(f"{i}. **{play['type'].upper()} {play['side'].upper()}** - Edge: {edge_pct:+.1f}% ({play['confidence']})")

    with tab3:
        st.markdown("#### Score Probability Distribution")

        # Create score distribution chart
        scores = results['monte_carlo']['scores']['distribution']
        score_df = pd.DataFrame([
            {'Score': k, 'Probability': v * 100}
            for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:15]
        ])

        st.bar_chart(score_df.set_index('Score'))

        st.markdown("#### Top 10 Most Likely Scores")
        for i, (score, prob) in enumerate(sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10], 1):
            st.write(f"{i}. **{score}** - {prob*100:.1f}%")

    with tab4:
        st.markdown("#### Model Configuration")
        st.write(f"- **Simulations:** 10,000")
        st.write(f"- **Home Ice Advantage:** 6%")
        st.write(f"- **OT Rate (historical):** ~23%")

        st.markdown("#### Team Stats Used")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**{home_team}**")
            st.write(f"- GF/60: {TEAMS[home_team]['gf_60']}")
            st.write(f"- GA/60: {TEAMS[home_team]['ga_60']}")
            st.write(f"- xGF/60: {TEAMS[home_team]['xgf_60']}")
            st.write(f"- CF%: {TEAMS[home_team]['cf_pct']}")
            st.write(f"- Goalie: {home_goalie_name}")
            st.write(f"- Goalie SV%: {home_goalie['sv_pct']}")
            st.write(f"- Goalie GSAX: {home_goalie['gsax']}")

        with col2:
            st.markdown(f"**{away_team}**")
            st.write(f"- GF/60: {TEAMS[away_team]['gf_60']}")
            st.write(f"- GA/60: {TEAMS[away_team]['ga_60']}")
            st.write(f"- xGF/60: {TEAMS[away_team]['xgf_60']}")
            st.write(f"- CF%: {TEAMS[away_team]['cf_pct']}")
            st.write(f"- Goalie: {away_goalie_name}")
            st.write(f"- Goalie SV%: {away_goalie['sv_pct']}")
            st.write(f"- Goalie GSAX: {away_goalie['gsax']}")

else:
    # Default state - show instructions
    st.info("👈 Select teams, goalies, and betting lines in the sidebar, then click **Run Prediction**")

    st.markdown("### How It Works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 1️⃣ Poisson Model")
        st.write("Calculates expected goals (λ) for each team based on offensive/defensive strength and goalie adjustments.")

    with col2:
        st.markdown("#### 2️⃣ Monte Carlo")
        st.write("Simulates 10,000 games using Poisson-distributed scores, including overtime/shootout resolution.")

    with col3:
        st.markdown("#### 3️⃣ Edge Analysis")
        st.write("Compares model probabilities to betting lines to identify value. Calculates Kelly criterion for optimal sizing.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 12px;'>"
    "NHL Prediction Model v1.0 | Poisson + Monte Carlo + ML Ensemble | "
    "Data: Natural Stat Trick, MoneyPuck"
    "</div>",
    unsafe_allow_html=True
)
