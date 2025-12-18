#!/usr/bin/env python3
"""
NHL Prediction Model - Demo Script

Demonstrates the full prediction pipeline:
1. Poisson goal modeling
2. Monte Carlo simulation
3. ML adjustments
4. Betting edge analysis
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.poisson_engine import PoissonEngine
from agents.monte_carlo import MonteCarloSimulator, MonteCarloConfig
from agents.feature_engineer import FeatureEngineer
from agents.edge_calculator import EdgeCalculator
from agents.data_ingestor import DataIngestor


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def demo_prediction(home_team: str, away_team: str):
    """Run full prediction demo for a game."""

    print_section(f"NHL GAME PREDICTION: {away_team} @ {home_team}")

    # Initialize agents
    data = DataIngestor()
    poisson = PoissonEngine()
    mc = MonteCarloSimulator(MonteCarloConfig(n_simulations=10000))
    edge = EdgeCalculator()

    # Fetch data
    print("\n[1] Fetching team data...")
    game_data = data.get_full_game_data(home_team, away_team)

    home_stats = game_data['home_team']['stats'].get('2024-25', {})
    away_stats = game_data['away_team']['stats'].get('2024-25', {})
    home_goalie = game_data['home_team']['goalie']
    away_goalie = game_data['away_team']['goalie']
    odds = game_data['odds']

    print(f"    {home_team}: GF/60={home_stats.get('gf_60', 'N/A')}, GA/60={home_stats.get('ga_60', 'N/A')}")
    print(f"    {away_team}: GF/60={away_stats.get('gf_60', 'N/A')}, GA/60={away_stats.get('ga_60', 'N/A')}")

    if home_goalie:
        print(f"    {home_team} Goalie: {home_goalie.get('name', 'TBD')} (SV%={home_goalie.get('sv_pct', 'N/A')}, GSAX={home_goalie.get('gsax', 'N/A')})")
    if away_goalie:
        print(f"    {away_team} Goalie: {away_goalie.get('name', 'TBD')} (SV%={away_goalie.get('sv_pct', 'N/A')}, GSAX={away_goalie.get('gsax', 'N/A')})")

    # Poisson Model
    print("\n[2] Running Poisson Model...")
    poisson_result = poisson.predict(
        home_stats={'gf_60': home_stats.get('gf_60', 3.0), 'ga_60': home_stats.get('ga_60', 3.0)},
        away_stats={'gf_60': away_stats.get('gf_60', 3.0), 'ga_60': away_stats.get('ga_60', 3.0)},
        home_goalie={'gsax': home_goalie.get('gsax', 0), 'sv_pct': home_goalie.get('sv_pct', 0.905)} if home_goalie else None,
        away_goalie={'gsax': away_goalie.get('gsax', 0), 'sv_pct': away_goalie.get('sv_pct', 0.905)} if away_goalie else None,
        over_under_line=odds.get('over_under', 6.0),
    )

    print(f"    λ_home (expected goals): {poisson_result['lambda_home']}")
    print(f"    λ_away (expected goals): {poisson_result['lambda_away']}")
    print(f"    Expected Total: {poisson_result['expected_total']}")

    # Monte Carlo Simulation
    print("\n[3] Running Monte Carlo Simulation (10,000 games)...")
    mc_result = mc.simulate_games(
        poisson_result['lambda_home'],
        poisson_result['lambda_away']
    )

    print(f"\n    REGULATION (60-min) Results:")
    print(f"      {home_team} Win: {mc_result['sixty_min']['home_win_prob']*100:.1f}%")
    print(f"      {away_team} Win: {mc_result['sixty_min']['away_win_prob']*100:.1f}%")
    print(f"      Tie (goes to OT): {mc_result['sixty_min']['tie_prob']*100:.1f}%")

    print(f"\n    FINAL Results (including OT/SO):")
    print(f"      {home_team} Win: {mc_result['final']['home_win_prob']*100:.1f}%")
    print(f"      {away_team} Win: {mc_result['final']['away_win_prob']*100:.1f}%")

    print(f"\n    Score Prediction:")
    print(f"      Most Likely Score: {mc_result['scores']['most_likely']} ({mc_result['scores']['most_likely_prob']*100:.1f}%)")
    print(f"      Expected Total: {mc_result['totals']['expected_total']:.1f} goals")

    # Over/Under Analysis
    print("\n[4] Over/Under Analysis...")
    ou_lines = [5.5, 6.0, 6.5, 7.0]
    ou_result = mc.calculate_over_under(
        poisson_result['lambda_home'],
        poisson_result['lambda_away'],
        ou_lines
    )

    for line, probs in ou_result.items():
        print(f"      O/U {line}: Over {probs['over_prob']*100:.1f}% | Under {probs['under_prob']*100:.1f}%")

    # Edge Analysis
    if odds.get('home_ml') and odds.get('away_ml'):
        print("\n[5] Betting Edge Analysis...")
        print(f"    Market Lines: {home_team} {odds['home_ml']:+d} / {away_team} {odds['away_ml']:+d}")
        print(f"    O/U Line: {odds.get('over_under', 6.0)}")

        edge_result = edge.full_game_analysis(
            home_win_prob=mc_result['final']['home_win_prob'],
            over_prob=ou_result.get(odds.get('over_under', 6.0), {}).get('over_prob', 0.5),
            home_ml=odds['home_ml'],
            away_ml=odds['away_ml'],
            total_line=odds.get('over_under', 6.0),
        )

        print(f"\n    MONEYLINE ANALYSIS:")
        print(f"      {home_team} Edge: {edge_result['moneyline']['home']['edge']*100:+.1f}%")
        print(f"      {away_team} Edge: {edge_result['moneyline']['away']['edge']*100:+.1f}%")
        print(f"      Recommendation: {edge_result['moneyline']['home']['recommendation']}")

        print(f"\n    OVER/UNDER ANALYSIS:")
        print(f"      Over Edge: {edge_result['over_under']['over']['edge']*100:+.1f}%")
        print(f"      Under Edge: {edge_result['over_under']['under']['edge']*100:+.1f}%")
        print(f"      Recommendation: {edge_result['over_under']['over']['recommendation']}")

        if edge_result['top_plays']:
            print(f"\n    TOP PLAYS:")
            for i, play in enumerate(edge_result['top_plays'], 1):
                print(f"      {i}. {play['type'].upper()} {play['side'].upper()} (Edge: {play['edge']*100:+.1f}%, Confidence: {play['confidence']})")

    print_section("PREDICTION COMPLETE")


def main():
    """Run demo predictions."""
    print("\n" + "="*60)
    print("  NHL PREDICTION MODEL - PROTOTYPE DEMO")
    print("  Poisson + Monte Carlo + ML Ensemble")
    print("="*60)

    # Demo games
    games = [
        ("TOR", "MTL"),  # Leafs vs Canadiens
        ("COL", "VGK"),  # Avalanche vs Golden Knights
    ]

    for home, away in games:
        demo_prediction(home, away)

    print("\n\n[API Server]")
    print("To start the API server, run:")
    print("  uvicorn api.main:app --reload")
    print("\nEndpoints:")
    print("  GET  /api/games          - Today's games")
    print("  GET  /api/teams/{code}   - Team stats")
    print("  GET  /api/goalies/{code} - Team goalies")
    print("  GET  /api/odds           - Betting odds")
    print("  POST /api/predict        - Full prediction")
    print("  GET  /api/predict/quick  - Quick prediction")


if __name__ == "__main__":
    main()
