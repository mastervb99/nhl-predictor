"""
PropsModel Agent
Role: Calculate prop bet probabilities (GIFT, SOG)
"""
from typing import Dict, Optional
from scipy.stats import poisson
import numpy as np


class PropsModel:
    """
    Prop bet probability calculations.

    GIFT (Goal In First Ten minutes):
    - League average: ~58% of games have a goal in first 10 mins
    - Model using Poisson with ~1/6 of 1st period lambda

    1+ SOG (Shot on Goal) in First 2 Minutes:
    - Most teams average 2-3 shots in first 2 mins
    - Model using team shot pace data
    """

    # Historical averages
    LEAGUE_GIFT_RATE = 0.58          # ~58% of games have goal in first 10 mins
    LEAGUE_SOG_2MIN_RATE = 0.85      # ~85% of games have SOG in first 2 mins
    FIRST_10_MIN_FACTOR = 0.167     # 10 mins / 60 mins

    def __init__(self):
        # Team-specific GIFT rates (if available, otherwise use league average)
        self._team_gift_rates = {}
        self._team_sog_rates = {}

    def set_team_gift_rate(self, team: str, rate: float):
        """Set team-specific GIFT rate."""
        self._team_gift_rates[team] = rate

    def set_team_sog_rate(self, team: str, rate: float):
        """Set team-specific SOG rate."""
        self._team_sog_rates[team] = rate

    def calculate_gift_probability(
        self,
        home_gf_60: float,
        away_gf_60: float,
        home_ga_60: float,
        away_ga_60: float,
        home_team: str = None,
        away_team: str = None
    ) -> Dict:
        """
        Calculate GIFT (Goal In First Ten) probability.

        Uses Poisson distribution with adjusted lambda for first 10 minutes.
        P(at least 1 goal) = 1 - P(0 goals home) * P(0 goals away)
        """
        # Calculate expected goals in first 10 minutes
        # Combine offensive and defensive factors
        home_scoring_rate = (home_gf_60 + away_ga_60) / 2  # Home team scoring vs away defense
        away_scoring_rate = (away_gf_60 + home_ga_60) / 2  # Away team scoring vs home defense

        # Scale to 10 minutes
        lambda_home_10 = home_scoring_rate * self.FIRST_10_MIN_FACTOR
        lambda_away_10 = away_scoring_rate * self.FIRST_10_MIN_FACTOR

        # P(no goals in first 10 mins)
        p_no_home_goals = poisson.pmf(0, lambda_home_10)
        p_no_away_goals = poisson.pmf(0, lambda_away_10)
        p_no_goals = p_no_home_goals * p_no_away_goals

        # P(at least 1 goal) = GIFT probability
        gift_prob = 1 - p_no_goals

        # Blend with historical rates if available
        if home_team and home_team in self._team_gift_rates:
            team_rate = (self._team_gift_rates.get(home_team, self.LEAGUE_GIFT_RATE) +
                        self._team_gift_rates.get(away_team, self.LEAGUE_GIFT_RATE)) / 2
            gift_prob = 0.7 * gift_prob + 0.3 * team_rate  # Weight model more

        return {
            'gift_prob': round(gift_prob, 4),
            'no_gift_prob': round(1 - gift_prob, 4),
            'lambda_home_10': round(lambda_home_10, 3),
            'lambda_away_10': round(lambda_away_10, 3),
            'league_avg': self.LEAGUE_GIFT_RATE,
            'vs_league': round(gift_prob - self.LEAGUE_GIFT_RATE, 4),
        }

    def calculate_sog_probability(
        self,
        home_shots_60: float = 30.0,
        away_shots_60: float = 30.0,
        home_team: str = None,
        away_team: str = None
    ) -> Dict:
        """
        Calculate 1+ SOG in first 2 minutes probability.

        Uses Poisson distribution for shot attempts.
        League average is ~30 shots per 60 mins = 1 shot per 2 mins per team.
        """
        # Calculate expected shots in first 2 minutes
        # 2 mins / 60 mins = 1/30 of game
        shots_factor = 2 / 60

        lambda_home_2 = home_shots_60 * shots_factor
        lambda_away_2 = away_shots_60 * shots_factor

        # P(at least 1 SOG from either team)
        p_no_home_sog = poisson.pmf(0, lambda_home_2)
        p_no_away_sog = poisson.pmf(0, lambda_away_2)

        # P(at least 1 SOG from either team)
        p_at_least_one = 1 - (p_no_home_sog * p_no_away_sog)

        # Individual team probabilities
        p_home_sog = 1 - p_no_home_sog
        p_away_sog = 1 - p_no_away_sog

        return {
            'either_team_sog_prob': round(p_at_least_one, 4),
            'home_sog_prob': round(p_home_sog, 4),
            'away_sog_prob': round(p_away_sog, 4),
            'lambda_home_2': round(lambda_home_2, 3),
            'lambda_away_2': round(lambda_away_2, 3),
            'league_avg': self.LEAGUE_SOG_2MIN_RATE,
        }

    def calculate_all_props(
        self,
        home_stats: Dict,
        away_stats: Dict
    ) -> Dict:
        """
        Calculate all prop probabilities.

        Args:
            home_stats: Dict with gf_60, ga_60, shots_60
            away_stats: Dict with gf_60, ga_60, shots_60

        Returns:
            Dict with all prop predictions
        """
        gift = self.calculate_gift_probability(
            home_gf_60=home_stats.get('gf_60', 3.0),
            away_gf_60=away_stats.get('gf_60', 3.0),
            home_ga_60=home_stats.get('ga_60', 3.0),
            away_ga_60=away_stats.get('ga_60', 3.0),
            home_team=home_stats.get('team'),
            away_team=away_stats.get('team'),
        )

        sog = self.calculate_sog_probability(
            home_shots_60=home_stats.get('shots_60', 30.0),
            away_shots_60=away_stats.get('shots_60', 30.0),
            home_team=home_stats.get('team'),
            away_team=away_stats.get('team'),
        )

        return {
            'gift': gift,
            'sog_2min': sog,
        }


# Convenience function
def quick_props(home_gf: float, away_gf: float, home_ga: float, away_ga: float) -> Dict:
    """Quick prop calculation with minimal input."""
    model = PropsModel()
    return model.calculate_gift_probability(home_gf, away_gf, home_ga, away_ga)
