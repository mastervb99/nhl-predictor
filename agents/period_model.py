"""
PeriodModel Agent
Role: Calculate 1st period expected goals and O/U probabilities
"""
from typing import Dict, Tuple
from scipy.stats import poisson
import numpy as np


class PeriodModel:
    """
    First period goal prediction model.

    NHL games have 3 periods. Historical data shows goals are distributed:
    - 1st period: ~30-33% of goals
    - 2nd period: ~33-35% of goals
    - 3rd period: ~32-35% of goals

    We use a scaling factor to convert full-game lambdas to 1st period lambdas.
    """

    FIRST_PERIOD_FACTOR = 0.32  # ~32% of goals in 1st period

    def __init__(self, period_factor: float = None):
        self.period_factor = period_factor or self.FIRST_PERIOD_FACTOR

    def calculate_period_lambdas(
        self,
        full_game_lambda_home: float,
        full_game_lambda_away: float
    ) -> Tuple[float, float]:
        """
        Convert full-game expected goals to 1st period expected goals.

        Args:
            full_game_lambda_home: Full game expected goals for home team
            full_game_lambda_away: Full game expected goals for away team

        Returns:
            Tuple of (1st_period_lambda_home, 1st_period_lambda_away)
        """
        p1_lambda_home = full_game_lambda_home * self.period_factor
        p1_lambda_away = full_game_lambda_away * self.period_factor

        return p1_lambda_home, p1_lambda_away

    def calculate_period_probabilities(
        self,
        p1_lambda_home: float,
        p1_lambda_away: float,
        max_goals: int = 5
    ) -> Dict:
        """
        Calculate 1st period outcome probabilities.

        Returns:
            Dict with home win, away win, tie probabilities
        """
        home_win_prob = 0.0
        away_win_prob = 0.0
        tie_prob = 0.0

        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                prob = (
                    poisson.pmf(home_goals, p1_lambda_home) *
                    poisson.pmf(away_goals, p1_lambda_away)
                )

                if home_goals > away_goals:
                    home_win_prob += prob
                elif away_goals > home_goals:
                    away_win_prob += prob
                else:
                    tie_prob += prob

        # Normalize
        total = home_win_prob + away_win_prob + tie_prob
        if total > 0:
            home_win_prob /= total
            away_win_prob /= total
            tie_prob /= total

        return {
            'home_win': round(home_win_prob, 4),
            'away_win': round(away_win_prob, 4),
            'tie': round(tie_prob, 4),
        }

    def calculate_period_over_under(
        self,
        p1_lambda_home: float,
        p1_lambda_away: float,
        line: float = 1.5
    ) -> Dict:
        """
        Calculate 1st period over/under probabilities.

        Common 1st period lines: 0.5, 1.5, 2.5
        """
        total_lambda = p1_lambda_home + p1_lambda_away

        # Calculate probabilities
        under_prob = poisson.cdf(int(line - 0.5), total_lambda)
        over_prob = 1 - poisson.cdf(int(line), total_lambda)

        return {
            'line': line,
            'over_prob': round(over_prob, 4),
            'under_prob': round(under_prob, 4),
            'expected_total': round(total_lambda, 2),
        }

    def predict_first_period(
        self,
        full_game_lambda_home: float,
        full_game_lambda_away: float
    ) -> Dict:
        """
        Full 1st period prediction.

        Args:
            full_game_lambda_home: Full game expected goals for home
            full_game_lambda_away: Full game expected goals for away

        Returns:
            Dict with all 1st period predictions
        """
        # Calculate period lambdas
        p1_home, p1_away = self.calculate_period_lambdas(
            full_game_lambda_home,
            full_game_lambda_away
        )

        # Calculate probabilities
        win_probs = self.calculate_period_probabilities(p1_home, p1_away)

        # Calculate O/U for common lines
        ou_results = {}
        for line in [0.5, 1.5, 2.5]:
            ou_results[line] = self.calculate_period_over_under(p1_home, p1_away, line)

        # Most likely score
        max_prob = 0
        most_likely = "0-0"
        for home_goals in range(4):
            for away_goals in range(4):
                prob = (
                    poisson.pmf(home_goals, p1_home) *
                    poisson.pmf(away_goals, p1_away)
                )
                if prob > max_prob:
                    max_prob = prob
                    most_likely = f"{home_goals}-{away_goals}"

        return {
            'lambda_home': round(p1_home, 2),
            'lambda_away': round(p1_away, 2),
            'expected_total': round(p1_home + p1_away, 2),
            'home_win_prob': win_probs['home_win'],
            'away_win_prob': win_probs['away_win'],
            'tie_prob': win_probs['tie'],
            'most_likely_score': most_likely,
            'over_under': ou_results,
        }
