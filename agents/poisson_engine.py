"""
PoissonEngine Agent
Role: Calculate expected goals using Poisson regression model
Inputs: Team stats, goalie stats, home/away context
Outputs: Lambda values (expected goals) for each team, win probabilities
"""
import numpy as np
from scipy.stats import poisson
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class PoissonConfig:
    """Configuration for Poisson model"""
    home_ice_advantage: float = 1.06  # 6% boost for home team
    league_avg_goals: float = 3.05    # 2024-25 league average
    max_goals: int = 10               # Max goals to consider in probability calc
    goalie_weight: float = 0.15       # Weight for goalie adjustment


class PoissonEngine:
    """
    Poisson-based goal prediction model.

    The model calculates expected goals (lambda) for each team using:
    λ_home = attack_home × defense_away × home_ice × goalie_adj × league_avg
    λ_away = attack_away × defense_home × goalie_adj × league_avg

    Where:
    - attack = team's offensive strength (goals_for / league_avg)
    - defense = opponent's defensive weakness (goals_against / league_avg)
    - home_ice = home ice advantage factor (~1.06)
    - goalie_adj = adjustment based on goalie's GSAX
    """

    def __init__(self, config: PoissonConfig = None):
        self.config = config or PoissonConfig()
        self._league_stats = None

    def set_league_stats(self, league_avg_gf: float, league_avg_ga: float):
        """Set league average stats for normalization."""
        self._league_stats = {
            'avg_gf': league_avg_gf,
            'avg_ga': league_avg_ga
        }

    def calculate_attack_strength(self, team_gf_60: float) -> float:
        """Calculate team's offensive strength relative to league average."""
        league_avg = self._league_stats['avg_gf'] if self._league_stats else self.config.league_avg_goals
        return team_gf_60 / league_avg

    def calculate_defense_strength(self, team_ga_60: float) -> float:
        """Calculate team's defensive strength (lower is better)."""
        league_avg = self._league_stats['avg_ga'] if self._league_stats else self.config.league_avg_goals
        return team_ga_60 / league_avg

    def calculate_goalie_adjustment(
        self,
        goalie_gsax: float,
        goalie_sv_pct: float,
        league_avg_sv_pct: float = 0.905
    ) -> float:
        """
        Calculate goalie adjustment factor.

        Positive GSAX = better than expected = reduces opponent's lambda
        Higher SV% = better = reduces opponent's lambda
        """
        # GSAX component: normalize to per-game impact
        gsax_adj = 1 - (goalie_gsax / 100) * self.config.goalie_weight

        # SV% component
        sv_diff = goalie_sv_pct - league_avg_sv_pct
        sv_adj = 1 - (sv_diff * 10)  # 1% better SV% = ~10% fewer goals

        # Combine adjustments
        return (gsax_adj + sv_adj) / 2

    def calculate_lambdas(
        self,
        home_gf_60: float,
        home_ga_60: float,
        away_gf_60: float,
        away_ga_60: float,
        home_goalie_gsax: float = 0,
        home_goalie_sv_pct: float = 0.905,
        away_goalie_gsax: float = 0,
        away_goalie_sv_pct: float = 0.905
    ) -> Tuple[float, float]:
        """
        Calculate expected goals (lambda) for each team.

        Returns:
            Tuple of (lambda_home, lambda_away)
        """
        # Attack and defense strengths
        home_attack = self.calculate_attack_strength(home_gf_60)
        home_defense = self.calculate_defense_strength(home_ga_60)
        away_attack = self.calculate_attack_strength(away_gf_60)
        away_defense = self.calculate_defense_strength(away_ga_60)

        # Goalie adjustments (affects opponent's scoring)
        home_goalie_adj = self.calculate_goalie_adjustment(
            home_goalie_gsax, home_goalie_sv_pct
        )
        away_goalie_adj = self.calculate_goalie_adjustment(
            away_goalie_gsax, away_goalie_sv_pct
        )

        # Calculate lambdas
        lambda_home = (
            home_attack *
            away_defense *
            self.config.home_ice_advantage *
            away_goalie_adj *
            self.config.league_avg_goals
        )

        lambda_away = (
            away_attack *
            home_defense *
            home_goalie_adj *
            self.config.league_avg_goals
        )

        return lambda_home, lambda_away

    def calculate_win_probabilities(
        self,
        lambda_home: float,
        lambda_away: float
    ) -> Dict[str, float]:
        """
        Calculate win/loss/tie probabilities from Poisson lambdas.

        Uses probability mass function to calculate exact score probabilities,
        then aggregates to win/loss/tie.
        """
        max_goals = self.config.max_goals
        home_win_prob = 0.0
        away_win_prob = 0.0
        tie_prob = 0.0

        # Calculate probability for each score combination
        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                prob = (
                    poisson.pmf(home_goals, lambda_home) *
                    poisson.pmf(away_goals, lambda_away)
                )

                if home_goals > away_goals:
                    home_win_prob += prob
                elif away_goals > home_goals:
                    away_win_prob += prob
                else:
                    tie_prob += prob

        # Normalize to ensure probabilities sum to 1
        total = home_win_prob + away_win_prob + tie_prob
        if total > 0:
            home_win_prob /= total
            away_win_prob /= total
            tie_prob /= total

        return {
            'home_win': home_win_prob,
            'away_win': away_win_prob,
            'tie': tie_prob
        }

    def calculate_score_probabilities(
        self,
        lambda_home: float,
        lambda_away: float
    ) -> Dict[str, float]:
        """
        Calculate probability distribution for exact scores.

        Returns dict mapping "home-away" score strings to probabilities.
        """
        max_goals = self.config.max_goals
        score_probs = {}

        for home_goals in range(max_goals + 1):
            for away_goals in range(max_goals + 1):
                prob = (
                    poisson.pmf(home_goals, lambda_home) *
                    poisson.pmf(away_goals, lambda_away)
                )
                score_key = f"{home_goals}-{away_goals}"
                score_probs[score_key] = prob

        return score_probs

    def calculate_over_under_prob(
        self,
        lambda_home: float,
        lambda_away: float,
        total_line: float
    ) -> Dict[str, float]:
        """
        Calculate over/under probabilities for a given total line.
        """
        lambda_total = lambda_home + lambda_away

        # Use Poisson CDF for total goals
        under_prob = poisson.cdf(int(total_line - 0.5), lambda_total)
        over_prob = 1 - poisson.cdf(int(total_line), lambda_total)

        # Push probability (exactly hitting the line)
        push_prob = 0
        if total_line == int(total_line):
            push_prob = poisson.pmf(int(total_line), lambda_total)

        return {
            'over': over_prob,
            'under': under_prob,
            'push': push_prob,
            'expected_total': lambda_total
        }

    def predict(
        self,
        home_stats: Dict,
        away_stats: Dict,
        home_goalie: Optional[Dict] = None,
        away_goalie: Optional[Dict] = None,
        over_under_line: float = 6.0
    ) -> Dict:
        """
        Full prediction for a game.

        Args:
            home_stats: Dict with keys 'gf_60', 'ga_60'
            away_stats: Dict with keys 'gf_60', 'ga_60'
            home_goalie: Dict with keys 'gsax', 'sv_pct' (optional)
            away_goalie: Dict with keys 'gsax', 'sv_pct' (optional)
            over_under_line: Total goals line for O/U calculation

        Returns:
            Dict with lambdas, win probs, score probs, O/U probs
        """
        # Extract goalie stats or use defaults
        home_goalie = home_goalie or {'gsax': 0, 'sv_pct': 0.905}
        away_goalie = away_goalie or {'gsax': 0, 'sv_pct': 0.905}

        # Calculate lambdas
        lambda_home, lambda_away = self.calculate_lambdas(
            home_gf_60=home_stats['gf_60'],
            home_ga_60=home_stats['ga_60'],
            away_gf_60=away_stats['gf_60'],
            away_ga_60=away_stats['ga_60'],
            home_goalie_gsax=home_goalie.get('gsax', 0),
            home_goalie_sv_pct=home_goalie.get('sv_pct', 0.905),
            away_goalie_gsax=away_goalie.get('gsax', 0),
            away_goalie_sv_pct=away_goalie.get('sv_pct', 0.905)
        )

        # Calculate probabilities
        win_probs = self.calculate_win_probabilities(lambda_home, lambda_away)
        score_probs = self.calculate_score_probabilities(lambda_home, lambda_away)
        ou_probs = self.calculate_over_under_prob(lambda_home, lambda_away, over_under_line)

        # Find most likely score
        most_likely = max(score_probs.items(), key=lambda x: x[1])

        return {
            'lambda_home': round(lambda_home, 2),
            'lambda_away': round(lambda_away, 2),
            'home_win_prob': round(win_probs['home_win'], 3),
            'away_win_prob': round(win_probs['away_win'], 3),
            'tie_prob': round(win_probs['tie'], 3),
            'most_likely_score': most_likely[0],
            'most_likely_score_prob': round(most_likely[1], 3),
            'over_prob': round(ou_probs['over'], 3),
            'under_prob': round(ou_probs['under'], 3),
            'expected_total': round(ou_probs['expected_total'], 2),
            'score_distribution': {k: round(v, 4) for k, v in sorted(
                score_probs.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]}  # Top 10 most likely scores
        }


# Convenience function for quick predictions
def quick_predict(
    home_gf: float,
    home_ga: float,
    away_gf: float,
    away_ga: float,
    ou_line: float = 6.0
) -> Dict:
    """Quick prediction with minimal input."""
    engine = PoissonEngine()
    return engine.predict(
        home_stats={'gf_60': home_gf, 'ga_60': home_ga},
        away_stats={'gf_60': away_gf, 'ga_60': away_ga},
        over_under_line=ou_line
    )
