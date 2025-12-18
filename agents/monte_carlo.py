"""
MonteCarloSimulator Agent
Role: Run Monte Carlo simulations to generate probability distributions
Inputs: Poisson lambdas for each team
Outputs: Win probabilities, score distributions, O/U probabilities, 60-min results
"""
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation"""
    n_simulations: int = 10_000
    overtime_prob: float = 0.23       # ~23% of NHL games go to OT
    shootout_prob: float = 0.50       # Of OT games, ~50% go to shootout
    home_ot_advantage: float = 0.52   # Slight home advantage in OT
    random_seed: Optional[int] = None


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for NHL game outcomes.

    Simulates thousands of games using Poisson-distributed goals,
    then handles overtime/shootout resolution to produce:
    - Win probability distributions
    - Score probability matrices
    - Over/Under probabilities
    - 60-minute result probabilities (for 60-min moneyline bets)
    """

    def __init__(self, config: MonteCarloConfig = None):
        self.config = config or MonteCarloConfig()
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

    def simulate_regulation(
        self,
        lambda_home: float,
        lambda_away: float,
        n_sims: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate regulation (60-minute) scores.

        Returns:
            Tuple of (home_goals_array, away_goals_array)
        """
        n = n_sims or self.config.n_simulations
        home_goals = np.random.poisson(lambda_home, n)
        away_goals = np.random.poisson(lambda_away, n)
        return home_goals, away_goals

    def resolve_overtime(
        self,
        tied_mask: np.ndarray,
        home_goals: np.ndarray,
        away_goals: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resolve tied games with OT/shootout logic.

        Returns:
            Tuple of (final_home_goals, final_away_goals, went_to_shootout)
        """
        n_tied = tied_mask.sum()
        if n_tied == 0:
            return home_goals, away_goals, np.zeros(len(home_goals), dtype=bool)

        # Copy arrays to avoid mutation
        final_home = home_goals.copy()
        final_away = away_goals.copy()
        shootout_mask = np.zeros(len(home_goals), dtype=bool)

        # Determine OT/SO outcomes for tied games
        tied_indices = np.where(tied_mask)[0]

        for idx in tied_indices:
            # Determine if shootout
            is_shootout = np.random.random() < self.config.shootout_prob
            shootout_mask[idx] = is_shootout

            # Determine winner (slight home advantage)
            home_wins = np.random.random() < self.config.home_ot_advantage

            if home_wins:
                final_home[idx] += 1
            else:
                final_away[idx] += 1

        return final_home, final_away, shootout_mask

    def simulate_games(
        self,
        lambda_home: float,
        lambda_away: float
    ) -> Dict:
        """
        Run full Monte Carlo simulation.

        Returns comprehensive results including:
        - Regulation results (60-min)
        - Final results (with OT/SO)
        - Score distributions
        - Over/Under analysis
        """
        n_sims = self.config.n_simulations

        # Simulate regulation
        reg_home, reg_away = self.simulate_regulation(lambda_home, lambda_away)

        # Identify tied games
        tied_mask = reg_home == reg_away

        # Resolve OT/SO
        final_home, final_away, shootout_mask = self.resolve_overtime(
            tied_mask, reg_home, reg_away
        )

        # Calculate results
        results = self._calculate_results(
            reg_home, reg_away,
            final_home, final_away,
            tied_mask, shootout_mask
        )

        return results

    def _calculate_results(
        self,
        reg_home: np.ndarray,
        reg_away: np.ndarray,
        final_home: np.ndarray,
        final_away: np.ndarray,
        tied_mask: np.ndarray,
        shootout_mask: np.ndarray
    ) -> Dict:
        """Calculate all result metrics from simulation arrays."""
        n_sims = len(reg_home)

        # 60-minute (regulation) results
        reg_home_wins = (reg_home > reg_away).sum()
        reg_away_wins = (reg_away > reg_home).sum()
        reg_ties = tied_mask.sum()

        # Final results
        final_home_wins = (final_home > final_away).sum()
        final_away_wins = (final_away > final_home).sum()

        # OT wins (not shootout)
        ot_mask = tied_mask & ~shootout_mask
        home_ot_wins = (ot_mask & (final_home > final_away)).sum()
        away_ot_wins = (ot_mask & (final_away > final_home)).sum()

        # SO wins
        home_so_wins = (shootout_mask & (final_home > final_away)).sum()
        away_so_wins = (shootout_mask & (final_away > final_home)).sum()

        # Total goals
        reg_totals = reg_home + reg_away
        expected_total = reg_totals.mean()

        # Score distribution (regulation)
        score_counts = Counter(
            f"{h}-{a}" for h, a in zip(reg_home, reg_away)
        )
        score_probs = {
            score: count / n_sims
            for score, count in score_counts.most_common(20)
        }

        # Most likely score
        most_likely = score_counts.most_common(1)[0]

        return {
            # 60-minute results
            'sixty_min': {
                'home_win_prob': round(reg_home_wins / n_sims, 4),
                'away_win_prob': round(reg_away_wins / n_sims, 4),
                'tie_prob': round(reg_ties / n_sims, 4),
            },

            # Final results (regulation + OT/SO)
            'final': {
                'home_win_prob': round(final_home_wins / n_sims, 4),
                'away_win_prob': round(final_away_wins / n_sims, 4),
                'home_regulation_wins': round(reg_home_wins / n_sims, 4),
                'away_regulation_wins': round(reg_away_wins / n_sims, 4),
                'home_ot_wins': round(home_ot_wins / n_sims, 4),
                'away_ot_wins': round(away_ot_wins / n_sims, 4),
                'home_so_wins': round(home_so_wins / n_sims, 4),
                'away_so_wins': round(away_so_wins / n_sims, 4),
            },

            # Totals
            'totals': {
                'expected_total': round(expected_total, 2),
                'std_dev': round(reg_totals.std(), 2),
                'median': int(np.median(reg_totals)),
            },

            # Score distribution
            'scores': {
                'most_likely': most_likely[0],
                'most_likely_prob': round(most_likely[1] / n_sims, 4),
                'distribution': score_probs,
            },

            # Simulation metadata
            'meta': {
                'n_simulations': n_sims,
                'overtime_games': round(reg_ties / n_sims, 4),
                'shootout_games': round(shootout_mask.sum() / n_sims, 4),
            }
        }

    def calculate_over_under(
        self,
        lambda_home: float,
        lambda_away: float,
        lines: list = None
    ) -> Dict[float, Dict]:
        """
        Calculate over/under probabilities for multiple lines.

        Args:
            lambda_home: Expected home goals
            lambda_away: Expected away goals
            lines: List of O/U lines to evaluate (default: [5.0, 5.5, 6.0, 6.5, 7.0])

        Returns:
            Dict mapping line to over/under/push probabilities
        """
        lines = lines or [5.0, 5.5, 6.0, 6.5, 7.0]

        # Run simulation
        home_goals, away_goals = self.simulate_regulation(lambda_home, lambda_away)
        totals = home_goals + away_goals

        results = {}
        for line in lines:
            over = (totals > line).sum()
            under = (totals < line).sum()
            push = (totals == line).sum()
            n = len(totals)

            results[line] = {
                'over_prob': round(over / n, 4),
                'under_prob': round(under / n, 4),
                'push_prob': round(push / n, 4) if line == int(line) else 0,
            }

        return results

    def simulate_with_variance(
        self,
        lambda_home: float,
        lambda_away: float,
        lambda_std: float = 0.3
    ) -> Dict:
        """
        Simulate with added variance in lambda values.

        This accounts for uncertainty in the underlying model parameters
        by sampling lambdas from a normal distribution centered on the
        point estimate.

        Args:
            lambda_home: Mean expected home goals
            lambda_away: Mean expected away goals
            lambda_std: Standard deviation for lambda sampling

        Returns:
            Simulation results with uncertainty bounds
        """
        n_sims = self.config.n_simulations

        # Sample lambdas with variance
        lambda_home_samples = np.maximum(0.5, np.random.normal(
            lambda_home, lambda_std, n_sims
        ))
        lambda_away_samples = np.maximum(0.5, np.random.normal(
            lambda_away, lambda_std, n_sims
        ))

        # Simulate each game with its sampled lambdas
        home_goals = np.array([
            np.random.poisson(lh) for lh in lambda_home_samples
        ])
        away_goals = np.array([
            np.random.poisson(la) for la in lambda_away_samples
        ])

        # Resolve ties
        tied_mask = home_goals == away_goals
        final_home, final_away, shootout_mask = self.resolve_overtime(
            tied_mask, home_goals, away_goals
        )

        # Calculate results
        results = self._calculate_results(
            home_goals, away_goals,
            final_home, final_away,
            tied_mask, shootout_mask
        )

        # Add uncertainty bounds
        home_win_samples = (final_home > final_away)
        results['uncertainty'] = {
            'home_win_95_ci': (
                round(np.percentile(home_win_samples.cumsum() / np.arange(1, n_sims + 1), 2.5), 4),
                round(np.percentile(home_win_samples.cumsum() / np.arange(1, n_sims + 1), 97.5), 4)
            ),
            'lambda_variance_used': lambda_std,
        }

        return results


def run_simulation(
    lambda_home: float,
    lambda_away: float,
    n_sims: int = 10000,
    ou_lines: list = None
) -> Dict:
    """
    Convenience function for running a full simulation.

    Args:
        lambda_home: Expected home goals from Poisson model
        lambda_away: Expected away goals from Poisson model
        n_sims: Number of simulations (default 10000)
        ou_lines: O/U lines to evaluate

    Returns:
        Complete simulation results
    """
    config = MonteCarloConfig(n_simulations=n_sims)
    simulator = MonteCarloSimulator(config)

    results = simulator.simulate_games(lambda_home, lambda_away)

    if ou_lines:
        results['over_under'] = simulator.calculate_over_under(
            lambda_home, lambda_away, ou_lines
        )

    return results
