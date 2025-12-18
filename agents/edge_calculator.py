"""
EdgeCalculatorAgent
Role: Compare model predictions to betting lines, calculate edges and optimal bet sizing
Inputs: Model probabilities, current betting odds
Outputs: Edge percentages, Kelly fractions, EV calculations, recommendations
"""
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Confidence(Enum):
    """Confidence tiers for recommendations"""
    STRONG = "STRONG"       # Edge > 5%, high model confidence
    MODERATE = "MODERATE"   # Edge 2-5%, good model confidence
    WEAK = "WEAK"           # Edge 1-2%, lower confidence
    NO_EDGE = "NO_EDGE"     # Edge < 1% or negative


@dataclass
class EdgeConfig:
    """Configuration for edge calculation"""
    min_edge_threshold: float = 0.01    # 1% minimum edge to consider
    max_kelly_fraction: float = 0.25    # Max 25% of bankroll
    kelly_divisor: float = 4            # Use 1/4 Kelly for safety
    juice_assumption: float = 0.05      # Assumed vig if odds not available


class EdgeCalculator:
    """
    Betting edge calculator for NHL predictions.

    Compares model probabilities to implied odds from betting lines
    to identify value bets. Calculates:
    - Edge percentage
    - Kelly criterion bet sizing
    - Expected value per bet
    - Confidence-based recommendations
    """

    def __init__(self, config: EdgeConfig = None):
        self.config = config or EdgeConfig()

    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """Convert American odds to decimal odds."""
        if american_odds > 0:
            return (american_odds / 100) + 1
        return (100 / abs(american_odds)) + 1

    @staticmethod
    def decimal_to_american(decimal_odds: float) -> int:
        """Convert decimal odds to American odds."""
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        return int(-100 / (decimal_odds - 1))

    @staticmethod
    def american_to_implied_prob(american_odds: int) -> float:
        """Convert American odds to implied probability."""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        return abs(american_odds) / (abs(american_odds) + 100)

    @staticmethod
    def prob_to_fair_american(prob: float) -> int:
        """Convert probability to fair American odds (no vig)."""
        if prob >= 0.5:
            return int(-100 * prob / (1 - prob))
        return int(100 * (1 - prob) / prob)

    def remove_vig(
        self,
        home_odds: int,
        away_odds: int
    ) -> Tuple[float, float]:
        """
        Remove vig from odds to get true implied probabilities.

        Returns:
            Tuple of (home_true_prob, away_true_prob)
        """
        home_implied = self.american_to_implied_prob(home_odds)
        away_implied = self.american_to_implied_prob(away_odds)

        # Total implied prob (includes vig)
        total = home_implied + away_implied

        # Remove vig by normalizing
        home_true = home_implied / total
        away_true = away_implied / total

        return home_true, away_true

    def calculate_edge(
        self,
        model_prob: float,
        market_odds: int
    ) -> float:
        """
        Calculate edge vs market.

        Edge = Model Prob - Implied Prob

        Positive edge means model thinks bet is undervalued.
        """
        implied_prob = self.american_to_implied_prob(market_odds)
        return model_prob - implied_prob

    def calculate_kelly(
        self,
        model_prob: float,
        decimal_odds: float
    ) -> float:
        """
        Calculate Kelly criterion fraction.

        Kelly = (bp - q) / b
        where:
            b = decimal odds - 1
            p = model probability of winning
            q = 1 - p

        Returns fraction of bankroll to bet (0 if negative EV).
        """
        b = decimal_odds - 1
        p = model_prob
        q = 1 - p

        kelly = (b * p - q) / b

        if kelly <= 0:
            return 0

        # Apply safety divisor and cap
        kelly = kelly / self.config.kelly_divisor
        kelly = min(kelly, self.config.max_kelly_fraction)

        return kelly

    def calculate_ev(
        self,
        model_prob: float,
        decimal_odds: float,
        stake: float = 100
    ) -> float:
        """
        Calculate expected value of a bet.

        EV = (prob * win_amount) - ((1-prob) * stake)
        """
        win_amount = stake * (decimal_odds - 1)
        ev = (model_prob * win_amount) - ((1 - model_prob) * stake)
        return ev

    def determine_confidence(
        self,
        edge: float,
        model_prob: float,
        variance: float = 0.05
    ) -> Confidence:
        """
        Determine confidence tier based on edge and model certainty.

        Args:
            edge: Edge percentage
            model_prob: Model's probability estimate
            variance: Model probability variance/uncertainty

        Returns:
            Confidence enum value
        """
        if edge < self.config.min_edge_threshold:
            return Confidence.NO_EDGE

        # Model confidence based on how decisive the probability is
        prob_confidence = abs(model_prob - 0.5) * 2  # 0 to 1 scale

        if edge >= 0.05 and prob_confidence >= 0.3:
            return Confidence.STRONG
        elif edge >= 0.02 and prob_confidence >= 0.15:
            return Confidence.MODERATE
        elif edge >= 0.01:
            return Confidence.WEAK

        return Confidence.NO_EDGE

    def generate_recommendation(
        self,
        edge: float,
        kelly: float,
        confidence: Confidence,
        bet_type: str
    ) -> str:
        """Generate human-readable recommendation."""
        if confidence == Confidence.NO_EDGE:
            return f"PASS - No edge on {bet_type}"

        action = {
            Confidence.STRONG: "STRONG",
            Confidence.MODERATE: "LEAN",
            Confidence.WEAK: "SLIGHT LEAN",
        }.get(confidence, "PASS")

        return f"{action} {bet_type.upper()}"

    def analyze_moneyline(
        self,
        home_prob: float,
        away_prob: float,
        home_odds: int,
        away_odds: int
    ) -> Dict:
        """
        Full analysis of moneyline bets.

        Args:
            home_prob: Model probability for home win
            away_prob: Model probability for away win
            home_odds: American odds for home ML
            away_odds: American odds for away ML

        Returns:
            Dict with edge analysis for both sides
        """
        # Calculate edges
        home_edge = self.calculate_edge(home_prob, home_odds)
        away_edge = self.calculate_edge(away_prob, away_odds)

        # Calculate Kelly fractions
        home_kelly = self.calculate_kelly(
            home_prob,
            self.american_to_decimal(home_odds)
        )
        away_kelly = self.calculate_kelly(
            away_prob,
            self.american_to_decimal(away_odds)
        )

        # Calculate EVs
        home_ev = self.calculate_ev(
            home_prob,
            self.american_to_decimal(home_odds)
        )
        away_ev = self.calculate_ev(
            away_prob,
            self.american_to_decimal(away_odds)
        )

        # Determine confidence
        home_conf = self.determine_confidence(home_edge, home_prob)
        away_conf = self.determine_confidence(away_edge, away_prob)

        # Best bet
        if home_edge > away_edge and home_edge > self.config.min_edge_threshold:
            best_bet = 'home'
            best_conf = home_conf
        elif away_edge > self.config.min_edge_threshold:
            best_bet = 'away'
            best_conf = away_conf
        else:
            best_bet = 'pass'
            best_conf = Confidence.NO_EDGE

        return {
            'home': {
                'edge': round(home_edge, 4),
                'kelly': round(home_kelly, 4),
                'ev_per_100': round(home_ev, 2),
                'confidence': home_conf.value,
                'recommendation': self.generate_recommendation(
                    home_edge, home_kelly, home_conf, 'Home ML'
                ),
            },
            'away': {
                'edge': round(away_edge, 4),
                'kelly': round(away_kelly, 4),
                'ev_per_100': round(away_ev, 2),
                'confidence': away_conf.value,
                'recommendation': self.generate_recommendation(
                    away_edge, away_kelly, away_conf, 'Away ML'
                ),
            },
            'best_bet': best_bet,
            'best_confidence': best_conf.value,
        }

    def analyze_over_under(
        self,
        over_prob: float,
        total_line: float,
        over_odds: int = -110,
        under_odds: int = -110
    ) -> Dict:
        """
        Full analysis of over/under bets.

        Args:
            over_prob: Model probability for over
            total_line: The total goals line
            over_odds: American odds for over
            under_odds: American odds for under

        Returns:
            Dict with edge analysis for both sides
        """
        under_prob = 1 - over_prob

        # Calculate edges
        over_edge = self.calculate_edge(over_prob, over_odds)
        under_edge = self.calculate_edge(under_prob, under_odds)

        # Calculate Kelly fractions
        over_kelly = self.calculate_kelly(
            over_prob,
            self.american_to_decimal(over_odds)
        )
        under_kelly = self.calculate_kelly(
            under_prob,
            self.american_to_decimal(under_odds)
        )

        # Calculate EVs
        over_ev = self.calculate_ev(
            over_prob,
            self.american_to_decimal(over_odds)
        )
        under_ev = self.calculate_ev(
            under_prob,
            self.american_to_decimal(under_odds)
        )

        # Determine confidence
        over_conf = self.determine_confidence(over_edge, over_prob)
        under_conf = self.determine_confidence(under_edge, under_prob)

        # Best bet
        if over_edge > under_edge and over_edge > self.config.min_edge_threshold:
            best_bet = 'over'
            best_conf = over_conf
        elif under_edge > self.config.min_edge_threshold:
            best_bet = 'under'
            best_conf = under_conf
        else:
            best_bet = 'pass'
            best_conf = Confidence.NO_EDGE

        return {
            'line': total_line,
            'over': {
                'edge': round(over_edge, 4),
                'kelly': round(over_kelly, 4),
                'ev_per_100': round(over_ev, 2),
                'confidence': over_conf.value,
                'recommendation': self.generate_recommendation(
                    over_edge, over_kelly, over_conf, f'Over {total_line}'
                ),
            },
            'under': {
                'edge': round(under_edge, 4),
                'kelly': round(under_kelly, 4),
                'ev_per_100': round(under_ev, 2),
                'confidence': under_conf.value,
                'recommendation': self.generate_recommendation(
                    under_edge, under_kelly, under_conf, f'Under {total_line}'
                ),
            },
            'best_bet': best_bet,
            'best_confidence': best_conf.value,
        }

    def full_game_analysis(
        self,
        home_win_prob: float,
        over_prob: float,
        home_ml: int,
        away_ml: int,
        total_line: float,
        over_odds: int = -110,
        under_odds: int = -110,
        sixty_min_home_prob: Optional[float] = None,
        sixty_min_home_ml: Optional[int] = None,
        sixty_min_away_ml: Optional[int] = None
    ) -> Dict:
        """
        Complete betting analysis for a game.

        Returns comprehensive edge analysis for all bet types.
        """
        away_win_prob = 1 - home_win_prob

        result = {
            'moneyline': self.analyze_moneyline(
                home_win_prob, away_win_prob,
                home_ml, away_ml
            ),
            'over_under': self.analyze_over_under(
                over_prob, total_line,
                over_odds, under_odds
            ),
        }

        # 60-minute analysis if available
        if sixty_min_home_prob and sixty_min_home_ml and sixty_min_away_ml:
            sixty_min_away_prob = 1 - sixty_min_home_prob
            result['sixty_min'] = self.analyze_moneyline(
                sixty_min_home_prob, sixty_min_away_prob,
                sixty_min_home_ml, sixty_min_away_ml
            )

        # Overall recommendation
        best_bets = []
        for bet_type, analysis in result.items():
            if analysis.get('best_bet') != 'pass':
                best_bets.append({
                    'type': bet_type,
                    'side': analysis['best_bet'],
                    'confidence': analysis['best_confidence'],
                    'edge': analysis[analysis['best_bet']]['edge'],
                })

        # Sort by edge
        best_bets.sort(key=lambda x: x['edge'], reverse=True)
        result['top_plays'] = best_bets[:3]

        return result
