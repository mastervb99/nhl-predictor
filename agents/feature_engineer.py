"""
FeatureEngineerAgent
Role: Transform raw team/goalie stats into model-ready features
Inputs: Raw stats from multiple seasons, team identifiers
Outputs: Weighted feature vectors for prediction models
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SeasonWeights:
    """Weights for multi-season aggregation"""
    current: float = 0.50      # 2024-25
    prior_1: float = 0.30      # 2023-24
    prior_2: float = 0.20      # 2022-23


class FeatureEngineer:
    """
    Feature engineering for NHL prediction model.

    Handles:
    - Multi-season weighted averages
    - Home/Away splits
    - Rolling form indicators
    - Goalie adjustments
    - Feature normalization
    """

    # Core features from raw stats
    TEAM_FEATURES = [
        'gf_60', 'ga_60',           # Goals per 60
        'xgf_60', 'xga_60',         # Expected goals per 60
        'cf_pct',                    # Corsi for %
        'scf_pct',                   # Scoring chances for %
        'pp_pct', 'pk_pct',         # Special teams
        'sh_pct', 'sv_pct',         # Shooting/Save %
        'fo_pct',                    # Faceoff %
    ]

    GOALIE_FEATURES = [
        'sv_pct', 'gaa',            # Basic stats
        'gsax', 'gsax_60',          # Goals saved above expected
        'es_sv_pct',                # Even strength save %
    ]

    def __init__(self, weights: SeasonWeights = None):
        self.weights = weights or SeasonWeights()
        self._league_averages = None

    def set_league_averages(self, averages: Dict[str, float]):
        """Set league averages for normalization."""
        self._league_averages = averages

    def calculate_weighted_stats(
        self,
        current_stats: Dict,
        prior_1_stats: Optional[Dict] = None,
        prior_2_stats: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Calculate weighted average of stats across seasons.

        Args:
            current_stats: Current season stats
            prior_1_stats: Prior season stats (optional)
            prior_2_stats: Two seasons ago stats (optional)

        Returns:
            Dict of weighted feature values
        """
        weighted = {}

        # Determine available weights
        available_weight = self.weights.current
        if prior_1_stats:
            available_weight += self.weights.prior_1
        if prior_2_stats:
            available_weight += self.weights.prior_2

        for feature in self.TEAM_FEATURES:
            if feature not in current_stats:
                continue

            weighted_sum = current_stats[feature] * self.weights.current

            if prior_1_stats and feature in prior_1_stats:
                weighted_sum += prior_1_stats[feature] * self.weights.prior_1

            if prior_2_stats and feature in prior_2_stats:
                weighted_sum += prior_2_stats[feature] * self.weights.prior_2

            # Normalize by available weight
            weighted[feature] = weighted_sum / available_weight

        return weighted

    def calculate_home_away_adjustment(
        self,
        team_stats: Dict,
        is_home: bool
    ) -> Dict[str, float]:
        """
        Adjust stats based on home/away splits.

        If splits are available, uses them. Otherwise applies
        league-average home/away adjustment factors.
        """
        adjusted = team_stats.copy()

        # Default home/away adjustment factors (based on historical data)
        HOME_GF_BOOST = 1.04   # ~4% more goals at home
        HOME_GA_REDUCE = 0.96  # ~4% fewer goals allowed at home

        if is_home:
            if 'home_gf_60' in team_stats:
                adjusted['gf_60'] = team_stats['home_gf_60']
                adjusted['ga_60'] = team_stats['home_ga_60']
            else:
                adjusted['gf_60'] = team_stats['gf_60'] * HOME_GF_BOOST
                adjusted['ga_60'] = team_stats['ga_60'] * HOME_GA_REDUCE
        else:
            if 'away_gf_60' in team_stats:
                adjusted['gf_60'] = team_stats['away_gf_60']
                adjusted['ga_60'] = team_stats['away_ga_60']
            else:
                adjusted['gf_60'] = team_stats['gf_60'] / HOME_GF_BOOST
                adjusted['ga_60'] = team_stats['ga_60'] / HOME_GA_REDUCE

        return adjusted

    def calculate_form_features(
        self,
        recent_games: List[Dict],
        n_games: int = 10
    ) -> Dict[str, float]:
        """
        Calculate recent form indicators from last N games.

        Args:
            recent_games: List of game results (most recent first)
            n_games: Number of games to consider

        Returns:
            Dict with form features
        """
        games = recent_games[:n_games]
        if not games:
            return {
                'l10_win_pct': 0.5,
                'l10_goal_diff': 0,
                'l10_gf_avg': 3.0,
                'l10_ga_avg': 3.0,
            }

        wins = sum(1 for g in games if g.get('result') == 'W')
        gf = sum(g.get('goals_for', 3) for g in games)
        ga = sum(g.get('goals_against', 3) for g in games)

        return {
            'l10_win_pct': wins / len(games),
            'l10_goal_diff': (gf - ga) / len(games),
            'l10_gf_avg': gf / len(games),
            'l10_ga_avg': ga / len(games),
        }

    def calculate_rest_features(
        self,
        days_rest: int,
        is_back_to_back: bool
    ) -> Dict[str, float]:
        """
        Calculate rest/schedule features.

        Args:
            days_rest: Days since last game
            is_back_to_back: True if second game in 2 days

        Returns:
            Dict with rest features
        """
        # Rest impact factors (based on historical data)
        # B2B teams score ~5% fewer goals
        # Well-rested teams (3+ days) score ~3% more

        rest_factor = 1.0
        if is_back_to_back:
            rest_factor = 0.95
        elif days_rest >= 3:
            rest_factor = 1.03

        return {
            'days_rest': days_rest,
            'is_b2b': float(is_back_to_back),
            'rest_factor': rest_factor,
        }

    def engineer_goalie_features(
        self,
        goalie_stats: Dict,
        league_avg_sv_pct: float = 0.905
    ) -> Dict[str, float]:
        """
        Engineer goalie-specific features.

        Args:
            goalie_stats: Raw goalie statistics
            league_avg_sv_pct: League average save percentage

        Returns:
            Dict with engineered goalie features
        """
        sv_pct = goalie_stats.get('sv_pct', league_avg_sv_pct)
        gsax = goalie_stats.get('gsax', 0)
        es_sv_pct = goalie_stats.get('es_sv_pct', sv_pct)

        # Calculate goalie quality index (GQI)
        # Combines save % and goals saved above expected
        sv_diff = sv_pct - league_avg_sv_pct
        gqi = (sv_diff * 100) + (gsax / 10)  # Normalize GSAX

        # Calculate opponent scoring adjustment
        # Better goalies reduce opponent's expected goals
        opponent_adj = 1 - (gqi / 100)
        opponent_adj = max(0.8, min(1.2, opponent_adj))  # Clip to reasonable range

        return {
            'sv_pct': sv_pct,
            'gsax': gsax,
            'es_sv_pct': es_sv_pct,
            'goalie_quality_index': gqi,
            'opponent_scoring_adj': opponent_adj,
        }

    def build_game_features(
        self,
        home_team_stats: Dict,
        away_team_stats: Dict,
        home_goalie_stats: Optional[Dict] = None,
        away_goalie_stats: Optional[Dict] = None,
        home_recent_games: Optional[List[Dict]] = None,
        away_recent_games: Optional[List[Dict]] = None,
        home_rest_days: int = 2,
        away_rest_days: int = 2,
        home_b2b: bool = False,
        away_b2b: bool = False
    ) -> Dict[str, float]:
        """
        Build complete feature vector for a game.

        Returns:
            Dict with all features needed for prediction
        """
        features = {}

        # Team stats with home/away adjustment
        home_adj = self.calculate_home_away_adjustment(home_team_stats, is_home=True)
        away_adj = self.calculate_home_away_adjustment(away_team_stats, is_home=False)

        # Add team features with prefix
        for key, value in home_adj.items():
            features[f'home_{key}'] = value
        for key, value in away_adj.items():
            features[f'away_{key}'] = value

        # Goalie features
        if home_goalie_stats:
            goalie_feats = self.engineer_goalie_features(home_goalie_stats)
            for key, value in goalie_feats.items():
                features[f'home_goalie_{key}'] = value

        if away_goalie_stats:
            goalie_feats = self.engineer_goalie_features(away_goalie_stats)
            for key, value in goalie_feats.items():
                features[f'away_goalie_{key}'] = value

        # Form features
        if home_recent_games:
            form_feats = self.calculate_form_features(home_recent_games)
            for key, value in form_feats.items():
                features[f'home_{key}'] = value

        if away_recent_games:
            form_feats = self.calculate_form_features(away_recent_games)
            for key, value in form_feats.items():
                features[f'away_{key}'] = value

        # Rest features
        home_rest = self.calculate_rest_features(home_rest_days, home_b2b)
        away_rest = self.calculate_rest_features(away_rest_days, away_b2b)
        for key, value in home_rest.items():
            features[f'home_{key}'] = value
        for key, value in away_rest.items():
            features[f'away_{key}'] = value

        # Derived differential features
        features['gf_60_diff'] = features.get('home_gf_60', 3) - features.get('away_gf_60', 3)
        features['ga_60_diff'] = features.get('home_ga_60', 3) - features.get('away_ga_60', 3)
        features['xgf_diff'] = features.get('home_xgf_60', 3) - features.get('away_xgf_60', 3)
        features['cf_pct_diff'] = features.get('home_cf_pct', 50) - features.get('away_cf_pct', 50)
        features['pp_diff'] = features.get('home_pp_pct', 20) - features.get('away_pp_pct', 20)

        return features

    def normalize_features(
        self,
        features: Dict[str, float],
        method: str = 'zscore'
    ) -> Dict[str, float]:
        """
        Normalize features for ML model input.

        Args:
            features: Raw feature values
            method: 'zscore' or 'minmax'

        Returns:
            Normalized feature values
        """
        if not self._league_averages:
            return features  # No normalization without league averages

        normalized = {}
        for key, value in features.items():
            # Extract base feature name (remove home_/away_ prefix)
            base_key = key.replace('home_', '').replace('away_', '').replace('goalie_', '')

            if base_key in self._league_averages:
                mean = self._league_averages[base_key]['mean']
                std = self._league_averages[base_key]['std']

                if method == 'zscore' and std > 0:
                    normalized[key] = (value - mean) / std
                else:
                    normalized[key] = value
            else:
                normalized[key] = value

        return normalized
