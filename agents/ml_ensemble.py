"""
MLEnsembleAgent
Role: Gradient boosting ensemble for enhanced win probability and totals prediction
Inputs: Engineered features, Poisson baseline predictions
Outputs: ML-adjusted probabilities, feature importances, confidence scores
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pickle
from pathlib import Path

# ML imports (graceful fallback if not installed)
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler


@dataclass
class MLConfig:
    """Configuration for ML models"""
    use_xgboost: bool = True
    use_lightgbm: bool = True
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    random_state: int = 42
    cv_folds: int = 5


class MLEnsemble:
    """
    Machine learning ensemble for NHL game prediction.

    Combines multiple models:
    - XGBoost (primary classifier)
    - LightGBM (fast, accurate)
    - Gradient Boosting (sklearn fallback)
    - Stacking meta-learner

    The ensemble incorporates Poisson baseline predictions as features,
    allowing it to learn adjustments rather than predictions from scratch.
    """

    # Features used for ML prediction
    FEATURE_COLUMNS = [
        # Team offensive metrics
        'home_gf_60', 'away_gf_60',
        'home_xgf_60', 'away_xgf_60',

        # Team defensive metrics
        'home_ga_60', 'away_ga_60',
        'home_xga_60', 'away_xga_60',

        # Possession metrics
        'home_cf_pct', 'away_cf_pct',
        'home_scf_pct', 'away_scf_pct',

        # Special teams
        'home_pp_pct', 'away_pp_pct',
        'home_pk_pct', 'away_pk_pct',

        # Goalie metrics
        'home_goalie_sv_pct', 'away_goalie_sv_pct',
        'home_goalie_gsax', 'away_goalie_gsax',

        # Form
        'home_l10_win_pct', 'away_l10_win_pct',
        'home_l10_goal_diff', 'away_l10_goal_diff',

        # Rest
        'home_days_rest', 'away_days_rest',
        'home_is_b2b', 'away_is_b2b',

        # Differentials
        'gf_60_diff', 'xgf_diff', 'cf_pct_diff',

        # Poisson baseline (key feature)
        'poisson_home_win_prob',
        'poisson_expected_total',
    ]

    def __init__(self, config: MLConfig = None):
        self.config = config or MLConfig()
        self.models = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        self._feature_importances = {}

    def _init_models(self):
        """Initialize model instances."""
        models = {}

        # XGBoost
        if HAS_XGBOOST and self.config.use_xgboost:
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
            )

        # LightGBM
        if HAS_LIGHTGBM and self.config.use_lightgbm:
            models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state,
                verbose=-1,
            )

        # Fallback: sklearn GradientBoosting
        models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            random_state=self.config.random_state,
        )

        # Meta-learner for stacking
        models['meta'] = LogisticRegression(random_state=self.config.random_state)

        return models

    def _prepare_features(
        self,
        X: pd.DataFrame,
        fit_scaler: bool = False
    ) -> np.ndarray:
        """
        Prepare features for ML models.

        Fills missing values, scales features, selects relevant columns.
        """
        # Select available feature columns
        available_cols = [c for c in self.FEATURE_COLUMNS if c in X.columns]
        X_subset = X[available_cols].copy()

        # Fill missing values with median
        X_subset = X_subset.fillna(X_subset.median())

        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X_subset)
        else:
            X_scaled = self.scaler.transform(X_subset)

        return X_scaled

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        validate: bool = True
    ) -> Dict:
        """
        Train the ensemble on historical data.

        Args:
            X: Feature DataFrame with columns matching FEATURE_COLUMNS
            y: Binary target (1 = home win, 0 = away win)
            validate: Whether to run cross-validation

        Returns:
            Dict with training metrics
        """
        self.models = self._init_models()

        # Prepare features
        X_scaled = self._prepare_features(X, fit_scaler=True)

        metrics = {'models': {}}

        # Train base models
        for name, model in self.models.items():
            if name == 'meta':
                continue

            model.fit(X_scaled, y)

            # Cross-validation
            if validate:
                cv = TimeSeriesSplit(n_splits=self.config.cv_folds)
                cv_scores = cross_val_score(
                    model, X_scaled, y,
                    cv=cv, scoring='accuracy'
                )
                metrics['models'][name] = {
                    'cv_accuracy': round(cv_scores.mean(), 4),
                    'cv_std': round(cv_scores.std(), 4),
                }

            # Feature importances
            if hasattr(model, 'feature_importances_'):
                available_cols = [c for c in self.FEATURE_COLUMNS if c in X.columns]
                self._feature_importances[name] = dict(
                    zip(available_cols, model.feature_importances_)
                )

        # Train meta-learner on base model predictions
        base_preds = self._get_base_predictions(X_scaled)
        self.models['meta'].fit(base_preds, y)

        self.is_fitted = True

        # Overall metrics
        ensemble_preds = self.predict_proba(X)[:, 1]
        metrics['ensemble'] = {
            'brier_score': round(brier_score_loss(y, ensemble_preds), 4),
            'log_loss': round(log_loss(y, ensemble_preds), 4),
            'accuracy': round(accuracy_score(y, (ensemble_preds > 0.5).astype(int)), 4),
        }

        return metrics

    def _get_base_predictions(self, X_scaled: np.ndarray) -> np.ndarray:
        """Get predictions from base models for meta-learner."""
        preds = []
        for name, model in self.models.items():
            if name == 'meta':
                continue
            preds.append(model.predict_proba(X_scaled)[:, 1])
        return np.column_stack(preds)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict win probabilities.

        Args:
            X: Feature DataFrame

        Returns:
            Array of shape (n_samples, 2) with [away_win_prob, home_win_prob]
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X_scaled = self._prepare_features(X)
        base_preds = self._get_base_predictions(X_scaled)
        proba = self.models['meta'].predict_proba(base_preds)
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict binary outcome (1 = home win, 0 = away win)."""
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

    def predict_single_game(
        self,
        features: Dict,
        poisson_home_prob: float = 0.5,
        poisson_total: float = 6.0
    ) -> Dict:
        """
        Predict outcome for a single game.

        Args:
            features: Dict of feature values
            poisson_home_prob: Poisson model's home win probability
            poisson_total: Poisson model's expected total goals

        Returns:
            Dict with prediction details
        """
        # Add Poisson baselines
        features['poisson_home_win_prob'] = poisson_home_prob
        features['poisson_expected_total'] = poisson_total

        # Create single-row DataFrame
        X = pd.DataFrame([features])

        # Fill missing with defaults
        for col in self.FEATURE_COLUMNS:
            if col not in X.columns:
                X[col] = 0.5 if 'pct' in col or 'prob' in col else 3.0

        proba = self.predict_proba(X)[0]

        return {
            'home_win_prob': round(proba[1], 4),
            'away_win_prob': round(proba[0], 4),
            'prediction': 'home' if proba[1] > 0.5 else 'away',
            'confidence': round(abs(proba[1] - 0.5) * 2, 4),  # 0 to 1 scale
            'adjustment_from_poisson': round(proba[1] - poisson_home_prob, 4),
        }

    def get_feature_importances(self) -> Dict[str, Dict]:
        """Get feature importances from base models."""
        return self._feature_importances

    def save(self, path: str):
        """Save trained ensemble to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            'models': self.models,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'feature_importances': self._feature_importances,
            'config': self.config,
        }

        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

    def load(self, path: str):
        """Load trained ensemble from disk."""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)

        self.models = save_data['models']
        self.scaler = save_data['scaler']
        self.is_fitted = save_data['is_fitted']
        self._feature_importances = save_data['feature_importances']
        self.config = save_data['config']


class TotalsPredictor:
    """
    Regression model for predicting total goals.

    Uses similar ensemble approach but for regression target.
    """

    def __init__(self, config: MLConfig = None):
        self.config = config or MLConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """Train totals predictor."""
        if HAS_XGBOOST:
            self.model = xgb.XGBRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state,
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state,
            )

        # Prepare and fit
        X_scaled = self.scaler.fit_transform(X.fillna(X.median()))
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        # Metrics
        preds = self.model.predict(X_scaled)
        rmse = np.sqrt(np.mean((preds - y) ** 2))
        mae = np.mean(np.abs(preds - y))

        return {'rmse': round(rmse, 3), 'mae': round(mae, 3)}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict total goals."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")

        X_scaled = self.scaler.transform(X.fillna(X.median()))
        return self.model.predict(X_scaled)


# Convenience function for quick predictions without training
def quick_ml_adjustment(
    poisson_home_prob: float,
    home_xgf: float,
    away_xgf: float,
    home_cf_pct: float,
    away_cf_pct: float
) -> float:
    """
    Quick heuristic adjustment to Poisson probability based on xG and Corsi.

    This is a simplified version that doesn't require training data.
    For production, use the full MLEnsemble with trained models.
    """
    # xG differential adjustment
    xg_diff = home_xgf - away_xgf
    xg_adj = xg_diff * 0.05  # ~5% per goal of xG difference

    # Corsi adjustment
    cf_diff = home_cf_pct - away_cf_pct
    cf_adj = cf_diff * 0.002  # ~0.2% per point of Corsi difference

    # Combine adjustments
    adjusted_prob = poisson_home_prob + xg_adj + cf_adj

    # Clip to valid probability range
    return max(0.15, min(0.85, adjusted_prob))
