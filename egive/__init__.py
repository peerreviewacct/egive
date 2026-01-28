"""
egive: A Python package for EGIVE, an efficient variable importance and interaction detection method for black-box ML models
"""

__version__ = "0.1.0"

from .core import run_egive
from .utils import neg_auc, rmse, mse, mae, retrieve_quantiles, get_neighborhood_points, logit_catch_singular, propensity_filters_and_weights, pdp_propensity_filters_and_weights, cutpoints, feature_importance_scores

__all__ = ['egive', 'run_egive', 'neg_auc', 'rmse', 'mse', 'mae','retrieve_quantiles', 'get_neighborhood_points', 'logit_catch_singular', 'propensity_filters_and_weights', 'pdp_propensity_filters_and_weights', 'cutpoints', 'feature_importance_scores']
