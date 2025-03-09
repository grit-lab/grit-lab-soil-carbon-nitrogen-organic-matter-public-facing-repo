"""
File: model.py
Author: Nayma Nur
Description: Defines and configures machine learning models including hyperparameter settings using
             RandomizedSearchCV. This file centralizes model configurations to streamline the management
             of model definitions.
"""



from utils import nrmse
from sklearn.metrics import make_scorer
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform, randint as sp_randint



def elastic_net_model_hyperparameters(X, y):
    """
    Performs hyperparameter tuning for an ElasticNet model using RandomizedSearchCV.

    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        y (numpy.ndarray): Target vector of shape (n_samples,).

    Returns:
        sklearn.pipeline.Pipeline: The best ElasticNet model found by RandomizedSearchCV.
    """

    # Define the parameter distribution
    param_dist = {
        'elasticnet__alpha': loguniform(0.001, 200),  # Expanding the alpha range slightly and using log scale
        'elasticnet__l1_ratio': uniform(0.1, 0.9)  # Uniform distribution over the range from 0.1 to 1.0

    }

    # Create a pipeline with StandardScaler and ElasticNet
    pipeline = make_pipeline(
        StandardScaler(),  # Applies standard scaling to the features
        ElasticNet(max_iter=1000000,  random_state=42)  # ElasticNet model

    )

    # Initialize RandomizedSearchCV for hyperparameter tuning
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=100,  # Number of parameter settings that are sampled
        cv=5,
        scoring=make_scorer(nrmse, greater_is_better=False),
        random_state=42,
        n_jobs=-1
    )

    # Fit RandomizedSearchCV to find the best model
    random_search.fit(X, y)

    # Retrieve the best model
    best_model = random_search.best_estimator_

    return best_model





def gradient_boosting_tree_model_hyperparameters(X, y):
    """
    Performs hyperparameter tuning for a Gradient Boosting Regressor model using RandomizedSearchCV.

    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        y (numpy.ndarray): Target vector of shape (n_samples,).

    Returns:
        sklearn.ensemble.GradientBoostingRegressor: The best Gradient Boosting Regressor model found by RandomizedSearchCV.
    """

    # Define the parameter distribution for Gradient Boosting Regressor
    param_dist = {

        'n_estimators': sp_randint(500, 700), # Number of boosting stages
        'learning_rate': [0.001, 0.005, 0.01, 0.05], # Learning rate
        'subsample': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], # Subsampling ratio
        'max_depth': sp_randint(3, 7), # Maximum depth of the individual estimators
        'min_samples_split': sp_randint(2, 11), # Minimum number of samples required to split an internal node
        'min_samples_leaf': sp_randint(1, 5) # Minimum number of samples required to be at a leaf node
    }

    # Initialize RandomizedSearchCV for hyperparameter tuning
    rnd_search = RandomizedSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=1000,  # Number of parameter settings that are sampled
        cv=5,
        scoring=make_scorer(nrmse, greater_is_better=False),
        random_state=42,
        n_jobs=-1
    )

    # Fit RandomizedSearchCV to find the best model
    rnd_search.fit(X, y)

    # Retrieve the best model
    best_model = rnd_search.best_estimator_

    return best_model






def stacked_model_hyperparameters(X, y, alpha):
    """
    Performs hyperparameter tuning for a stacked regression model using RandomizedSearchCV.

    Args:
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        y (numpy.ndarray): Target vector of shape (n_samples,).
        alpha (float): Regularization strength for the Ridge meta-model.

    Returns:
        sklearn.ensemble.StackingRegressor: The best stacked model found by RandomizedSearchCV.
    """


    # Define base models
    base_models = [
        ('elastic_net', make_pipeline(StandardScaler(), ElasticNet(max_iter=1000000,  random_state=42))),
        ('gradient_boosting', GradientBoostingRegressor(random_state=42))
    ]


    ## Define the meta-model
    meta_model = Ridge(alpha=alpha)

    # Create the stacking ensemble
    stacked_model = StackingRegressor(estimators=base_models, final_estimator=meta_model, n_jobs=-1)

    # Define the parameter grid to search
    param_distributions = {
        'elastic_net__elasticnet__alpha': loguniform(0.001, 200),
        'elastic_net__elasticnet__l1_ratio': uniform(0.1, 0.9),
        'gradient_boosting__n_estimators': sp_randint(500, 700),
        'gradient_boosting__learning_rate': [0.001, 0.005, 0.01, 0.05],
        'gradient_boosting__subsample': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'gradient_boosting__max_depth': sp_randint(3, 7),
        'gradient_boosting__min_samples_split': sp_randint(2, 11),
        'gradient_boosting__min_samples_leaf': sp_randint(1, 5)

    }


    # Initialize RandomizedSearchCV for hyperparameter tuning
    random_search = RandomizedSearchCV(stacked_model, param_distributions, n_iter=100, cv=5,
                                       scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
    random_search.fit(X, y)


    # Retrieve the best model
    best_model = random_search.best_estimator_

    return best_model


