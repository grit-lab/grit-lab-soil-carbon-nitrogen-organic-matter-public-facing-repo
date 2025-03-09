"""
File: bootstrap.py
Author: Nayma Nur
Description: Contains functions to perform bootstrap analysis on different regression models.
             This includes resampling the dataset, fitting models on these samples, and assessing
             model performance metrics over numerous iterations.
"""



from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
from utils import nrmse




def bootstrap_EN(model, X, y, n_iterations, test_size):
    """
        Performs bootstrap resampling on an Elastic Net model.

        Args:
            model (sklearn estimator): The Elastic Net regression model to be bootstrapped.
            X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
            y (numpy.ndarray): Target vector of shape (n_samples,).
            n_iterations (int): Number of bootstrap iterations.
            test_size (float): Proportion of the dataset to include in the test split.

        Returns:
            tuple: A tuple containing:
                - list: nRMSE scores across bootstrap iterations.
                - list: R² scores across bootstrap iterations.
                - numpy.ndarray: Mean coefficients from Elastic Net.
                - numpy.ndarray: Predicted values corresponding to the iteration closest to mean R².
                - numpy.ndarray: Measured values corresponding to the iteration closest to mean R².
        """


    nrmse_scores = []
    coefficients_list = []
    r2_scores = []
    predicted_test_values = []
    measured_test_values = []


    for _ in range(n_iterations):

        # Resample the dataset
        X_resampled, y_resampled = resample(X, y)

        # Split the resampled data into training and testing sets
        X_train_resampled, X_test_resampled, y_train_resampled, \
            y_test_resampled = train_test_split(X_resampled, y_resampled, test_size=test_size)

        # Fit the model
        model.fit(X_train_resampled, y_train_resampled)
        y_test_pred = model.predict(X_test_resampled)
        nrmse_scores.append(nrmse(y_test_resampled, y_test_pred))
        r2_scores.append(r2_score(y_test_resampled, y_test_pred))
        coefficients_list.append(model.named_steps['elasticnet'].coef_)  # Access the ElasticNet step
        predicted_test_values.append(y_test_pred)
        measured_test_values.append(y_test_resampled)


    # Calculate the mean R2 score
    mean_r2 = np.mean(r2_scores)

    # Find the index of the score closest to the mean
    differences = [abs(score - mean_r2) for score in r2_scores]
    closest_index = np.argmin(differences)

    # Retrieve the closest predicted and measured values
    closest_predicted = predicted_test_values[closest_index]
    closest_measured = measured_test_values[closest_index]

    mean_coefficients = np.mean(coefficients_list, axis=0)

    return nrmse_scores, r2_scores, mean_coefficients, closest_predicted, closest_measured






def bootstrap_GBT(model, X, y, n_iterations, test_size):
    """
    Performs bootstrap resampling on a Gradient Boosting Trees model.

    Args:
        model (sklearn estimator): The Gradient Boosting Trees model to be bootstrapped.
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        y (numpy.ndarray): Target vector of shape (n_samples,).
        n_iterations (int): Number of bootstrap iterations.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: A tuple containing:
            - list: nRMSE scores across bootstrap iterations.
            - list: R² scores across bootstrap iterations.
            - numpy.ndarray: Mean feature importances from the Gradient Boosting model.
            - numpy.ndarray: Predicted values corresponding to the iteration closest to mean R².
            - numpy.ndarray: Measured values corresponding to the iteration closest to mean R².
    """

    nrmse_scores = []
    r2_scores = []
    feature_importances_list = []
    predicted_test_values = []
    measured_test_values = []


    for _ in range(n_iterations):

        # Resample the dataset
        X_resampled, y_resampled = resample(X, y)

        # Split the resampled data into training and testing sets
        X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled \
            = train_test_split(X_resampled, y_resampled, test_size=test_size)

        # Fit the model
        model.fit(X_train_resampled, y_train_resampled)
        y_test_pred = model.predict(X_test_resampled)

        # Calculate and store performance metrics
        nrmse_scores.append(nrmse(y_test_resampled, y_test_pred))
        r2_scores.append(r2_score(y_test_resampled, y_test_pred))
        feature_importances_list.append(model.feature_importances_)
        predicted_test_values.append(y_test_pred)
        measured_test_values.append(y_test_resampled)

    # Calculate the mean R2 score
    mean_r2 = np.mean(r2_scores)

    # Find the index of the score closest to the mean
    differences = [abs(score - mean_r2) for score in r2_scores]
    closest_index = np.argmin(differences)

    # Retrieve the closest predicted and measured values
    closest_predicted = predicted_test_values[closest_index]
    closest_measured = measured_test_values[closest_index]

    mean_feature_importances = np.mean(feature_importances_list, axis=0)

    return nrmse_scores, r2_scores, mean_feature_importances, closest_predicted, closest_measured



def bootstrap_stacked_model(model, X, y, n_iterations, test_size):
    """
    Performs bootstrap resampling on a stacked model (ElasticNet + Gradient Boosting).

    Args:
        model (sklearn estimator): The stacked machine learning model to be bootstrapped.
        X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        y (numpy.ndarray): Target vector of shape (n_samples,).
        n_iterations (int): Number of bootstrap iterations.
        test_size (float): Proportion of the dataset to include in the test split.

    Returns:
        tuple: A tuple containing:
            - list: nRMSE scores across bootstrap iterations.
            - list: R² scores across bootstrap iterations.
            - list: Feature importances from the stacked model.
            - numpy.ndarray: Predicted values corresponding to the iteration closest to mean R².
            - numpy.ndarray: Measured values corresponding to the iteration closest to mean R².
            - numpy.ndarray: Predicted values corresponding to the iteration closest to max R².
            - numpy.ndarray: Measured values corresponding to the iteration closest to max R².
            - list: Top five predicted values with highest R².
            - list: Top five measured values with highest R².
            - list: Indices of the top five iterations.
    """

    nrmse_scores = []
    r2_scores = []
    feature_importances_list = []
    predicted_test_values = []
    measured_test_values = []

    for _ in range(n_iterations):

        # Resample the dataset
        X_resampled, y_resampled = resample(X, y)

        # Split the resampled data into training and testing sets
        X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(
            X_resampled, y_resampled, test_size=test_size)

        # Fit the stacked model
        model.fit(X_train_resampled, y_train_resampled)
        y_test_pred = model.predict(X_test_resampled)

        # Calculate and store performance metrics
        nrmse_scores.append(nrmse(y_test_resampled, y_test_pred))
        r2_scores.append(r2_score(y_test_resampled, y_test_pred))

        predicted_test_values.append(y_test_pred)
        measured_test_values.append(y_test_resampled)

    # Calculate the mean R2 score
    mean_r2 = np.mean(r2_scores)

    # Find the index of the score closest to the mean
    differences = [abs(score - mean_r2) for score in r2_scores]
    closest_index = np.argmin(differences)

    # Retrieve the closest predicted and measured values
    closest_predicted = predicted_test_values[closest_index]
    closest_measured = measured_test_values[closest_index]

    # Identify top five iterations with maximum R2
    top_five_indices = np.argsort(r2_scores)[-5:]


    # Retrieve the predicted and measured values for the top five iterations
    top_five_predicted = [predicted_test_values[i] for i in top_five_indices]
    top_five_measured = [measured_test_values[i] for i in top_five_indices]

    # Calculate the max R2 score
    max_r2 = np.max(r2_scores)

    # Find the index of the score closest to the mean
    differences_max = [abs(score - max_r2) for score in r2_scores]
    closest_index_max = np.argmin(differences_max)

    # Retrieve the closest predicted and measured values
    closest_predicted_max = predicted_test_values[closest_index_max]
    closest_measured_max = measured_test_values[closest_index_max]

    return (nrmse_scores, r2_scores, feature_importances_list, closest_predicted, closest_measured,
            closest_predicted_max,closest_measured_max, top_five_predicted, top_five_measured, top_five_indices )





