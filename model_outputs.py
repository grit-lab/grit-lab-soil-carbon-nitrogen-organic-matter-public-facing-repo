"""
File: model_outputs.py
Author: Nayma Nur
Description: Provides functionality to output model evaluation results, including saving statistical
             summaries and visualization outputs such as performance plots to specified directories.
"""


from plots import *
from save_outputs import *
import os
import numpy as np


def model_outputs_EN_GBT(nrmse_scores, mean_feature_importances, r2_scores, closest_predicted, closest_measured, model_output,
                  img_output, feature_file ):
    """
    Generates and saves evaluation results for Elastic Net and Gradient Boosting models.

    Args:
        nrmse_scores (list): List of normalized root mean squared error (nRMSE) scores.
        mean_feature_importances (numpy.ndarray): Mean importance values of features.
        r2_scores (list): List of R² scores across bootstrap iterations.
        closest_predicted (numpy.ndarray): Predicted values closest to the mean R² score.
        closest_measured (numpy.ndarray): Corresponding measured values.
        model_output (str): Path to save model output files.
        img_output (str): Path to save image outputs.
        feature_file (str): Path to a CSV file containing feature names.

    Returns:
        int: Returns 0 upon successful execution.
    """

    # Ensure output directories exist, create them if they don't
    if not os.path.exists(model_output):
        os.makedirs(model_output)
    if not os.path.exists(img_output):
        os.makedirs(img_output)

    # Calculate statistics for NMSE scores
    mu_nrmse = np.mean(nrmse_scores)
    sigma_nrmse = np.std(nrmse_scores)
    median_nrmse = np.median(nrmse_scores)
    min_nrmse = np.min(nrmse_scores)

    # Calculate statistics for R² scores
    mu_r2 = np.mean(r2_scores)
    sigma_r2 = np.std(r2_scores)
    median_r2 = np.median(r2_scores)
    max_r2 = np.max(r2_scores)


    # Generate plots
    plot_nrmse_histogram(nrmse_scores, img_output, mu_nrmse, median_nrmse, min_nrmse, sigma_nrmse)
    plot_r2_histogram(r2_scores, img_output, mu_r2, median_r2, max_r2, sigma_r2)
    plot_estimated_vs_measured(closest_measured, closest_predicted, img_output)
    plot_top_15_feature_importances(mean_feature_importances, img_output, feature_file)

    # Write statistical results to a file in the specified model output directory
    write_statistics_to_file(mu_nrmse, median_nrmse, min_nrmse, sigma_nrmse, mu_r2, median_r2, max_r2, sigma_r2,
                             model_output)

    return 0



def model_outputs_SM(nrmse_scores, r2_scores, closest_predicted, closest_measured, closest_predicted_max,closest_measured_max,
                     top_five_predicted, top_five_measured, top_five_indices, model_output, img_output ):
    """
    Generates and saves evaluation results for the stacked model.

    Args:
        nrmse_scores (list): List of normalized root mean squared error (nRMSE) scores.
        r2_scores (list): List of R² scores across bootstrap iterations.
        closest_predicted (numpy.ndarray): Predicted values closest to the mean R² score.
        closest_measured (numpy.ndarray): Corresponding measured values.
        closest_predicted_max (numpy.ndarray): Predicted values closest to the maximum R² score.
        closest_measured_max (numpy.ndarray): Corresponding measured values.
        top_five_predicted (list of numpy.ndarray): Top five predicted values with highest R² scores.
        top_five_measured (list of numpy.ndarray): Top five measured values with highest R² scores.
        top_five_indices (list): Indices of the top five R² iterations.
        model_output (str): Path to save model output files.
        img_output (str): Path to save image outputs.

    Returns:
        int: Returns 0 upon successful execution.
    """

    # Ensure output directories exist, create them if they don't
    if not os.path.exists(model_output):
        os.makedirs(model_output)
    if not os.path.exists(img_output):
        os.makedirs(img_output)

    # Calculate statistics for NMSE scores
    mu_nrmse = np.mean(nrmse_scores)
    sigma_nrmse = np.std(nrmse_scores)
    median_nrmse = np.median(nrmse_scores)
    min_nrmse = np.min(nrmse_scores)

    # Calculate statistics for R² scores
    mu_r2 = np.mean(r2_scores)
    sigma_r2 = np.std(r2_scores)
    median_r2 = np.median(r2_scores)
    max_r2 = np.max(r2_scores)


    # Generate plots
    plot_nrmse_histogram(nrmse_scores, img_output, mu_nrmse, median_nrmse, min_nrmse, sigma_nrmse)
    plot_r2_histogram(r2_scores, img_output, mu_r2, median_r2, max_r2, sigma_r2)
    plot_estimated_vs_measured(closest_measured, closest_predicted, img_output)
    plot_top_five_iterations_with_errorbar(top_five_measured, top_five_predicted, top_five_indices, img_output)

    # Write statistical results to a file in the specified model output directory
    write_statistics_to_file(mu_nrmse, median_nrmse, min_nrmse, sigma_nrmse, mu_r2, median_r2, max_r2, sigma_r2,
                             model_output)

    return 0

