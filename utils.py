"""
File: utils.py
Author: Nayma Nur
Description: Utility functions that support various tasks throughout the project such as data
             manipulation, file operations, and other helper functions that are reused across
             different modules.
"""


import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Function to load features and target labels from CSV files
def read_features_targets(features_file,labels_file):
    """
    Loads features and target labels from CSV files into NumPy arrays, excluding the header row.

    Args:
        features_file (str): Path to the CSV file containing feature data.
        labels_file (str): Path to the CSV file containing label data.

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Feature matrix (X) of shape (n_samples, n_features).
            - numpy.ndarray: Target vector (y) of shape (n_samples,).

    Notes:
        - Assumes that the first row of each file is a header row and should be excluded from the data.
        - The files should be formatted with data entries separated by commas.
    """


    # Load the data from CSV files into numpy arrays
    X = np.genfromtxt(features_file, delimiter=',')
    y = np.genfromtxt(labels_file, delimiter=',')

    # Remove the first row from both X and y
    X = X[1:, :]  # Drop the first row and keep all columns
    y = y[1:]  # Drop the first row
    return X,y



# Function to calculate Normalized Root Mean Square Error
def nrmse(y_true, y_pred):
    """
    Computes the Normalized Root Mean Square Error (NRMSE).

    Args:
        y_true (numpy.ndarray): Array of true values.
        y_pred (numpy.ndarray): Array of predicted values.

    Returns:
        float: The normalized RMSE value.

    Notes:
        - NRMSE is calculated as RMSE divided by the range of true values (max - min).
        - A lower NRMSE indicates better model performance.
    """

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    range_y = y_true.max() - y_true.min()
    return rmse / range_y

