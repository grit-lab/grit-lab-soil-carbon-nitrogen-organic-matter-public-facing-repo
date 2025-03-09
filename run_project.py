"""
File: run_project.py
Author: Nayma Nur
Description: Main script for running the entire model pipeline, from data loading and preprocessing,
             through model fitting and evaluation, to outputting results. This script coordinates
             the workflow and user interaction.
"""


from utils import read_features_targets
from bootstrap import bootstrap_EN, bootstrap_GBT, bootstrap_stacked_model
from model_outputs import model_outputs_EN_GBT, model_outputs_SM
from model import (elastic_net_model_hyperparameters, gradient_boosting_tree_model_hyperparameters,
                   stacked_model_hyperparameters)
import os


# Define file paths for the feature data.
features_file = f'./inputs/features/input_features.csv'


# Prompt user to select a parameter to predict and ensure the input is handled correctly.
choice = input("Which parameter do you want to predict? Enter 'som', 'c', or 'n': ").strip().lower()
labels_file = f'./inputs/labels/{choice}.csv'



# Validate user choice to ensure it's one of the acceptable values.
if choice not in ['som', 'c', 'n']:
    raise ValueError("Invalid choice entered. Please restart and enter a valid choice (som, c, or n).")


# Define alpha based on choice. Alpha is a hybrid model hyperparameter that adjusts regularization.
alpha = 1 if choice in ['som', 'c'] else 0.05

# Convert the user's choice to uppercase to use in folder naming.
folder_name = choice.upper()

# Prompt user to enter the number of bootstrap iterations and convert it to an integer.
num_iterations = int(input("Enter the number of bootstrap iterations you want to perform: "))

# Prompt user to enter the proportion of the dataset to use as the test set and convert to a float.
TEST_SIZE = float(input("Enter the test data size (e.g., 0.2 for 80% training and 20% test): "))

# Convert the test size to a percentage for use in folder naming.
split_percentage = int(TEST_SIZE * 100)


# Load the data from CSV files into numpy arrays.
X, y = read_features_targets(features_file,labels_file)


# Prompt user to choose the model for which to perform analysis.
model_choice = input("Enter the number:\n"
                     "1 for Elastic Net\n"
                     "2 for Gradient Boosting Tree\n"
                     "3 for Stacked Model\n"
                     "Choice: ")


# Process based on user's model choice.
if model_choice in ['1', '2', '3']:
    if model_choice == '1':
        # Tune hyperparameters and perform bootstrap for Elastic Net.
        model_params = elastic_net_model_hyperparameters(X, y)
        print("Elastic Net hyperparameter tuning completed.")
        model_name = 'elastic_net'
        nrmse_scores, r2_scores, mean_feature_importances, closest_predicted, closest_measured = (
            bootstrap_EN(model_params, X, y, num_iterations, TEST_SIZE))

    elif model_choice == '2':
        # Tune hyperparameters and perform bootstrap for Gradient Boosting Tree.
        model_params = gradient_boosting_tree_model_hyperparameters(X, y)
        print("Gradient Boosting Tree hyperparameter tuning completed.")
        model_name = 'gradient_boosting_tree'
        nrmse_scores, r2_scores, mean_feature_importances, closest_predicted, closest_measured = (
            bootstrap_GBT(model_params, X, y, num_iterations, TEST_SIZE))

    elif model_choice == '3':
        # Tune hyperparameters and perform bootstrap for Stacked Model.
        model_params = stacked_model_hyperparameters(X, y, alpha)
        print("Stacked model hyperparameter tuning completed.")
        model_name = 'stacked_model'
        nrmse_scores, r2_scores, mean_feature_importances, closest_predicted, closest_measured, closest_predicted_max,closest_measured_max, top_five_predicted, top_five_measured, top_five_indices\
            = (bootstrap_stacked_model(model_params, X, y, num_iterations, TEST_SIZE))

    print("Bootstrap analysis completed.")

    # Construct output folder paths based on the choices and parameters
    output_folder_name = f"{num_iterations}_iterations/{100-split_percentage}_train_{split_percentage}_test"
    output_base = f'./outputs/{model_name}/{output_folder_name}/{folder_name}'

    output = output_base + '/'
    img_output = output_base + '/image_output'
    model_output = output_base + '/bootstrap_output/'

    # Ensure output directories exist, create if necessary.
    os.makedirs(output, exist_ok=True)
    os.makedirs(img_output, exist_ok=True)
    os.makedirs(model_output, exist_ok=True)

    # Call the appropriate function to generate outputs such as plots and save results.
    if model_choice in ['1', '2']:
        model_outputs_EN_GBT(nrmse_scores, mean_feature_importances, r2_scores, closest_predicted, closest_measured,
                             model_output, img_output, features_file)
    elif model_choice == '3':
        model_outputs_SM(nrmse_scores, r2_scores, closest_predicted, closest_measured, closest_predicted_max,closest_measured_max,
                         top_five_predicted, top_five_measured, top_five_indices,  model_output, img_output)

else:
    # Handle invalid model choice input
    print("Invalid choice. Please enter '1', '2', or '3'")

