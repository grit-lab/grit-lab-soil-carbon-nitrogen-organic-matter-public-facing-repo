"""
File: plots.py
Author: Nayma Nur
Description: Contains functions to create and save various visualizations related to model performance,
             feature importances, and diagnostic plots to help with analysis and presentation of results.
"""


from matplotlib import colors
from sklearn.metrics import r2_score
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd



def plot_nrmse_histogram(nrmse_scores, img_output, mu_nrmse, median_nrmse, min_nrmse, sigma_nrmse ):
    """
    Plots the distribution of Normalized Root Mean Square Error (NRMSE) scores.

    Args:
        nrmse_scores (list or numpy.ndarray): The NRMSE scores obtained from bootstrap resampling.
        img_output (str): Directory path to save the image file.
        mu_nrmse (float): Mean of NRMSE scores.
        median_nrmse (float): Median of NRMSE scores.
        min_nrmse (float): Minimum of NRMSE scores.
        sigma_nrmse (float): Standard deviation of NRMSE scores.
    """


    fig = plt.figure(tight_layout=True)

    # Normalize the value within the appropriate range
    nrmse_scores_array = np.array(nrmse_scores)



    # Create histogram
    N, bins, patches = plt.hist(nrmse_scores_array, bins=100, edgecolor='k', alpha=0.75, density=True)

    # Color by height
    fracs = N / N.max()
    norm = colors.Normalize(fracs.min(), fracs.max())

    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.nipy_spectral(norm(thisfrac))
        thispatch.set_facecolor(color)



    sns.kdeplot(nrmse_scores, color='red', lw=2)  # Add KDE plot
    plt.xlabel('Normalized root mean square error (NRMSE)', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    # lower_limit = max(0, min_nrmse - 0.01)
    plt.xlim(0, 0.6)

    # Define x_max for placing text annotations
    x_max = 0.6


    # Adding text annotations for mean, median, and minimum
    plt.text(x_max * 0.97, plt.ylim()[1] * 0.95, f'Mean NRMSE: {mu_nrmse:.3f}', ha='right', va='top', fontsize=14)
    plt.text(x_max * 0.97, plt.ylim()[1] * 0.90, f'Median NRMSE: {median_nrmse:.3f}', ha='right', va='top', fontsize=14)
    plt.text(x_max * 0.97, plt.ylim()[1] * 0.85, f'Minimum NRMSE: {min_nrmse:.3f}', ha='right', va='top', fontsize=14)
    plt.text(x_max * 0.97, plt.ylim()[1] * 0.80, f'Std Dev NRMSE: {sigma_nrmse:.3f}', ha='right', va='top', fontsize=14)

    plt.tick_params(axis='both', which='major', labelsize=13)  # Set the fontsize of the tick labels

    # Check if output directory exists, if not create it
    if not os.path.exists(img_output):
        os.makedirs(img_output)
    fig.savefig(img_output + '/nrmse_histogram.png', dpi=400, bbox_inches='tight', pad_inches=0.1)


def plot_r2_histogram(r2_scores, img_output, mu_r2, median_r2, max_r2, sigma_r2):
    """
    Plots the distribution of R² scores.

    Args:
        r2_scores (list or numpy.ndarray): The R² scores obtained from model evaluation.
        img_output (str): Directory path to save the image file.
        mu_r2 (float): Mean of R² scores.
        median_r2 (float): Median of R² scores.
        max_r2 (float): Maximum of R² scores.
        sigma_r2 (float): Standard deviation of R² scores.
    """


    fig = plt.figure(tight_layout=True)

    # Normalize the value within the appropriate range
    r2_scores_array = np.array(r2_scores)

    # Create histogram
    N, bins, patches = plt.hist(r2_scores_array, bins=100, edgecolor='k', alpha=0.75, density=True)

    # Color by height
    fracs = N / N.max()
    norm = colors.Normalize(fracs.min(), fracs.max())

    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.Spectral(norm(thisfrac))
        thispatch.set_facecolor(color)

    sns.kdeplot(r2_scores, color='blue', lw=2)  # Add KDE plot
    plt.xlabel('R² Score', fontsize=16)
    plt.ylabel('Frequency', fontsize=16)
    plt.xlim(0, 1)  # Set x-axis to focus on relevant range

    # Adding text annotations for mean, median, and minimum in the top left
    plt.text(0.015, plt.ylim()[1] * 0.95, f'Mean R²: {mu_r2:.3f}', ha='left', va='top', fontsize=14)
    plt.text(0.015, plt.ylim()[1] * 0.90, f'Median R²: {median_r2:.3f}', ha='left', va='top', fontsize=14)
    plt.text(0.015, plt.ylim()[1] * 0.85, f'Maximum R²: {max_r2:.3f}', ha='left', va='top', fontsize=14)
    plt.text(0.015, plt.ylim()[1] * 0.80, f'Std Dev R²: {sigma_r2:.3f}', ha='left', va='top', fontsize=14)

    plt.tick_params(axis='both', which='major', labelsize=13)  # Set the fontsize of the tick labels

    # Check if output directory exists, if not create it
    if not os.path.exists(img_output):
        os.makedirs(img_output)
    fig.savefig(img_output + '/r2_histogram.png', dpi=400, bbox_inches='tight', pad_inches=0.1)




def plot_estimated_vs_measured(measured, estimated, img_output):
    """
    Plots the top 15 most important features as a horizontal bar chart.

    Args:
        mean_feature_importances (numpy.ndarray): Array containing the mean importance of each feature.
        img_output (str): Directory path where the output image will be saved.
        feature_file (str): Path to a CSV file containing feature names.
    """


    fig, ax = plt.subplots(tight_layout=True)

    # Define colors
    scatter_color = 'red'  # Color for scatter plot points
    line_color = 'blue'  # Color for the reference line


    plt.scatter(measured, estimated, color=scatter_color, alpha=0.7)



    # Reference line with defined color
    plt.plot([min(measured), max(measured)], [min(measured), max(measured)], '--', color=line_color)

    # Calculate R^2 value
    r2_value = r2_score(measured, estimated)

    # Text annotation for R^2 value
    plt.text(0.05, 0.95, f'R² = {r2_value:.3f}', ha='left', va='top', fontsize=14, transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.5))
    plt.xlabel('Measured Values', fontsize=16)
    plt.ylabel('Estimated Values', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=13)  # Set the fontsize of the tick labels
    fig.savefig(img_output + '/estimated_vs_measured_test.png', dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.close()




def plot_top_five_iterations_with_errorbar(measured_values, predicted_values, top_five_indices, img_output):


    fig, ax = plt.subplots(tight_layout=True)


    colors = ['y', 'm', 'c', 'b', 'g']
    rank_labels = ["5th Best", "4th Best", "3rd Best", "2nd Best", "1st Best"]

    combined_measured = []
    combined_predicted = []


    # Calculate the mean and standard deviation for error bars
    for i in range(len(measured_values)):
        y_measured = measured_values[i]
        y_predicted = predicted_values[i]

        combined_measured.extend(y_measured)
        combined_predicted.extend(y_predicted)

        # Calculate standard deviation of predicted values
        y_error = np.abs(y_measured - y_predicted)

        # Scatter plot with error bars
        plt.errorbar(y_measured, y_predicted, yerr=y_error, fmt='o', ecolor=colors[i % len(colors)], color=colors[i % len(colors)], label=rank_labels[i])


    combined_measured = np.array(combined_measured)
    combined_predicted = np.array(combined_predicted)

    # Calculate combined R^2 score
    combined_r2 = r2_score(combined_measured, combined_predicted)

    # Text annotation for R^2 value
    plt.text(0.05, 0.95, f'Combined $R^2$ = {combined_r2:.3f}', ha='left', va='top', fontsize=14, transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.5))




    plt.xlabel('Measured Values', fontsize=16)
    plt.ylabel('Estimated Values', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=13)  # Set the fontsize of the tick labels
    legend = plt.legend(title='Top 5 Iterations', fontsize=12, loc='lower right')
    legend.get_title().set_fontsize(12)  # Set the font size of the legend title

    fig.savefig(img_output + '/estimated_vs_measured_best_5_iter_with_error.png', dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.close()




def plot_top_15_feature_importances(mean_feature_importances, img_output, feature_file):

    """
    Plot a horizontal bar chart of the top 15 feature importances.

    Parameters:
    - mean_feature_importances (array-like): Array containing the mean importance of each feature.
    - img_output (str): Directory path where the output image will be saved.
    - feature_file (str): Path to a CSV file containing feature names.
    """


    # Read the feature names from the CSV file
    feature_names = pd.read_csv(feature_file, nrows=0).columns.tolist()

    fig, ax = plt.subplots(figsize=(12, 8))  # Adjusted figure size

    # Convert feature importances to percentages
    feature_importances_percentage = 100 * np.abs(mean_feature_importances) / np.sum(np.abs(mean_feature_importances))

    # Get indices of the top 15 features
    sorted_idx = np.argsort(feature_importances_percentage)[::-1]
    top_15_idx = sorted_idx[:15]

    # Define a color map
    cmap = cm.get_cmap('tab20', len(top_15_idx))  # Get a colormap with enough colors

    # Create the bar plot
    bars = ax.barh(range(15), feature_importances_percentage[top_15_idx], align='center',
                   color=[cmap(i) for i in range(len(top_15_idx))])  # Apply colors from colormap
    ax.set_yticks(range(15))
    ax.set_yticklabels([feature_names[i] for i in top_15_idx])  # Use feature names for y-tick labels
    ax.set_xlabel('Importance (%)', fontsize=28, labelpad=20)
    ax.set_ylabel('Features', fontsize=28, labelpad=10)
    ax.invert_yaxis()  # Invert y-axis to have the highest importance at the top

    # Annotate the bar plot with the percentage values
    max_width = max([bar.get_width() for bar in bars])
    ax.set_xlim(0, max_width + 4)  # Adjust xlim to make room for text, 4% more space

    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.2f}%', va='center',
                fontsize=22)

    # Setting tick parameters separately for each axis
    ax.tick_params(axis='x', which='major', labelsize=22)  # Set the fontsize of the x-tick labels
    ax.tick_params(axis='y', which='major', labelsize=20)  # Set the fontsize of the y-tick labels

    plt.tight_layout()  # Adjust layout to make room for tick labels
    fig.savefig(img_output + '/top_15_features.png', dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.close()




