import pandas as pd
from typing import List, Optional
from typing import Callable, Dict, List, Optional
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

# Define age bins and labels
AGE_BINS = [0, 3, 18, 42, 67, 96]
AGE_LABELS = ["0-2", "3-17", "18-41", "42-66", "67-96"]

def calculate_percent_diff(df, col1, col2):
    """Calculate the percentage difference between two columns."""
    return 100 * (df[col2] - df[col1]) / df[col1]

def calculate_significance(df, col1, col2):
    """Calculate p-values for significance between two columns using paired t-test."""
    """Calculate p-values for significance between two columns using paired t-test, with check for zero variance."""
    # Calculate differences
    differences = df[col1] - df[col2]
    
    # Check for zero variance (all values in the differences are the same)
    if np.all(differences == 0):
        print("No difference between colums")
        return 1  # or return a custom message like "No difference"

    # If there's variance, proceed with the t-test
    t_stat, p_val = stats.ttest_rel(df[col1], df[col2], nan_policy='omit')
    return p_val

def plot_combined_percent_diff_and_significance(classifier, pivot_results, output_dir, reconstruction=False):
    """Plot both the percent differences and significance for both genders in one plot per classifier."""
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Combine male and female dataframes for comparison
    male_data = pivot_results.get('M')  # Assuming 'M' for male
    female_data = pivot_results.get('F')  # Assuming 'F' for female

    if male_data is not None and female_data is not None:
        # Align the male and female data by their index (age_bin)
        combined_data = pd.concat([male_data.add_suffix('_male'), female_data.add_suffix('_female')], axis=1)

        # Create figure for combined gender comparison
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
        fig.suptitle(f"{classifier} - Male vs Female Comparison", fontsize=16)

        # Plot percent differences (GT vs Pred, GT vs Recon, Pred vs Recon)
        percent_diff_cols = [
            f"{classifier}_gt_vs_pred_diff_male", f"{classifier}_gt_vs_pred_diff_female",
        ]

        if reconstruction:
            percent_diff_cols += [f"{classifier}_gt_vs_recon_diff_male", f"{classifier}_gt_vs_recon_diff_female",
            f"{classifier}_pred_vs_recon_diff_male", f"{classifier}_pred_vs_recon_diff_female"
        ]

        ax1 = axes[0]
        combined_data[percent_diff_cols].plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title(f"Percent Differences - Male vs Female")
        ax1.set_xlabel('Age Bins')
        ax1.set_ylabel('Percentage Difference')

        # Display values on the bars
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%.2f', label_type='edge')

        # Plot significance (p-values)
        signif_cols = [
            f"{classifier}_gt_vs_pred_signif_male", f"{classifier}_gt_vs_pred_signif_female"
        ]

        if reconstruction:
            signif_cols += [f"{classifier}_gt_vs_recon_signif_male", f"{classifier}_gt_vs_recon_signif_female",
            f"{classifier}_pred_vs_recon_signif_male", f"{classifier}_pred_vs_recon_signif_female"
        ]


        ax2 = axes[1]
        combined_data[signif_cols].plot(kind='bar', ax=ax2, width=0.8)
        ax2.set_title(f"Statistical Significance (p-values) - Male vs Female")
        ax2.set_xlabel('Age Bins')
        ax2.set_ylabel('p-values')

        # Display values on the bars
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.4f', label_type='edge')

        # Save the figure
        combined_filename = f"{classifier}_combined_gender_evaluation.png"
        output_path = os.path.join(output_dir, combined_filename)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path)
        plt.close()


def create_evaluation_charts(classifier_name, df, output_dir):
    """Create pivot tables with differences for GT, Pred, and Recon, split by gender."""
    # Bin the ages
    df['age_bin'] = pd.cut(df['age'], bins=AGE_BINS, labels=AGE_LABELS)

    # Separate by gender
    genders = df['sex'].unique()

    # Check if reconstruction columns exist
    recon_col = f"{classifier_name}_recon" if f"{classifier_name}_recon" in df.columns else None

    pivot_results = {}
    
    for gender in genders:
        # Filter data by gender
        gender_df = df[df['sex'] == gender]

        # Create a pivot table with age bins, and ground truth, prediction, and reconstruction (if available)
        pivot_data = gender_df.pivot_table(
            index='age_bin', 
            values=[f"{classifier_name}_gt", f"{classifier_name}_pred"] + ([recon_col] if recon_col else []),
            aggfunc='mean'
        )

        # Calculate percent differences
        pivot_data[f"{classifier_name}_gt_vs_pred_diff"] = calculate_percent_diff(pivot_data, f"{classifier_name}_gt", f"{classifier_name}_pred")
        if recon_col:
            pivot_data[f"{classifier_name}_gt_vs_recon_diff"] = calculate_percent_diff(pivot_data, f"{classifier_name}_gt", recon_col)
            pivot_data[f"{classifier_name}_pred_vs_recon_diff"] = calculate_percent_diff(pivot_data, f"{classifier_name}_pred", recon_col)

        # Calculate statistical significance
        pivot_data[f"{classifier_name}_gt_vs_pred_signif"] = calculate_significance(pivot_data, f"{classifier_name}_gt", f"{classifier_name}_pred")
        if recon_col:
            pivot_data[f"{classifier_name}_gt_vs_recon_signif"] = calculate_significance(pivot_data, f"{classifier_name}_gt", recon_col)
            pivot_data[f"{classifier_name}_pred_vs_recon_signif"] = calculate_significance(pivot_data, f"{classifier_name}_pred", recon_col)

        pivot_results[gender] = pivot_data

    plot_combined_percent_diff_and_significance(classifier_name, pivot_results, output_dir, (recon_col is not None))


def create_predictions_charts(classifier, df, output_dir):
    """Create and save combined prediction charts for both male and female on the same plot."""
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Bin the ages
    df['age_bin'] = pd.cut(df['age'], bins=AGE_BINS, labels=AGE_LABELS)

    # Separate by gender
    male_df = df[df['sex'] == 'M']
    female_df = df[df['sex'] == 'F']

    # Check if reconstruction columns exist
    recon_col = f"{classifier}_recon" if f"{classifier}_recon" in df.columns else None

    # Create pivot tables for both male and female
    male_pivot = male_df.pivot_table(
        index='age_bin', 
        values=[f"{classifier}_gt", f"{classifier}_pred"] + ([recon_col] if recon_col else []),
        aggfunc='mean'
    )

    female_pivot = female_df.pivot_table(
        index='age_bin', 
        values=[f"{classifier}_gt", f"{classifier}_pred"] + ([recon_col] if recon_col else []),
        aggfunc='mean'
    )

    # Concatenate male and female pivot tables for side-by-side plotting
    combined_pivot = pd.concat([male_pivot.add_suffix('_male'), female_pivot.add_suffix('_female')], axis=1)

    # Plot the combined data for both male and female
    ax = combined_pivot.plot(kind='bar', figsize=(12, 8), width=0.8)

    # Add labels and title
    ax.set_title(f"{classifier} - Male vs Female Predictions")
    ax.set_xlabel('Age Bins')
    ax.set_ylabel('Values')

    # Display percentages on the bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', label_type='edge')

    # Save the combined figure
    filename = f"{classifier}_combined_predictions.png"
    output_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def evaluate_predictions(results_df: pd.DataFrame, classifiers: List[str], output_dir: str):
    
    for classifier in classifiers:
        classifier_name = classifier["name"]
        print(f"Evaluating {classifier_name}...")
        create_predictions_charts(classifier_name, results_df, output_dir)
        create_evaluation_charts(classifier_name, results_df, output_dir)
        




