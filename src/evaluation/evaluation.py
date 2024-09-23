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
        






import pandas as pd
from scipy.stats import ttest_ind
import numpy as np

import os
import plotly.express as px

import os
import plotly.express as px

def grouped_bar_chart(df, x, x_label, y, y_label, color, color_label, facet_col, facet_col_label, category_order, title, output_dir, output_name, facet_row=None, facet_row_label=None):
    """Create a grouped bar chart with improved text on bars."""
    
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define labels for the plot
    labels = {
        x: x_label,
        y: y_label,
        color: color_label,
        facet_col: facet_col_label
    }
    
    if facet_row:
        labels[facet_row] = facet_row_label

    # Create the bar chart
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color,
        barmode="group",
        facet_col=facet_col,
        facet_row=facet_row,
        category_orders=category_order,
        labels=labels,
        title=title,
        text=y  # Display the y values as text on the bars
    )

    # Ensure the numbers on the axis display correctly
    fig.update_layout(
        yaxis_tickformat=".3f",  # Format axis to show up to three decimal places
        uniformtext_minsize=8,    # Minimum size for text on bars
        uniformtext_mode='hide',  # Hide text if it gets too small
    )

    # Adjust the text position and styling on the bars
    fig.update_traces(
        texttemplate='%{text:.3f}',    # Format text to three decimal places
        textposition='outside',        # Text outside bars (adjust to 'inside' if preferred)
        textfont=dict(
            color="black",            # Change all text to a consistent black color (adjust as needed)
            size=12,                  # Make the text size consistent (adjust size as needed)
        )
    )

    # Save the plot as an image
    fig.write_image(os.path.join(output_dir, output_name))

    return fig  # Optionally, return the figure if you want to visualize it in the notebook



# Define helper functions for percentage difference and t-test

def percent_difference(a, b):
    """Calculate percent difference between two series."""
    return (abs(a - b) / ((a + b) / 2)) * 100

def perform_t_test(a, b):
    """Perform t-test between two series and return the p-value."""
    if np.allclose(a, b):
        return 1.0
    t_stat, p_value = ttest_ind(a, b, equal_var=False, nan_policy='omit')
    return p_value

def evaluate_classifier_predictions(df, classifier, output_dir):
    df['age_bin'] = pd.cut(df['age'], bins=AGE_BINS, labels=AGE_LABELS, right=True)
    df = df.copy()
    grouped = df.groupby(['sex', 'age_bin'])

    # Store results in a list to build the final dataframe
    results = []

    # Loop through each group (sex, age_bin)
    for (sex, age_bin), group in grouped:
        metrics = ['gt', 'pred', 'recon']
        available_columns = {metric: f"{classifier['name']}_{metric}" for metric in metrics if f"{classifier['name']}_{metric}" in group.columns}

        # Check if recon is available
        gt_col = available_columns.get('gt')
        pred_col = available_columns.get('pred')
        recon_col = available_columns.get('recon', None)

        # Calculate percent differences and t-tests between gt, pred, and recon
        if gt_col and pred_col:
            # GT vs. Pred
            percent_gt_pred = percent_difference(group[gt_col], group[pred_col]).mean()
            significance_gt_pred = perform_t_test(group[gt_col], group[pred_col])
            results.append({'sex': sex, 'age_bin': age_bin, 'type': 'percent', 'metric': 'gt_pred', 'value': percent_gt_pred})
            results.append({'sex': sex, 'age_bin': age_bin, 'type': 'significance', 'metric': 'gt_pred', 'value': significance_gt_pred})

        if pred_col and recon_col:
            # Pred vs. Recon
            percent_pred_recon = percent_difference(group[pred_col], group[recon_col]).mean()
            significance_pred_recon = perform_t_test(group[pred_col], group[recon_col])
            results.append({'sex': sex, 'age_bin': age_bin, 'type': 'percent', 'metric': 'pred_recon', 'value': percent_pred_recon})
            results.append({'sex': sex, 'age_bin': age_bin, 'type': 'significance', 'metric': 'pred_recon', 'value': significance_pred_recon})

        if gt_col and recon_col:
            # GT vs. Recon
            percent_gt_recon = percent_difference(group[gt_col], group[recon_col]).mean()
            significance_gt_recon = perform_t_test(group[gt_col], group[recon_col])
            results.append({'sex': sex, 'age_bin': age_bin, 'type': 'percent', 'metric': 'gt_recon', 'value': percent_gt_recon})
            results.append({'sex': sex, 'age_bin': age_bin, 'type': 'significance', 'metric': 'gt_recon', 'value': significance_gt_recon})

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save or return results as needed, output_dir handling can be done here
    return results_df

def adjust_age_bins(df, age_column, bins, labels):
    """Adjust the age bins and labels to only include those present in the dataframe."""
    # Get the minimum and maximum age values from the data
    
    adjusted_bins = bins.copy()
    adjusted_labels = labels.copy()

    min_age = df[age_column].min()
    max_age = df[age_column].max()

    while adjusted_bins[1] < min_age and len(adjusted_bins) > 1:
        adjusted_bins.pop(0)
        adjusted_labels.pop(0)

    while adjusted_bins[-2] > max_age and len(adjusted_bins) > 1:
        adjusted_bins.pop(-1)
        adjusted_labels.pop(-1)

    return adjusted_bins, adjusted_labels


def aggregate_classifier_predictions(df, classifier, output_dir): 
    # Define your age bins and labels
    AGE_BINS = [0, 20, 40, 60, 80, 100]
    AGE_LABELS = ['0-20', '21-40', '41-60', '61-80', '81-100']
    
    # Create a copy to avoid modifying original DataFrame
    df = df.copy()
    
    # Categorize 'age' into bins
    df['age_bin'] = pd.cut(df['age'], bins=AGE_BINS, labels=AGE_LABELS, right=True)
    
    # Group by 'sex' and 'age_bin'
    grouped = df.groupby(['sex', 'age_bin'])

    results = []

    # Iterate over each group
    for (sex, age_bin), group in grouped:
        # Copy the group
        group_copy = group.copy()
        
        # Select only 'gt', 'pred', 'recon' columns
        relevant_columns = [col for col in group_copy.columns if col.endswith('_gt') 
                            or col.endswith('_pred') 
                            or col.endswith('_recon')]
        group_relevant = group_copy[relevant_columns]
        
        # Get number of samples
        n_samples = len(group_relevant)
        
        # Apply the custom function
        print(f"For sex {sex} and age bin {age_bin}:")
        print(group_relevant)
        metrics = classifier.accumulation_function(group_relevant)
        print(metrics)
        
        # Add sample size to metrics
        metrics['n_samples'] = n_samples
        
        # Append metrics to results
        for metric_name, value in metrics.items():
            results.append({
                'sex': sex,
                'age_bin': age_bin,
                'metric': metric_name,
                'value': value,
                'n_samples': n_samples
            })

    # Create the final DataFrame
    metrics_df = pd.DataFrame(results)

    print("\nMetrics DataFrame:")
    print(metrics_df)