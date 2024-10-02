import os

import pandas as pd
import plotly.express as px
from scipy.stats import ttest_ind

from src.utils.hypothesis_test import hypothesis_test


def grouped_bar_chart(
    df,
    x,
    x_label,
    y,
    y_label,
    color,
    color_label,
    facet_col,
    facet_col_label,
    category_order,
    title,
    output_dir,
    output_name,
    facet_row=None,
    facet_row_label=None,
):
    """Create a grouped bar chart with improved text on bars."""

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define labels for the plot
    labels = {x: x_label, y: y_label, color: color_label, facet_col: facet_col_label}

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
        text=y,  # Display the y values as text on the bars
    )

    # Ensure the numbers on the axis display correctly
    fig.update_layout(
        yaxis_tickformat=".3f",  # Format axis to show up to three decimal places
        uniformtext_minsize=8,  # Minimum size for text on bars
        uniformtext_mode="hide",  # Hide text if it gets too small
        width=1000,  # Increase the width of the chart to provide more space
        height=600,
        title_x=0.5,
    )

    # Adjust the text position and styling on the bars
    fig.update_traces(
        texttemplate="%{text:.3f}",  # Format text to three decimal places
        textposition="outside",  # Text outside bars (adjust to 'inside' if preferred)
        textfont=dict(
            color="black",  # Change all text to a consistent black color (adjust as needed)
            size=12,  # Make the text size consistent (adjust size as needed)
        ),
    )

    # Save the plot as an image
    fig.write_image(os.path.join(output_dir, output_name))


def get_age_bins(df):
    """Adjust the age bins and labels to only include those present in the dataframe."""
    # Get the minimum and maximum age values from the data
    bins = [0, 3, 18, 42, 67, 96]
    labels = ["0-2", "3-17", "18-41", "42-66", "67-96"]

    adjusted_bins = bins.copy()
    adjusted_labels = labels.copy()

    min_age = df["age"].min()
    max_age = df["age"].max()

    while adjusted_bins[1] < min_age and len(adjusted_bins) > 1:
        adjusted_bins.pop(0)
        adjusted_labels.pop(0)

    while adjusted_bins[-2] > max_age and len(adjusted_bins) > 1:
        adjusted_bins.pop(-1)
        adjusted_labels.pop(-1)

    return adjusted_bins, adjusted_labels


def apply_function_to_groups(grouped_df, classifier, groups, metric, columns, func):
    """Calculate the significance between the GT and prediction/reconstruction values."""
    results = []

    if metric == "score":
        appendix = "_score"
    else:
        appendix = ""

    for group_keys, group in grouped_df:

        for col1, col1_name, col2, col2_name in columns:
            gt = group[f"{classifier['name']}_gt{appendix}"]
            column1 = group[f"{classifier['name']}_{col1}{appendix}"]
            column2 = group[f"{classifier['name']}_{col2}{appendix}"]

            # Calculate significance between the two columns
            result = func(gt, column1, column2)

            # Append the results
            if isinstance(group_keys, tuple):
                group_info = {col: val for col, val in zip(groups, group_keys)}
            else:
                group_info = {groups[0]: group_keys}

            results.append(
                {
                    **group_info,  # Add the group info (sex, age_bin, etc.)
                    "metric": f"{col1_name} - {col2_name}",
                    "value": result,
                }
            )

    metrics_df = pd.DataFrame(results)

    return metrics_df


def aggregate_predictions(grouped_df, classifier, groups, metric, age_bins, age_labels):

    results = []

    # Assume `grouping_cols` is a list like ['sex', 'age_bin']
    for group_keys, group in grouped_df:
        # Copy the group
        group_copy = group.copy()

        # Select only 'gt', 'pred', 'recon' columns based on metric type
        if metric == "score":
            appendix = "_score"
        else:
            appendix = ""

        # Dynamically build relevant column names based on classifier name and appendix
        relevant_columns = [
            col
            for col in group_copy.columns
            if col
            in [
                f"{classifier['name']}_gt{appendix}",
                f"{classifier['name']}_pred{appendix}",
                f"{classifier['name']}_recon{appendix}",
            ]
        ]

        # Filter the group to include only relevant columns
        group_relevant = group_copy[relevant_columns]

        # Rename columns to remove classifier name and appendix
        group_relevant.columns = group_relevant.columns.str.replace(
            f"{classifier['name']}_", ""
        ).str.replace(appendix, "")

        # Change gt, pred and recon
        group_relevant = group_relevant.rename(
            columns={
                f"gt": "Ground Truth (GT)",
                f"pred": "Classifier on GT",
                f"recon": "Classifier on Reconstruction",
            }
        )

        # Get number of samples
        n_samples = len(group_relevant)

        # Apply the custom function (mean in this case)
        metrics = group_relevant.mean()

        # Prepare a dictionary to hold grouping information
        # If you are grouping by multiple columns, group_keys will be a tuple of values
        if isinstance(group_keys, tuple):
            group_info = {col: val for col, val in zip(groups, group_keys)}
        else:
            group_info = {groups[0]: group_keys}

        # Append metrics to results
        for metric_name, value in metrics.items():
            results.append(
                {
                    **group_info,  # Add the group info (sex, age_bin, etc.)
                    "metric": metric_name,
                    "value": value,
                    "n_samples": n_samples,
                }
            )

    # Create the final DataFrame
    metrics_df = pd.DataFrame(results)

    return metrics_df


def classifier_evaluation(df, classifiers, output_dir):
    """Evaluate the classifier predictions and generate visualizations."""
    age_bins, age_labels = get_age_bins(df)

    # Categorize 'age' into bins
    df["age_bin"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=True)

    # Group by 'sex' and 'age_bin'
    grouped_df = df.groupby(["sex", "age_bin"], observed=False)

    for classifier in classifiers:
        classifier_name = classifier["name"]
        # Classifier predictions with significance
        metrics = aggregate_predictions(
            grouped_df,
            classifier,
            ["sex", "age_bin"],
            "prediction",
            age_bins,
            age_labels,
        )
        grouped_bar_chart(
            metrics,
            "sex",
            "Sex",
            "value",
            "Classifier True Predictions",
            "metric",
            "Legend",
            "age_bin",
            "Age Group",
            {},
            f"{classifier_name} Predictions",
            output_dir,
            f"{classifier_name}_predictions.png",
        )

        f = lambda gt, y, x: hypothesis_test(y, x)
        significance_results = apply_function_to_groups(
            grouped_df,
            classifier,
            ["sex", "age_bin"],
            "prediction",
            [
                ("gt", "GT", "pred", "Classifier on GT"),
                ("pred", "Classifier on GT", "recon", "Classifier on Reconstruction"),
                ("gt", "GT", "recon", "Classifier on Reconstruction"),
            ],
            f,
        )
        grouped_bar_chart(
            significance_results,
            "sex",
            "Sex",
            "value",
            "Classifier Predictions Significance",
            "metric",
            "Legend",
            "age_bin",
            "Age Group",
            {},
            f"{classifier_name} Predictions Significance",
            output_dir,
            f"{classifier_name}_predictions_significance.png",
        )

        # Classifier score with significance
        metrics = aggregate_predictions(
            grouped_df, classifier, ["sex", "age_bin"], "score", age_bins, age_labels
        )
        grouped_bar_chart(
            metrics,
            "sex",
            "Sex",
            "value",
            "Classifier Score Predictions",
            "metric",
            "Legend",
            "age_bin",
            "Age Group",
            {},
            f"{classifier_name} Score",
            output_dir,
            f"{classifier_name}_score.png",
        )

        f = lambda gt, y, x: hypothesis_test(y, x)
        significance_results = apply_function_to_groups(
            grouped_df,
            classifier,
            ["sex", "age_bin"],
            "score",
            [
                ("gt", "GT", "pred", "Classifier on GT"),
                ("pred", "Classifier on GT", "recon", "Classifier on Reconstruction"),
                ("gt", "GT", "recon", "Classifier on Reconstruction"),
            ],
            f,
        )
        grouped_bar_chart(
            significance_results,
            "sex",
            "Sex",
            "value",
            "Classifier Score Significance",
            "metric",
            "Legend",
            "age_bin",
            "Age Group",
            {},
            f"{classifier_name} Score Significance",
            output_dir,
            f"{classifier_name}_score_significance.png",
        )

        # Classifier performance metrics with significance
        f = lambda gt, y, x: classifier["model"].performance_metric(y, x)
        metrics = apply_function_to_groups(
            grouped_df,
            classifier,
            ["sex", "age_bin"],
            classifier["model"].performance_metric_value,
            [
                ("gt", "GT", "pred", "Classifier on GT"),
                ("gt", "GT", "recon", "Classifier on Reconstruction"),
            ],
            f,
        )
        grouped_bar_chart(
            metrics,
            "sex",
            "Sex",
            "value",
            f"{classifier['model'].performance_metric_name}",
            "metric",
            "Legend",
            "age_bin",
            "Age Group",
            {},
            f"{classifier_name} {classifier['model'].performance_metric_name}",
            output_dir,
            f"{classifier_name}_performance.png",
        )

        significance_results = apply_function_to_groups(
            grouped_df,
            classifier,
            ["sex", "age_bin"],
            classifier["model"].performance_metric_value,
            [("pred", "Classifier on GT", "recon", "Classifier on Reconstruction")],
            classifier["model"].significance,
        )
        grouped_bar_chart(
            significance_results,
            "sex",
            "Sex",
            "value",
            f"{classifier['model'].performance_metric_name} Significance",
            "metric",
            "Legend",
            "age_bin",
            "Age Group",
            {},
            f"{classifier_name} {classifier['model'].performance_metric_name} Significance",
            output_dir,
            f"{classifier_name}_performance_significance.png",
        )


def reconstruction_evaluation(df, classifier, output_dir):
    """Evaluate the reconstruction predictions and generate visualizations."""
    # Evaluate the classifier predictions
    pass
