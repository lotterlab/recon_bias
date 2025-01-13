import os

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from PIL import Image, ImageDraw, ImageFont
from plotly.subplots import make_subplots
from scipy.stats import ttest_ind

from src.utils.hypothesis_test import hypothesis_test


def grouped_bar_chart(
    df,
    overall_df,
    x,
    x_label,
    y,
    y_label,
    color,
    color_label,
    category_order,
    title,
    output_dir,
    output_name,
    facet_col=None,
    facet_col_label=None,
    facet_row=None,
    facet_row_label=None,
    error_y=None,
):
    """Create a grouped bar chart with an additional bar plot representing overall performance."""
    # Split the title at "grouped by" and create new titles
    if "grouped by" in title:
        title_parts = title.split("grouped by")
        base_title = title_parts[0].strip()  # Get the first part of the title
    else:
        base_title = title  # Fallback if "grouped by" isn't found

    overall_title = f"{base_title} Overall"

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define labels for the plot
    labels = {x: x_label, y: y_label, color: color_label, facet_col: facet_col_label}

    # Add facet labels only if they are valid (not None)
    if facet_row is not None and facet_row_label is not None:
        labels[facet_row] = facet_row_label

    if facet_col is not None and facet_col_label is not None:
        labels[facet_col] = facet_col_label

    # Dynamically prepare the kwargs for px.bar
    bar_chart_kwargs = {
        "x": x,
        "y": y,
        "color": color,
        "barmode": "group",
        "category_orders": category_order,
        "labels": labels,
        "text": y,
        "title": title,
    }

    # Add facet_col and facet_row only if they are valid (not None)
    if facet_row is not None:
        bar_chart_kwargs["facet_row"] = facet_row

    if facet_col is not None:
        bar_chart_kwargs["facet_col"] = facet_col

    if error_y is not None:
        bar_chart_kwargs["error_y"] = error_y

    # Create the detailed grouped bar chart using Plotly
    fig_bar = px.bar(df, **bar_chart_kwargs)

    # Ensure the numbers on the axis display correctly
    fig_bar.update_layout(
        yaxis_tickformat=".3f",
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        title_x=0.5,
        showlegend=False,  # Keep legend for the detailed performance chart
    )

    # Adjust the text position and styling on the bars
    fig_bar.update_traces(
        texttemplate="%{text:.3f}",  # Format text to three decimal places
        textposition="outside",  # Text outside bars (adjust to 'inside' if preferred)
        textfont=dict(
            color="black",  # Change all text to a consistent black color (adjust as needed)
            size=12,  # Make the text size consistent (adjust size as needed)
        ),
    )

    # Save the detailed grouped bar chart as an image
    bar_chart_path = os.path.join(output_dir, "temp_bar_chart.png")
    fig_bar.write_image(bar_chart_path)

    bar_chart_kwargs = {
        "x": "overall",
        "y": "value",
        "color": "metric",
        "barmode": "group",
        "labels": {
            x: "",
            y: "",
            color: color_label,
        },
        "text": y,
        "title": overall_title,
    }

    if error_y is not None:
        bar_chart_kwargs["error_y"] = error_y

    # Create the overall performance bar chart using Plotly
    fig_overall = px.bar(
        overall_df,
        **bar_chart_kwargs,
    )

    fig_overall.update_layout(
        yaxis_tickformat=".3f",
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        title_x=0.5,
        xaxis_title=None,  # Explicitly set 'Overall' as the x-axis title
        showlegend=True,  # Show legend for the overall plot
    )

    fig_overall.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
        textfont=dict(
            color="black",
            size=12,
        ),
    )

    # Save the overall performance bar chart as an image
    overall_chart_path = os.path.join(output_dir, "temp_overall_chart.png")
    fig_overall.write_image(overall_chart_path)

    # Combine both images using PIL
    bar_chart_img = Image.open(bar_chart_path)
    overall_chart_img = Image.open(overall_chart_path)

    # Determine the final image size, including space for the title
    total_width = bar_chart_img.width + overall_chart_img.width
    max_height = max(bar_chart_img.height, overall_chart_img.height)
    title_height = 0  # Adjust this value based on the font size

    # Create a new image with a white background
    combined_img = Image.new("RGB", (total_width, max_height + title_height), "white")

    # Paste both charts below the title
    combined_img.paste(bar_chart_img, (0, title_height))
    combined_img.paste(overall_chart_img, (bar_chart_img.width, title_height))

    # Save the final combined image
    output_path = os.path.join(output_dir, output_name)
    combined_img.save(output_path)

    # Clean up temporary files
    if os.path.exists(bar_chart_path):
        os.remove(bar_chart_path)
    if os.path.exists(overall_chart_path):
        os.remove(overall_chart_path)


def get_age_bins(df, age_bins):
    """Adjust the age bins and labels to only include those present in the dataframe."""
    # Get the minimum and maximum age values from the data
    age_labels = [f"{age_bins[i]}-{age_bins[i+1]-1}" for i in range(len(age_bins) - 1)]

    adjusted_bins = age_bins.copy()
    adjusted_labels = age_labels.copy()

    min_age = df["age"].min()
    max_age = df["age"].max()

    while adjusted_bins[1] < min_age and len(adjusted_bins) > 1:
        adjusted_bins.pop(0)
        adjusted_labels.pop(0)

    while adjusted_bins[-2] > max_age and len(adjusted_bins) > 1:
        adjusted_bins.pop(-1)
        adjusted_labels.pop(-1)

    return adjusted_bins, adjusted_labels


def apply_function_to_column_pairs(grouped_df, model, groups, metric, columns, func):
    """Calculate the significance between the GT and prediction/reconstruction values."""
    results = []
    overall = []

    if metric == "score":
        appendix = "_score"
    else:
        appendix = ""

    for group_keys, group in grouped_df:

        for col1, col1_name, col2, col2_name in columns:
            gt = group[f"{model['name']}_gt{appendix}"]
            column1 = group[f"{model['name']}_{col1}{appendix}"]
            column2 = group[f"{model['name']}_{col2}{appendix}"]

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

    # Calculate overall value
    overall_group = (
        grouped_df.obj
    )  # Access the entire dataframe from the grouped object

    for col1, col1_name, col2, col2_name in columns:
        gt = overall_group[f"{model['name']}_gt{appendix}"]
        column1 = overall_group[f"{model['name']}_{col1}{appendix}"]
        column2 = overall_group[f"{model['name']}_{col2}{appendix}"]

        # Calculate overall significance
        overall_result = func(gt, column1, column2)

        # Append the overall result with a special "overall" label for group
        overall.append(
            {
                "overall": "",  # Mark as overall group
                "metric": f"{col1_name} - {col2_name}",
                "value": overall_result,
            }
        )

    metrics_df = pd.DataFrame(results)
    overall_df = pd.DataFrame(overall)

    return metrics_df, overall_df


def apply_function_to_single_column(
    grouped_df, model, groups, metric, columns, func, error=None
):
    """Calculate the significance between the GT and prediction/reconstruction values."""
    results = []
    overall = []

    if metric == "score":
        appendix = "_score"
    else:
        appendix = ""

    for group_keys, group in grouped_df:

        for col, col_name in columns:
            column = group[f"{model['name']}_{col}{appendix}"]

            # Calculate significance between the two columns
            result = func(column)

            error_result = None
            if error:
                error_result = error(column)

            # Append the results
            if isinstance(group_keys, tuple):
                group_info = {col: val for col, val in zip(groups, group_keys)}
            else:
                group_info = {groups[0]: group_keys}

            dict = {
                **group_info,  # Add the group info (sex, age_bin, etc.)
                "metric": f"{col_name}",
                "value": result,
            }

            if error_result:
                dict["error"] = error_result

            results.append(dict)

    # Calculate overall value
    overall_group = (
        grouped_df.obj
    )  # Access the entire dataframe from the grouped object

    for col, col_name in columns:
        column = overall_group[f"{model['name']}_{col}{appendix}"]

        # Calculate overall significance
        overall_result = func(column)

        error_result = None
        if error:
            error_result = error(column)

        # Append the overall result with a special "overall" label for group
        dict = {
            "overall": "",  # Mark as overall group
            "metric": f"{col_name}",
            "value": overall_result,
        }

        if error_result:
            dict["error"] = error_result

        overall.append(dict)

    metrics_df = pd.DataFrame(results)
    overall_df = pd.DataFrame(overall)

    return metrics_df, overall_df


def classifier_evaluation(df, classifiers, age_bins, output_dir):
    """Evaluate the classifier predictions and generate visualizations."""
    age_bins, age_labels = get_age_bins(df, age_bins)

    # Categorize 'age' into bins
    df["age_bin"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=False)
    base_dir = output_dir

    for classifier in classifiers:
        classifier_name = classifier["name"]
        print(f"Evaluating {classifier_name} predictions...")

        for group, plot_config, group_name in classifier["model"].evaluation_groups:

            print(f"Grouping by {group_name}...")

            output_dir = os.path.join(base_dir, classifier_name, group_name)

            x = plot_config["x"]
            x_label = plot_config["x_label"]
            facet_col = plot_config.get("facet_col")
            facet_col_label = plot_config.get("facet_col_label")

            df_copy = df.copy()
            grouped_df = df_copy.groupby(group, observed=False)

            # Classifier predictions with significance
            metrics, overall = apply_function_to_single_column(
                grouped_df,
                classifier,
                group,
                "prediction",
                [
                    ("gt", "Ground Truth"),
                    ("pred", "Classifier on GT"),
                    ("recon", "Classifier on Reconstruction"),
                ],
                lambda x: x.mean(),
            )
            grouped_bar_chart(
                df=metrics,
                overall_df=overall,
                x=x,
                x_label=x_label,
                y="value",
                y_label="Classifier True Predictions",
                color="metric",
                color_label="Legend",
                category_order={},
                title=f"{classifier_name} predictions grouped by {group_name}",
                output_dir=output_dir,
                output_name=f"{classifier_name}_predictions_{group_name}.png",
                facet_col=facet_col,
                facet_col_label=facet_col_label,
            )

            f = lambda gt, y, x: hypothesis_test(y, x)
            significance_results, overall = apply_function_to_column_pairs(
                grouped_df,
                classifier,
                group,
                "prediction",
                [
                    ("gt", "GT", "pred", "Classifier on GT"),
                    (
                        "pred",
                        "Classifier on GT",
                        "recon",
                        "Classifier on Reconstruction",
                    ),
                    ("gt", "GT", "recon", "Classifier on Reconstruction"),
                ],
                f,
            )
            grouped_bar_chart(
                df=significance_results,
                overall_df=overall,
                x=x,
                x_label=x_label,
                y="value",
                y_label="Classifier Predictions Significance",
                color="metric",
                color_label="Legend",
                category_order={},
                title=f"{classifier_name} predictions significance grouped by {group_name}",
                output_dir=output_dir,
                output_name=f"{classifier_name}_predictions_significance_{group_name}.png",
                facet_col=facet_col,
                facet_col_label=facet_col_label,
            )
            # Classifier score with significance
            metrics, overall = apply_function_to_single_column(
                grouped_df,
                classifier,
                group,
                "score",
                [
                    ("pred", "Classifier on GT"),
                    ("recon", "Classifier on Reconstruction"),
                ],
                lambda x: x.mean(),
                lambda x: x.std(),
            )

            grouped_bar_chart(
                df=metrics,
                overall_df=overall,
                x=x,
                x_label=x_label,
                y="value",
                y_label="Classifier Score Predictions",
                color="metric",
                color_label="Legend",
                category_order={},
                title=f"{classifier_name} score grouped by {group_name}",
                output_dir=output_dir,
                output_name=f"{classifier_name}_score_{group_name}.png",
                facet_col=facet_col,
                facet_col_label=facet_col_label,
                error_y="error",
            )

            f = lambda gt, y, x: hypothesis_test(y, x)
            significance_results, overall = apply_function_to_column_pairs(
                grouped_df,
                classifier,
                group,
                "score",
                [
                    (
                        "pred",
                        "Classifier on GT",
                        "recon",
                        "Classifier on Reconstruction",
                    ),
                ],
                f,
            )

            grouped_bar_chart(
                df=significance_results,
                overall_df=overall,
                x=x,
                x_label=x_label,
                y="value",
                y_label="Classifier Score Significance",
                color="metric",
                color_label="Legend",
                category_order={},
                title=f"{classifier_name} score significance grouped by {group_name}",
                output_dir=output_dir,
                output_name=f"{classifier_name}_score_significance_{group_name}.png",
                facet_col=facet_col,
                facet_col_label=facet_col_label,
            )

            # Classifier performance metrics with significance
            f = lambda gt, y, x: classifier["model"].evaluation_performance_metric(x, y)
            metrics, overall = apply_function_to_column_pairs(
                grouped_df,
                classifier,
                group,
                classifier["model"].performance_metric_input_value,
                [
                    ("gt", "GT", "pred", "Classifier on GT"),
                    ("gt", "GT", "recon", "Classifier on Reconstruction"),
                ],
                f,
            )
            grouped_bar_chart(
                df=metrics,
                overall_df=overall,
                x=x,
                x_label=x_label,
                y="value",
                y_label=f"{classifier['model'].performance_metric_name}",
                color="metric",
                color_label="Legend",
                category_order={},
                title=f"{classifier_name} {classifier['model'].performance_metric_name} grouped by {group_name}",
                output_dir=output_dir,
                output_name=f"{classifier_name}_performance_{group_name}.png",
                facet_col=facet_col,
                facet_col_label=facet_col_label,
            )

            significance_results, overall = apply_function_to_column_pairs(
                grouped_df,
                classifier,
                group,
                classifier["model"].performance_metric_input_value,
                [("pred", "Classifier on GT", "recon", "Classifier on Reconstruction")],
                classifier["model"].significance,
            )
            grouped_bar_chart(
                df=significance_results,
                overall_df=overall,
                x=x,
                x_label=x_label,
                y="value",
                y_label=f"{classifier['model'].performance_metric_name} significance",
                color="metric",
                color_label="Legend",
                category_order={},
                title=f"{classifier_name} {classifier['model'].performance_metric_name} significance grouped by {group_name}",
                output_dir=output_dir,
                output_name=f"{classifier_name}_performance_significance_{group_name}.png",
                facet_col=facet_col,
                facet_col_label=facet_col_label,
            )


def reconstruction_evaluation(df, reconstruction, age_bins, output_dir):
    """Evaluate the reconstruction predictions and generate visualizations."""
    # Evaluate the classifier predictions
    age_bins, age_labels = get_age_bins(df, age_bins)
    output_dir = os.path.join(output_dir, reconstruction["name"])

    # Categorize 'age' into bins
    df["age_bin"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, right=False)
    performance_metrics = [("psnr", "PSNR"), ("ssim", "SSIM"), ("nrmse", "NRSME")]
    print(f"Evaluating reconstruction ...")

    for group, plot_config, group_name in reconstruction["model"].evaluation_groups:

        for performance_metric, performance_metric_label in performance_metrics:
            x = plot_config["x"]
            x_label = plot_config["x_label"]
            facet_col = plot_config.get("facet_col")
            facet_col_label = plot_config.get("facet_col_label")

            df_copy = df.copy()
            grouped_df = df_copy.groupby(group, observed=False)

            # Show performance metrics
            metrics, overall = apply_function_to_single_column(
                grouped_df,
                reconstruction,
                group,
                "prediction",
                [(performance_metric, performance_metric_label)],
                lambda x: x.mean(),
                lambda x: x.std(),
            )
            grouped_bar_chart(
                df=metrics,
                overall_df=overall,
                x=x,
                x_label=x_label,
                y="value",
                y_label=performance_metric_label,
                color="metric",
                color_label="Legend",
                category_order={},
                title=f"{reconstruction['name']} performance grouped by {group_name}",
                output_dir=output_dir,
                output_name=f"{reconstruction['name']}_{performance_metric}_{group_name}.png",
                facet_col=facet_col,
                facet_col_label=facet_col_label,
                error_y="error",
            )
