import os 
import pandas as pd
from PIL import Image
import plotly.express as px
from src.utils.hypothesis_test import hypothesis_test

def grouped_bar_chart(
    df,
    overall_df,
    x,
    x_label,
    y,
    y_label,
    std_y,
    color,
    color_label,
    category_order,
    title,
    output_dir,
    output_name,
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
    labels = {x: x_label, y: y_label, color: color_label}

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

    if std_y: 
        bar_chart_kwargs["error_y"] = std_y

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
        "error_y": std_y,
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

def apply_function_to_column_pairs(grouped_df, groups, columns, func):
    """Calculate the significance between the GT and prediction/reconstruction values."""
    results = []
    overall = []

    for group_keys, group in grouped_df:
        for col1, col1_name, col2, col2_name in columns:
            group_info = {groups: group_keys}

            column1 = group[col1]
            column2 = group[col2]

            # Calculate significance between the two columns
            result = func(column1, column2)

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
        column1 = overall_group[col1]
        column2 = overall_group[col2]

        # Calculate overall significance
        overall_result = func(column1, column2)

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
    grouped_df, groups, columns, func, std
):
    """Calculate the significance between the GT and prediction/reconstruction values."""
    results = []
    overall = []

    for group_keys, group in grouped_df:
        group_info = {groups: group_keys}
        for col, col_name in columns:
            column = group[col]

            result = func(column)
            std_result = std(column)

            dict = {
                **group_info,  # Add the group info (sex, age_bin, etc.)
                "metric": f"{col_name}",
                "value": result,
                "std_y"  : std_result
            }

            results.append(dict)

    # Calculate overall value
    overall_group = (
        grouped_df.obj
    )  # Access the entire dataframe from the grouped object

    for col, col_name in columns:
        column = overall_group[col]

        # Calculate overall significance
        overall_result = func(column)
        std_result = std(column)


        # Append the overall result with a special "overall" label for group
        dict = {
            "overall": "",  # Mark as overall group
            "metric": f"{col_name}",
            "value": overall_result,
            "std_y"  : std_result
        }

        overall.append(dict)

    metrics_df = pd.DataFrame(results)
    overall_df = pd.DataFrame(overall)

    return metrics_df, overall_df


def individual_evaluation(df, segmentation_model, reconstruction_models, output_dir):
    base_dir = output_dir
    objectives = ["dice", "sum"]

    # Create a new column for grouping based on WHO CNS Grade
    df["grade_group"] = df["WHO CNS Grade"].apply(lambda x: "<4" if x < 4 else "4")

    for reconstruction_info in reconstruction_models: 
        name = reconstruction_info["name"]
        output_dir = os.path.join(base_dir, name)
        print(f"Evaluating {name} reconstruction model")

        for objective in objectives:

            x = "grade_group"  # Update grouping key
            x_label = "Grade Group (<4 or 4)"  # Update x-axis label
            facet_col = None
            facet_col_label = None

            df_copy = df.copy()
            grouped_df = df_copy.groupby("grade_group", observed=False)  # Update grouping key

            cols = [
                        ("gt_sum", "Ground Truth (GT)"),
                        ("segmentation_sum", "Segmentation on GT"),
                        (f"{name}_segmentation_sum", "Segmentation on Reconstruction"),
                    ] if objective == "sum" else [
                        ("segmentation_dice", "Segmentation on GT"),
                        (f"{name}_segmentation_dice", "Segmentation on Reconstruction"),
                    ]

            metrics, overall_metrics = apply_function_to_single_column(
                    grouped_df,
                    "grade_group",  # Update grouping key
                    cols,
                    lambda x: x.mean(),
                    lambda x: x.std(),
                )  

            grouped_bar_chart(
                df=metrics,
                overall_df=overall_metrics,
                x=x,
                x_label=x_label,
                y="value",
                y_label="Dice Coefficient" if objective == "dice" else "Sum",
                std_y="std_y",
                color="metric",
                color_label="Metric",
                category_order={},
                title=f"Segmentation {objective} by {x_label}",
                output_dir=output_dir,
                output_name=f"{name}_segmentation_{objective}.png",
            )

            cols = [
                ("gt_sum", "Ground Truth (GT)", "segmentation_sum", "Segmentation on GT"), 
                ("segmentation_sum", "Segmentation on GT", f"{name}_segmentation_sum", "Segmentation on Reconstruction"), 
                ("gt_sum", "Ground Truth (GT)", f"{name}_segmentation_sum", "Segmentation on Reconstruction")
            ] if objective == "sum" else [
                ("segmentation_dice", "Segmentation on GT", f"{name}_segmentation_dice", "Segmentation on Reconstruction")
            ]

            # significance
            significance, overall_significance = apply_function_to_column_pairs(
                grouped_df,
                "grade_group",  # Update grouping key
                cols,
                lambda x, y: hypothesis_test(y, x))
            
            grouped_bar_chart(
                df=significance,
                overall_df=overall_significance,
                x=x,
                x_label=x_label,
                y="value",
                y_label="Significance",
                std_y=None,
                color="metric",
                color_label="Metric",
                category_order={},
                title=f"{objective} Significance by {x_label}",
                output_dir=output_dir,
                output_name=f"{name}_significance_{objective}.png",
            )

def evaluate_segmentation(df, segmentation_model, reconstruction_models, output_dir):
    individual_evaluation(df, segmentation_model, reconstruction_models, output_dir)