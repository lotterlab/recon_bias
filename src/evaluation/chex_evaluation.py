import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import os
import plotly.express as px
from PIL import Image
import plotly.graph_objects as go


def calculate_roc_auc_safely(y_true, y_pred, pathology=None):
    """Calculate ROC AUC score with safety checks for number of classes."""
    unique_classes = np.unique(y_true)
    if len(unique_classes) < 2:
        print(f"Only found classes {unique_classes} in ground truth{f' for {pathology}' if pathology else ''} - ROC AUC cannot be calculated")
        return np.nan
    return roc_auc_score(y_true, y_pred)


def grouped_bar_chart(
    df,
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
    baseline_performance=None
):
    """Create a grouped bar chart with an additional bar plot representing overall performance."""

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define labels for the plot
    labels = {x: x_label, y: y_label, color: color_label}

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
        # Add category orders for photon count to ensure high to low ordering
        if facet_col == "photon_count":
            bar_chart_kwargs["category_orders"] = {
                "photon_count": sorted(df["photon_count"].unique(), reverse=True)
            }

    if error_y is not None:
        bar_chart_kwargs["error_y"] = error_y

    # Create the detailed grouped bar chart using Plotly
    fig_bar = px.bar(df, **bar_chart_kwargs)

    # Add baseline if provided
    if baseline_performance is not None:
        if isinstance(baseline_performance, dict):
            # Define different styles for each group
            styles = {
                'dash': ['dash', 'dot', 'dashdot', 'longdash', 'longdashdot'],
                'color': ['gray', 'darkgray', 'dimgray', 'lightgray', 'slategray']
            }
            
            # For grouped plots, add a line for each group's baseline
            for i, (group, value) in enumerate(sorted(baseline_performance.items())):
                dash_style = styles['dash'][i % len(styles['dash'])]
                color_style = styles['color'][i % len(styles['color'])]
                
                fig_bar.add_shape(
                    type="line",
                    x0=0,
                    x1=1,
                    y0=value,
                    y1=value,
                    xref="paper",
                    line=dict(
                        dash=dash_style,
                        color=color_style,
                        width=2
                    )
                )
                
                # Add to legend using an invisible scatter trace
                fig_bar.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode='lines',
                        line=dict(dash=dash_style, color=color_style, width=2),
                        name=f"Original Classification ({group})",
                        showlegend=True
                    )
                )
        else:
            # For overall plot, use the same approach
            fig_bar.add_shape(
                type="line",
                x0=0,
                x1=1,
                y0=baseline_performance,
                y1=baseline_performance,
                xref="paper",
                line=dict(
                    dash="dash",
                    color="gray",
                    width=2
                )
            )
            
            # Add to legend using an invisible scatter trace
            fig_bar.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='lines',
                    line=dict(dash="dash", color="gray", width=2),
                    name="Original Classification",
                    showlegend=True
                )
            )

    # Update layout with better spacing and legend formatting
    fig_bar.update_layout(
        yaxis_tickformat=".2f",
        uniformtext_minsize=8,
        uniformtext_mode="hide",
        title_x=0.5,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,  # Move legend further right
            itemwidth=40,  # Fixed width for legend items
            font=dict(size=10),  # Adjust font size if needed
        ),
        width=1200,  # Increase overall width
        height=600,  # Adjust height if needed
        margin=dict(r=150),  # Add right margin for legend
    )

    # Adjust spacing between facets
    fig_bar.update_layout(
        bargap=0.15,  # Gap between bars in the same group
        bargroupgap=0.1,  # Gap between bar groups
    )

    # Remove text on bars completely
    fig_bar.update_traces(
        textposition="none",
        selector=dict(type="bar")
    )

    # Save the detailed grouped bar chart as an image
    bar_chart_path = os.path.join(output_dir, f"{output_name}.png")
    fig_bar.write_image(bar_chart_path)


def apply_function_to_single_column(
    grouped_df, reconstruction_models, group_by, column, error=None
):
    """Calculate the significance between the GT and prediction/reconstruction values."""
    results = []
    photon_counts = np.unique([model["photon_count"] for model in reconstruction_models])

    for group_keys, group in grouped_df:
        for photon_count in photon_counts:
            for reconstruction_model in reconstruction_models:
                if reconstruction_model["photon_count"] != photon_count:
                    continue

                recon_column = group[
                    f"{column}_{reconstruction_model['network']}_{reconstruction_model['photon_count']}"
                ]
                
                valid_mask = ~(group[column].isna() | recon_column.isna() | group[column].isin([-1]) | recon_column.isin([-1]))
                if valid_mask.sum() == 0:
                    continue
                    
                result = calculate_roc_auc_safely(
                    group[column][valid_mask],
                    recon_column[valid_mask],
                    pathology=f"{column} ({group_keys})"
                )

                error_result = None
                if error:
                    error_result = error(recon_column[valid_mask])

                dict = {
                    f"{group_by}": group_keys,
                    "model": f"{reconstruction_model['network']}",
                    "photon_count": reconstruction_model["photon_count"],
                    "value": result,
                }

                if error_result:
                    dict["error"] = error_result

                results.append(dict)

    metrics_df = pd.DataFrame(results)
    return metrics_df


def plot_classifier_metrics(df, pathologies, reconstruction_models, output_dir):
    # add column that splits age into young and old, according to the median age in the df
    median_age = df["Age"].median()
    df["age_bin"] = df["Age"].apply(lambda x: "young" if x < median_age else "old")
    groups = ["age_bin", "Sex", "Race"]
    group_map = {
        "age_bin": "Age",
        "Sex": "Sex",
        "Race": "Race",
    }

    base_dir = output_dir

    for pathology in pathologies:
        output_dir = os.path.join(base_dir, pathology)        

        # Calculate baseline performance using original classification
        valid_mask = ~(df[pathology].isna() | df[f"{pathology}_class"].isna() | df[pathology].isin([-1]) | df[f"{pathology}_class"].isin([-1]))
        baseline = calculate_roc_auc_safely(
            df[pathology][valid_mask],
            df[f"{pathology}_class"][valid_mask],
            pathology=f"{pathology} (baseline)"
        )

        # Overall performance plot
        overall_results = apply_function_to_single_column(
            [(None, df)],
            reconstruction_models,
            "group",
            pathology,
        )
        overall_results["group"] = "Overall"
        
        grouped_bar_chart(
            overall_results,
            x="group",
            x_label="Overall Performance",
            y="value",
            y_label="AUROC",
            color="model",
            color_label="Model",
            category_order={},
            title=f"{pathology} Overall Performance",
            output_dir=output_dir,
            output_name=f"{pathology}_overall",
            facet_col="photon_count",
            facet_col_label="Photon Count",
            facet_row=None,
            facet_row_label=None,
            error_y=None,
            baseline_performance=baseline
        )

        # For grouped analysis, calculate baseline per group
        for group in groups:
            df_copy = df.copy()
            grouped_df = df_copy.groupby(group, observed=False)

            # Calculate baseline for each group
            group_baselines = {}
            for group_name, group_data in grouped_df:
                valid_mask = ~(group_data[pathology].isna() | group_data[f"{pathology}_class"].isna() | group_data[pathology].isin([-1]) | group_data[f"{pathology}_class"].isin([-1]))
                if valid_mask.sum() > 0:  # Only calculate if we have valid data
                    group_baselines[group_name] = calculate_roc_auc_safely(
                        group_data[pathology][valid_mask],
                        group_data[f"{pathology}_class"][valid_mask],
                        pathology=f"{pathology} ({group_name})"
                    )

            results = apply_function_to_single_column(
                grouped_df,
                reconstruction_models,
                group,
                pathology,
            )

            grouped_bar_chart(
                results,
                x=group,
                x_label=group_map[group],
                y="value",
                y_label="AUROC",
                color="model",
                color_label="Model",
                category_order={},
                title=f"{pathology} predictions grouped by {group_map[group]}",
                output_dir=output_dir,
                output_name=f"{pathology}_predictions_{group_map[group]}",
                facet_col="photon_count",
                facet_col_label="Photon Count",
                facet_row=None,
                facet_row_label=None,
                error_y=None,
                baseline_performance=group_baselines
            )


def compute_thresh_eq_sens_spec(scores, y):
    """Find threshold where sensitivity equals specificity using ROC curve."""
    fpr, sens, threshs = roc_curve(y, scores)
    spec = 1 - fpr
    return threshs[np.argmin(np.abs(spec - sens))]

def calculate_fairness_metrics(y_true, y_pred_proba, protected_attribute, threshold=None):
    """Calculate fairness metrics for binary classification."""
    # Find optimal threshold if none provided
    if threshold is None:
        # Remove any NaN or -1 values before computing threshold
        valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred_proba) | (y_true == -1))
        if not valid_mask.any():
            threshold = 0.5
        else:
            threshold = compute_thresh_eq_sens_spec(y_pred_proba[valid_mask], y_true[valid_mask])
    
    # Convert to binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Remove any NaN values and -1 values
    valid_mask = ~(
        np.isnan(y_true) | 
        np.isnan(y_pred) | 
        pd.isna(protected_attribute) | 
        (y_true == -1) | 
        (y_pred == -1)
    )
    
    if not valid_mask.any():
        return {
            'demographic_parity': np.nan,
            'equalized_odds': np.nan,
            'equal_opportunity': np.nan
        }
    
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    protected_attribute = protected_attribute[valid_mask]
    
    groups = np.unique(protected_attribute)
    if len(groups) < 2:
        return {
            'demographic_parity': np.nan,
            'equalized_odds': np.nan,
            'equal_opportunity': np.nan
        }
    
    # Initialize metrics
    dem_parity = {}
    eq_odds_tpr = {}
    eq_odds_fpr = {}
    eq_opp = {}
    
    # Calculate metrics for each group
    for group in groups:
        mask = protected_attribute == group
        if not mask.any():  # Skip if no samples in this group
            continue
            
        # Demographic Parity
        dem_parity[group] = np.mean(y_pred[mask])
        
        # For TPR and FPR calculations
        pos_mask = y_true[mask] == 1
        neg_mask = y_true[mask] == 0
        
        # Skip if no positive/negative samples
        if pos_mask.any():
            tpr = np.mean(y_pred[mask][pos_mask])
            eq_odds_tpr[group] = tpr
            eq_opp[group] = tpr
        
        if neg_mask.any():
            eq_odds_fpr[group] = np.mean(y_pred[mask][neg_mask])
    
    # Only calculate metrics if we have enough data
    metrics = {}
    if len(dem_parity) >= 2:
        metrics['demographic_parity'] = max(dem_parity.values()) - min(dem_parity.values())
    else:
        metrics['demographic_parity'] = np.nan
        
    if len(eq_odds_tpr) >= 2 and len(eq_odds_fpr) >= 2:
        metrics['equalized_odds'] = (
            (max(eq_odds_tpr.values()) - min(eq_odds_tpr.values()) +
             max(eq_odds_fpr.values()) - min(eq_odds_fpr.values())) / 2
        )
    else:
        metrics['equalized_odds'] = np.nan
        
    if len(eq_opp) >= 2:
        metrics['equal_opportunity'] = max(eq_opp.values()) - min(eq_opp.values())
    else:
        metrics['equal_opportunity'] = np.nan
    
    return metrics

def plot_fairness_metrics(df, pathologies, reconstruction_models, output_dir):
    """Plot fairness metrics across different protected attributes."""
    protected_attributes = ["Sex", "Race", "age_bin"]
    group_map = {
        "age_bin": "Age",
        "Sex": "Sex",
        "Race": "Race",
    }
    metric_names = {
        "demographic_parity": "DP",
        "equalized_odds": "EODD",
        "equal_opportunity": "EOPP"
    }
    
    # Add age binning if not already present
    if "age_bin" not in df.columns:
        median_age = df["Age"].median()
        df["age_bin"] = df["Age"].apply(lambda x: "young" if x < median_age else "old")

    base_dir = output_dir

    for pathology in pathologies:
        output_dir = os.path.join(base_dir, pathology)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for protected_attr in protected_attributes:
            results = []
            baseline_results = []
            
            # Calculate baseline metrics using original classification
            y_true = df[pathology].values
            y_pred = df[f"{pathology}_class"].values
            valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            baseline_metrics = calculate_fairness_metrics(
                y_true[valid_mask],
                y_pred[valid_mask],
                df[protected_attr].values[valid_mask]
            )
            
            # Store baseline metrics
            for metric_name, value in baseline_metrics.items():
                baseline_results.append({
                    'metric': metric_names[metric_name],
                    'value': value
                })
            
            # Calculate metrics for each model and photon count
            for model in reconstruction_models:
                pred_col = f"{pathology}_{model['network']}_{model['photon_count']}"
                y_pred = df[pred_col].values
                valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                
                metrics = calculate_fairness_metrics(
                    y_true[valid_mask],
                    y_pred[valid_mask],
                    df[protected_attr].values[valid_mask]
                )
                
                for metric_name, value in metrics.items():
                    results.append({
                        "model": model["network"],
                        "photon_count": model["photon_count"],
                        "metric": metric_names[metric_name],
                        "value": value
                    })
            
            # Create DataFrame
            results_df = pd.DataFrame(results)
            baseline_df = pd.DataFrame(baseline_results)
            
            # Create single plot with all metrics
            grouped_bar_chart(
                results_df,
                x="metric",
                x_label="Fairness Metric",
                y="value",
                y_label="Disparity",
                color="model",
                color_label="Model",
                category_order={},
                title=f"Fairness Metrics for {pathology} by {group_map[protected_attr]}",
                output_dir=output_dir,
                output_name=f"{pathology}_fairness_{group_map[protected_attr]}",
                facet_col="photon_count",
                facet_col_label="Photon Count",
                baseline_performance=dict(zip(baseline_df['metric'], baseline_df['value']))
            )

def plot_image_metrics(df, reconstruction_models, output_dir):
    """Plot image quality metrics (PSNR, SSIM, NRMSE) across different groups."""
    # Define metrics and groups
    image_metrics = ["psnr", "ssim", "nrmse"]
    metric_names = {
        "psnr": "PSNR",
        "ssim": "SSIM",
        "nrmse": "NRMSE"
    }
    protected_attributes = ["Sex", "Race", "age_bin"]
    group_map = {
        "age_bin": "Age",
        "Sex": "Sex",
        "Race": "Race",
    }
    
    # Add age binning if not already present
    if "age_bin" not in df.columns:
        median_age = df["Age"].median()
        df["age_bin"] = df["Age"].apply(lambda x: "young" if x < median_age else "old")

    base_dir = output_dir
    output_dir = os.path.join(base_dir, "image_metrics")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # For each metric
    for metric in image_metrics:
        # First create overall plot (no grouping)
        results = []
        for model in reconstruction_models:
            metric_col = f"{model['network']}_{model['photon_count']}_{metric}"
            results.append({
                "model": model["network"],
                "photon_count": model["photon_count"],
                "value": df[metric_col].mean(),
                "group": "Overall"
            })
        
        overall_df = pd.DataFrame(results)
        grouped_bar_chart(
            overall_df,
            x="group",
            x_label="Overall Performance",
            y="value",
            y_label=metric_names[metric],
            color="model",
            color_label="Model",
            category_order={},
            title=f"Overall {metric_names[metric]}",
            output_dir=output_dir,
            output_name=f"{metric}_overall",
            facet_col="photon_count",
            facet_col_label="Photon Count",
            error_y=None
        )

        # Then create grouped plots
        for attr in protected_attributes:
            results = []
            for model in reconstruction_models:
                metric_col = f"{model['network']}_{model['photon_count']}_{metric}"
                # Calculate mean for each group
                group_means = df.groupby(attr)[metric_col].mean().reset_index()
                
                for _, row in group_means.iterrows():
                    results.append({
                        "model": model["network"],
                        "photon_count": model["photon_count"],
                        "value": row[metric_col],
                        "group": row[attr]
                    })
            
            results_df = pd.DataFrame(results)
            grouped_bar_chart(
                results_df,
                x="group",
                x_label=attr,
                y="value",
                y_label=metric_names[metric],
                color="model",
                color_label="Model",
                category_order={},
                title=f"{metric_names[metric]} by {group_map[attr]}",
                output_dir=output_dir,
                output_name=f"{metric}_{group_map[attr]}",
                facet_col="photon_count",
                facet_col_label="Photon Count",
                error_y=None
            )