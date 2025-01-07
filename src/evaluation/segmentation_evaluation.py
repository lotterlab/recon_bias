import matplotlib.pyplot as plt
import numpy as np
import os

def create_metric_plot(df, metric, models, accelerations, output_dir, 
                      group_by=None, group_values=None, group_labels=None,
                      baseline_value=None, baseline_label=None):
    """
    Create plot for a specific metric.
    """
    plt.figure(figsize=(10, 6))
    
    # Colors from the screenshot
    colors = ['#5052F8', '#E73D2E', '#1AC586']  # Blue, Red, Green
    background_color = '#DFE7F4'  # Light gray
    line_styles = ['-', '--']  # Solid for group 1, dashed for group 2
    
    # Set plot style
    ax = plt.gca()
    ax.set_facecolor(background_color)
    ax.grid(True, color='white', linestyle='-', linewidth=1)
    
    # Determine the actual group type before plotting
    actual_group = group_by
    if group_by == 'temp_group':
        if '≤58' in group_labels:
            actual_group = 'age'
        elif '<4' in group_labels:
            actual_group = 'grade'
        elif 'GBM' in group_labels:
            actual_group = 'diagnosis'
    
    for model_idx, model in enumerate(models):
        if group_by is None:
            # Plot for overall performance without creating legend entries for each point
            for acc in accelerations:
                col_name = f"{model}_{acc}_{metric}"
                means = df[col_name].mean()
                stds = df[col_name].std()
                plt.errorbar(acc, means, yerr=stds, color=colors[model_idx],
                           marker='o', capsize=5,
                           capthick=1, elinewidth=1)
                
            # Connect points for each model and create single legend entry
            acc_values = [float(acc) for acc in accelerations]
            mean_values = [df[f"{model}_{acc}_{metric}"].mean() for acc in accelerations]
            plt.plot(acc_values, mean_values, color=colors[model_idx], label=f"{model}")
            
        else:
            # Plot for subgroups
            for group_idx, (value, label) in enumerate(zip(group_values, group_labels)):
                group_df = df[df[group_by] == value]
                
                means = [group_df[f"{model}_{acc}_{metric}"].mean() for acc in accelerations]
                stds = [group_df[f"{model}_{acc}_{metric}"].std() for acc in accelerations]
                
                plt.errorbar(accelerations, means, yerr=stds, 
                           color=colors[model_idx], linestyle=line_styles[group_idx],
                           marker='o', capsize=5,
                           capthick=1, elinewidth=1)
                plt.plot(accelerations, means, color=colors[model_idx], 
                        linestyle=line_styles[group_idx],
                        label=f"{model} - {label}")
    
    # Add baseline if provided
    if baseline_value is not None:
        if metric == 'segmentation_dice':
            baseline_label = 'Original Segmentation Dice'
        elif metric == 'segmentation_sum':
            baseline_label = 'Original Segmentation'
        plt.axhline(y=baseline_value, color='gray', linestyle=':', label=baseline_label)
        
        # Add GT sum line for segmentation_sum plots
        if metric == 'segmentation_sum':
            gt_value = df['gt_sum'].mean()
            plt.axhline(y=gt_value, color='gray', linestyle='--', label='Ground Truth')
    
    plt.xlabel('Acceleration Rate')
    plt.ylabel('Sum' if metric == 'segmentation_sum' else metric.capitalize())
    
    # Create title with proper capitalization
    metric_display = 'Sum' if metric == 'segmentation_sum' else metric.capitalize()
    if group_by:
        group_display = actual_group.replace('_', ' ').title()
        if actual_group == 'who_cns_grade':
            group_display = 'Grade Group (<4 or 4)'
        elif actual_group == 'final_diagnosis':
            group_display = 'Diagnosis'
        plt.title(f'{metric_display} by {group_display}')
    else:
        plt.title(f'{metric_display} vs Acceleration Rate')
    
    plt.legend()
    
    # Set x-axis ticks to only show the acceleration values we're using
    plt.xticks(accelerations)
    
    # Save plot with more specific names for group plots
    group_suffix = f"_by_{actual_group}" if group_by else ""
    plt.savefig(os.path.join(output_dir, f'{metric}{group_suffix}.png'))
    plt.close()

def evaluate_segmentation(df, reconstruction_models, output_dir):
    """
    Generate and save evaluation plots for segmentation results.
    """
    # Extract models and accelerations from reconstruction_models
    models = list(set(model['network'] for model in reconstruction_models))
    accelerations = sorted(list(set(model['acceleration'] for model in reconstruction_models)))
    
    # Define metrics to plot
    metrics = ['segmentation_sum', 'segmentation_dice', 'psnr', 'ssim', 'nrmse']
    
    # Define grouping criteria
    groupings = [
        ('sex', ['M', 'F'], ['Male', 'Female']),
        ('age', [lambda x: x <= 58, lambda x: x > 58], ['≤58', '>58']),
        ('who_cns_grade', [lambda x: x < 4, lambda x: x == 4], ['<4', '4']),
        ('final_diagnosis', 
         [lambda x: x == "Glioblastoma, IDH-wildtype", 
          lambda x: x != "Glioblastoma, IDH-wildtype"],
         ['GBM IDH-wt', 'Other'])
    ]
    
    # Create overall plots
    for metric in metrics:
        baseline_value = None
        baseline_label = None
        
        if metric == 'segmentation_sum':
            baseline_value = df['segmentation_sum'].mean()
            baseline_label = 'Segmentation on GT Sum'
        elif metric == 'segmentation_dice':
            baseline_value = df['segmentation_dice'].mean()
            baseline_label = 'Segmentatoin on GT Dice'
            
        create_metric_plot(df, metric, models, accelerations, output_dir,
                         baseline_value=baseline_value, baseline_label=baseline_label)
    
    # Create subgroup plots
    for group_col, group_conditions, group_labels in groupings:
        for metric in metrics:
            # Create temporary columns for complex conditions
            if callable(group_conditions[0]):
                temp_df = df.copy()
                temp_df['temp_group'] = temp_df[group_col].apply(
                    lambda x: group_labels[0] if group_conditions[0](x) else group_labels[1]
                )
                create_metric_plot(temp_df, metric, models, accelerations, output_dir,
                                 group_by='temp_group', 
                                 group_values=group_labels,
                                 group_labels=group_labels)
            else:
                create_metric_plot(df, metric, models, accelerations, output_dir,
                                 group_by=group_col,
                                 group_values=group_conditions,
                                 group_labels=group_labels)