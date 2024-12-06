import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# Function to process the data and generate plots
def plot_auroc_and_significance(df, output_dir):
    results = []
    pathology_aurocs = {}

    # Compute AUROCs for each pathology
    for pathology in df['pathology'].unique():
        subset = df[df['pathology'] == pathology]
        if len(subset['gt'].unique()) == 1:
            print(f"Only one class present for pathology: {pathology}")
            continue  # Skip AUROC calculation if only one class is present
        auroc_pred = roc_auc_score(subset['gt'], subset['pred'])
        auroc_pred_recon = roc_auc_score(subset['gt'], subset['pred_recon'])
        pathology_aurocs[pathology] = (auroc_pred, auroc_pred_recon)
        results.append({
            "pathology": pathology,
            "count": len(subset),
            "auroc_pred": auroc_pred,
            "auroc_pred_recon": auroc_pred_recon,
        })

    results_df = pd.DataFrame(results)

    # Format labels with sample counts
    results_df["formatted_label"] = results_df.apply(
        lambda row: f"{row['pathology']} (#{row['count']})", axis=1
    )

    # Plot 1: AUROC for pred and pred_recon
    plt.figure(figsize=(12, 8))
    x_positions = np.arange(len(results_df))
    plt.bar(x_positions - 0.2, results_df["auroc_pred"], width=0.4, label="Pred")
    plt.bar(x_positions + 0.2, results_df["auroc_pred_recon"], width=0.4, label="Pred Recon")
    plt.xticks(x_positions, results_df["formatted_label"], rotation=90)
    plt.xlabel("Pathology")
    plt.ylabel("AUROC")
    plt.title("AUROC Comparison for Pred and Pred Recon")

    # Annotate AUROC values on bars
    for i, (auroc_pred, auroc_recon) in enumerate(zip(results_df["auroc_pred"], results_df["auroc_pred_recon"])):
        plt.text(i - 0.2, auroc_pred + 0.01, f"{auroc_pred:.2f}", ha='center', va='bottom')
        plt.text(i + 0.2, auroc_recon + 0.01, f"{auroc_recon:.2f}", ha='center', va='bottom')

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/auroc_comparison.png")
    plt.close()

    # Plot 2: Significance of difference between AUROCs
    p_values = []
    for pathology, (auroc_pred, auroc_pred_recon) in pathology_aurocs.items():
        # Simulate the sampling distribution to calculate p-value
        subset = df[df['pathology'] == pathology]
        gt = subset['gt'].values
        bootstrap_diffs = []
        for _ in range(1000):  # Bootstrap
            indices = np.random.choice(len(gt), len(gt), replace=True)
            bootstrap_pred = subset['pred'].values[indices]
            bootstrap_pred_recon = subset['pred_recon'].values[indices]
            bootstrap_gt = gt[indices]
            auroc_bootstrap_pred = roc_auc_score(bootstrap_gt, bootstrap_pred)
            auroc_bootstrap_recon = roc_auc_score(bootstrap_gt, bootstrap_pred_recon)
            bootstrap_diffs.append(auroc_bootstrap_pred - auroc_bootstrap_recon)
        
        # Compute p-value (two-sided test)
        mean_diff = auroc_pred - auroc_pred_recon
        bootstrap_diffs = np.array(bootstrap_diffs)
        p_value = (np.sum(np.abs(bootstrap_diffs) >= np.abs(mean_diff)) + 1) / (len(bootstrap_diffs) + 1)
        p_values.append(p_value)

    results_df["p_value"] = p_values

    plt.figure(figsize=(12, 8))
    plt.bar(x_positions, results_df["p_value"])
    plt.xticks(x_positions, results_df["formatted_label"], rotation=90)
    plt.xlabel("Pathology")
    plt.ylabel("p-value")
    plt.title("Significance of Difference in AUROCs")
    plt.axhline(0.05, color="red", linestyle="--", label="p=0.05")

    # Annotate p-values on bars
    for i, p_value in enumerate(results_df["p_value"]):
        plt.text(i, -np.log10(p_value) + 0.1, f"{p_value:.3f}", ha='center', va='bottom')

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/significance_comparison.png")
    plt.close()

    # Save the results as a CSV
    results_csv_path = f"{output_dir}/pathology_auroc_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    return results_csv_path