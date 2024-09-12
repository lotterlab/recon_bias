import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap


def plot_data(csv, output_dir): 
    # check if the output file already exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sns.set(style="whitegrid")
    cmap = plt.get_cmap('viridis')

    # Plot 1: Distribution of Sex with count and percentage
    plt.figure(figsize=(10, 6))
    values = csv['Sex'].value_counts()
    colors = [cmap(i / len(values)) for i in range(len(values))]  # Get color for each bar
    ax1 = csv['Sex'].value_counts().plot(kind='bar', color=colors)
    total = len(csv)
    for p in ax1.patches:
        count = p.get_height()
        percent = f'{100 * count / total:.1f}%'
        ax1.annotate(f'{count} ({percent})', (p.get_x() + p.get_width() / 2., count),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.title('Distribution of Sex', fontsize=16)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('Sex', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sex_distribution.png")

    # Plot 2: Age distribution in buckets with colormap
    age_bins = pd.cut(csv['Age at MRI'], bins=[0, 3, 18, 42, 67, 96], right=False)
    age_sex_dist = pd.crosstab(age_bins, csv['Sex'])
    plt.figure(figsize=(10, 6))
    ax = age_sex_dist.plot(kind='bar', stacked=True, color=[cmap(0.2), cmap(0.7)], edgecolor='black')
    total = age_sex_dist.sum().sum()
    for p in ax.patches:
        height = p.get_height()
        width = p.get_width()
        x = p.get_x()
        y = p.get_y()
        if height > 0:
            percent = f'{100 * height / total:.1f}%'
            annotation = f'{int(height)} ({percent})'  
            ax.annotate(annotation, (x + width / 2, y + height / 2), ha='center', va='center', fontsize=8)

    plt.title('Age Distribution by Sex', fontsize=16)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('Age Range', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(f"{output_dir}/age_distribution_by_sex.png")

    # Plot 3: WHO CNS Grade distribution with colormap
    plt.figure(figsize=(10, 6))
    values = csv['WHO CNS Grade'].value_counts()
    colors = [cmap(i / len(values)) for i in range(len(values))]
    ax3 = values.plot(kind='bar', color=colors)

    for p in ax3.patches:
        count = p.get_height()
        percent = f'{100 * count / total:.1f}%'
        ax3.annotate(f'{count} ({percent})', (p.get_x() + p.get_width() / 2., count),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plt.title('Distribution of WHO CNS Grade', fontsize=16)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('WHO CNS Grade', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/who_grade_distribution.png")

    # Plot 4: Dead or alive distribution with colormap
    plt.figure(figsize=(10, 6))
    values = csv['1-dead 0-alive'].value_counts()
    colors = [cmap(i / len(values)) for i in range(len(values))]
    ax4 = values.plot(kind='bar', color=colors)

    for p in ax4.patches:
        count = p.get_height()
        percent = f'{100 * count / total:.1f}%'
        ax4.annotate(f'{count} ({percent})', (p.get_x() + p.get_width() / 2., count),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plt.title('Distribution of Dead (1) vs Alive (0)', fontsize=16)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('Status', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dead_alive_distribution.png")

    # Plot 5: Distribution of Overall Survival (OS) in days with colormap
    plt.figure(figsize=(10, 6))
    plt.hist(csv['OS'], bins=20, color=cmap(0.5))  # Assuming OS is in months, converting to days
    plt.title('Distribution of Overall Survival (OS) in Days', fontsize=16)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('OS (days)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/os_distribution_days.png")

    # Plot 6: Age vs OS Correlation (Scatter Plot) with colormap of tumor grade
    csv['Tumor Grade'] = pd.to_numeric(csv['WHO CNS Grade'], errors='coerce')

    # Define a discrete colormap (we will use four discrete colors)
    cmap = ListedColormap(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])  # Custom colors for tumor grades

    # Plot 1: Age vs OS Correlation (Scatter Plot) with discrete tumor grade colors
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(csv['Age at MRI'], csv['OS'], c=csv['Tumor Grade'], cmap=cmap, s=50)

    # Add a colorbar with discrete tumor grades
    cbar = plt.colorbar(scatter, ticks=[1, 2, 3, 4])
    cbar.set_label('Tumor Grade')
    cbar.set_ticklabels(['I', 'II', 'III', 'IV'])  # Setting the discrete tumor grade labels

    plt.title('Age vs Overall Survival (OS) by Tumor Grade', fontsize=16)
    plt.xlabel('Age at MRI', fontsize=12)
    plt.ylabel('Overall Survival (days)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/age_vs_os_by_tumor_grade_scatter.png")

    # Plot 7: Tumor Grade vs OS (Boxplot) with colormap
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='WHO CNS Grade', y=csv['OS'], data=csv, palette='viridis')
    plt.title('WHO CNS Grade vs Overall Survival (OS)', fontsize=16)
    plt.xlabel('WHO CNS Grade', fontsize=12)
    plt.ylabel('Overall Survival (days)', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/who_grade_vs_os_boxplot.png")

    # Plot 8: OS distribution vs tumor grade
    plt.figure(figsize=(10, 6))
    grades = csv['WHO CNS Grade'].unique()
    for grade in grades:
        plt.hist(csv[csv['WHO CNS Grade'] == grade]['OS'], bins=10, alpha=0.5, label=grade)
    plt.title('Overall Survival Distribution by Tumor Grade', fontsize=16)
    plt.xlabel('Overall Survival (days)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Tumor Grade')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/os_distribution_by_tumor_grade.png")

    # Plot 9: Age vs Dead or Alive (Stacked Bar Plot) with colormap
    plt.figure(figsize=(10, 6))
    age_status = pd.crosstab(age_bins, csv['1-dead 0-alive'])
    age_status.plot(kind='bar', stacked=True, color=[cmap(0.7), cmap(0.2)], figsize=(10, 6))
    plt.title('Survival Status by Age Group', fontsize=16)
    plt.xlabel('Age Group', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/survival_status_by_age_group.png")

    # Plot 10: Boxplot showing Tumor Grade vs Age
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Tumor Grade', y='Age at MRI', data=csv, palette=cmap.colors)
    plt.title('Tumor Grade vs Age Distribution', fontsize=16)
    plt.xlabel('Tumor Grade', fontsize=12)
    plt.ylabel('Age at MRI', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/who_grade_vs_age_boxplot.png")

csv = pd.read_csv('USCF-PDGM/UCSF-PDGM-metadata.csv')
output_dir = "USCF-PDGM/statistics" 
plot_data(csv, output_dir)