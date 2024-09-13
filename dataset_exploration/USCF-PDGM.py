import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap
import textwrap


def plot_data(csv, output_dir): 
    # check if the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set uniform style and context
    sns.set(style="whitegrid")
    sns.set_context("talk")  # Ensures consistent font sizes across all plots
    
    cmap = plt.get_cmap('viridis')

    # Define a uniform figure size
    figure_size = (10, 6)

    # Plot 1: Distribution of Sex with count and percentage
    plt.figure(figsize=figure_size)
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
    plt.ylabel('Count', fontsize=16)
    plt.xlabel('Sex', fontsize=16)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sex_distribution.png")

    # Plot 2: Age distribution in buckets with colormap
    age_bins = pd.cut(csv['Age at MRI'], bins=[0, 3, 18, 42, 67, 96], right=False)
    age_sex_dist = pd.crosstab(age_bins, csv['Sex'])
    plt.figure(figsize=figure_size)
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
    plt.ylabel('Count', fontsize=16)
    plt.xlabel('Age Range', fontsize=16)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/age_distribution_by_sex.png")

    # Plot 3: WHO CNS Grade distribution with colormap
    plt.figure(figsize=figure_size)
    values = csv['WHO CNS Grade'].value_counts()
    colors = [cmap(i / len(values)) for i in range(len(values))]
    ax3 = values.plot(kind='bar', color=colors)

    for p in ax3.patches:
        count = p.get_height()
        percent = f'{100 * count / total:.1f}%'
        ax3.annotate(f'{count} ({percent})', (p.get_x() + p.get_width() / 2., count),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plt.title('Distribution of WHO CNS Grade', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.xlabel('WHO CNS Grade', fontsize=16)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/who_grade_distribution.png")

    # Plot 4: Dead or alive distribution with colormap
    plt.figure(figsize=figure_size)
    values = csv['1-dead 0-alive'].value_counts()
    colors = [cmap(i / len(values)) for i in range(len(values))]
    ax4 = values.plot(kind='bar', color=colors)

    for p in ax4.patches:
        count = p.get_height()
        percent = f'{100 * count / total:.1f}%'
        ax4.annotate(f'{count} ({percent})', (p.get_x() + p.get_width() / 2., count),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plt.title('Distribution of Dead (1) vs Alive (0)', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.xlabel('Status', fontsize=16)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dead_alive_distribution.png")

    # Plot 5: Distribution of Overall Survival (OS) in days with colormap
    plt.figure(figsize=figure_size)
    plt.hist(csv['OS'], bins=20, color=cmap(0.5))  # Assuming OS is in months, converting to days
    plt.title('Distribution of Overall Survival (OS) in Days', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.xlabel('OS (days)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/os_distribution_days.png")

    # Plot 6: Age vs OS Correlation (Scatter Plot) with colormap of tumor grade
    csv['Tumor Grade'] = pd.to_numeric(csv['WHO CNS Grade'], errors='coerce')
    valid_grades = sorted(csv['Tumor Grade'].dropna().unique())
    cmap = plt.get_cmap('viridis')
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(csv['Age at MRI'], csv['OS'], 
                        c=csv['Tumor Grade'], cmap=cmap, s=50)
    cbar = plt.colorbar(scatter, ticks=valid_grades)
    cbar.set_label('Tumor Grade')
    plt.title('Age vs Overall Survival (OS) by Tumor Grade', fontsize=16)
    plt.xlabel('Age at MRI', fontsize=16)
    plt.ylabel('Overall Survival (days)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/age_vs_os_by_tumor_grade_scatter.png")

    # Plot 7: Tumor Grade vs OS (Boxplot) with colormap
    plt.figure(figsize=figure_size)
    sns.boxplot(x='WHO CNS Grade', y=csv['OS'], data=csv, palette='viridis')
    plt.title('WHO CNS Grade vs Overall Survival (OS)', fontsize=16)
    plt.xlabel('WHO CNS Grade', fontsize=16)
    plt.ylabel('Overall Survival (days)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/who_grade_vs_os_boxplot.png")

    # Plot 8: OS distribution vs tumor grade
    plt.figure(figsize=figure_size)
    grades = csv['WHO CNS Grade'].unique()
    for grade in grades:
        plt.hist(csv[csv['WHO CNS Grade'] == grade]['OS'], bins=10, alpha=0.5, label=grade)
    plt.title('Overall Survival Distribution by Tumor Grade', fontsize=16)
    plt.xlabel('Overall Survival (days)', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.legend(title='Tumor Grade')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/os_distribution_by_tumor_grade.png")

    # Plot 9: Age vs Dead or Alive (Stacked Bar Plot) with colormap
    plt.figure(figsize=figure_size)
    age_status = pd.crosstab(age_bins, csv['1-dead 0-alive'])
    age_status.plot(kind='bar', stacked=True, color=[cmap(0.7), cmap(0.2)], figsize=figure_size)
    plt.title('Survival Status by Age Group', fontsize=16)
    plt.xlabel('Age Group', fontsize=16)
    plt.ylabel('Count', fontsize=16)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/survival_status_by_age_group.png")

    # Plot 10: Boxplot showing Tumor Grade vs Age
    plt.figure(figsize=figure_size)
    sns.boxplot(x='Tumor Grade', y='Age at MRI', data=csv, palette=cmap.colors)
    plt.title('Tumor Grade vs Age Distribution', fontsize=16)
    plt.xlabel('Tumor Grade', fontsize=16)
    plt.ylabel('Age at MRI', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/who_grade_vs_age_boxplot.png")

    # Plot 11: Bar plot showing final pathologic diagnosis
    diagnosis_counts = csv['Final pathologic diagnosis (WHO 2021)'].value_counts()
    colors = [cmap(i / len(diagnosis_counts)) for i in range(len(diagnosis_counts))]
    plt.figure(figsize=figure_size)
    wrapped_labels = [textwrap.fill(label, width=20) for label in diagnosis_counts.index]
    plt.bar(wrapped_labels, diagnosis_counts.values, color=colors)
    plt.title('Distribution of Final Pathologic Diagnoses', fontsize=16)
    plt.xlabel('Final Pathologic Diagnosis', fontsize=16)
    plt.ylabel('Number of Patients', fontsize=16)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/final_pathological_diagnosis.png")

    # Plot 12: Age, Sex, and Average Survival Days (Separate Bar per Gender)
    plt.figure(figsize=(10, 6))
    avg_survival_sex_age = csv.groupby([age_bins, 'Sex'])['OS'].mean().unstack()

    avg_survival_sex_age.plot(kind='bar', color=[cmap(0.3), cmap(0.7)], figsize=(10, 6), width=0.8)
    plt.title('Average Survival Days by Age and Sex', fontsize=16)
    plt.xlabel('Age Group', fontsize=16)
    plt.ylabel('Average Survival (days)', fontsize=16)
    plt.xticks(rotation=45)
    for p in plt.gca().patches:
        height = p.get_height()
        if height > 0:
            plt.gca().annotate(f'{int(height)}', 
                               (p.get_x() + p.get_width() / 2., height), 
                               ha='center', va='center', xytext=(0, 10), textcoords='offset points')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/age_sex_avg_survival.png')

    # Plot 13: Stacked Bar Chart by sex, age, who grade
    age_bins = pd.cut(csv['Age at MRI'], bins=[0, 3, 18, 42, 67, 96], right=False)
    grouped_data = csv.groupby([age_bins, 'Sex', 'WHO CNS Grade']).size().unstack(fill_value=0)
    male_data = grouped_data.loc[(slice(None), 'M'), :]
    female_data = grouped_data.loc[(slice(None), 'F'), :]
    age_bin_labels = grouped_data.index.get_level_values(0).unique()
    fig, ax = plt.subplots(figsize=(10, 6))
    values = csv['WHO CNS Grade'].value_counts()
    colors = [cmap(i / len(values)) for i in range(len(values))]

    num_bins = len(age_bin_labels)
    x = np.arange(num_bins)

    bottom_male = np.zeros(num_bins)
    for i, grade in enumerate(male_data.columns):
        bars_male = ax.bar(x - 0.2, male_data[grade], width=0.4, bottom=bottom_male, color=colors[i], label=f'WHO CNS Grade {grade}')
        bottom_male += male_data[grade].values
        for bar in bars_male:
            height = bar.get_height()
            if height > 0:  # Only annotate non-zero bars
                ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2),
                            ha='center', va='center', fontsize=9, color='white')

    bottom_female = np.zeros(num_bins)
    for i, grade in enumerate(female_data.columns):
        bars_female = ax.bar(x + 0.2, female_data[grade], width=0.4, bottom=bottom_female, color=colors[i])
        bottom_female += female_data[grade].values
        
        for bar in bars_female:
            height = bar.get_height()
            if height > 0:  # Only annotate non-zero bars
                ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2),
                            ha='center', va='center', fontsize=9, color='white')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{label}' for label in age_bin_labels])
    for i in range(num_bins):
        ax.text(i - 0.2, -0.1, 'M', ha='center', va='center', transform=ax.transData, fontsize=10)
        ax.text(i + 0.2, -0.1, 'F', ha='center', va='center', transform=ax.transData, fontsize=10)
    ax.set_xlabel('Age Bins')
    ax.set_ylabel('Count')
    ax.set_title('Stacked Bar Chart by WHO CNS Grade, Sex (M/F), and Age at MRI')
    ax.legend(title='WHO CNS Grade')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/age_sex_who_grade.png')

    # Plot 14: Stacked Bar Chart by sex, age, survival status
    age_bins = pd.cut(csv['Age at MRI'], bins=[0, 3, 18, 42, 67, 96], right=False)
    grouped_data = csv.groupby([age_bins, 'Sex', '1-dead 0-alive']).size().unstack(fill_value=0)
    male_data = grouped_data.loc[(slice(None), 'M'), :]
    female_data = grouped_data.loc[(slice(None), 'F'), :]
    age_bin_labels = grouped_data.index.get_level_values(0).unique()
    fig, ax = plt.subplots(figsize=(10, 6))
    values = csv['1-dead 0-alive'].value_counts()
    colors = [cmap(i / len(values)) for i in range(len(values))]

    num_bins = len(age_bin_labels)
    x = np.arange(num_bins)

    bottom_male = np.zeros(num_bins)
    for i, grade in enumerate(male_data.columns):
        bars_male = ax.bar(x - 0.2, male_data[grade], width=0.4, bottom=bottom_male, color=colors[i], label=f'WHO CNS Grade {grade}')
        bottom_male += male_data[grade].values
        for bar in bars_male:
            height = bar.get_height()
            if height > 0:  # Only annotate non-zero bars
                ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2),
                            ha='center', va='center', fontsize=9, color='white')

    bottom_female = np.zeros(num_bins)
    for i, grade in enumerate(female_data.columns):
        bars_female = ax.bar(x + 0.2, female_data[grade], width=0.4, bottom=bottom_female, color=colors[i])
        bottom_female += female_data[grade].values
        
        for bar in bars_female:
            height = bar.get_height()
            if height > 0:  # Only annotate non-zero bars
                ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2),
                            ha='center', va='center', fontsize=9, color='white')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{label}' for label in age_bin_labels])
    for i in range(num_bins):
        ax.text(i - 0.2, -0.1, 'M', ha='center', va='center', transform=ax.transData, fontsize=10)
        ax.text(i + 0.2, -0.1, 'F', ha='center', va='center', transform=ax.transData, fontsize=10)
    ax.set_xlabel('Age Bins')
    ax.set_ylabel('Count')
    ax.set_title('Stacked Bar Chart by Survival Status (1-dead 0-alive), Sex (M/F), and Age at MRI')
    ax.legend(title='Survival Status (1-dead 0-alive)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/age_sex_survival_status.png')

    # Plot 15: Stacked Bar Chart by sex, age, and pathological diagnosis
    age_bins = pd.cut(csv['Age at MRI'], bins=[0, 3, 18, 42, 67, 96], right=False)
    grouped_data = csv.groupby([age_bins, 'Sex', 'Final pathologic diagnosis (WHO 2021)']).size().unstack(fill_value=0)
    male_data = grouped_data.loc[(slice(None), 'M'), :]
    female_data = grouped_data.loc[(slice(None), 'F'), :]
    age_bin_labels = grouped_data.index.get_level_values(0).unique()
    fig, ax = plt.subplots(figsize=(10, 6))
    values = csv['Final pathologic diagnosis (WHO 2021)'].value_counts()
    colors = [cmap(i / len(values)) for i in range(len(values))]

    num_bins = len(age_bin_labels)
    x = np.arange(num_bins)

    bottom_male = np.zeros(num_bins)
    for i, grade in enumerate(male_data.columns):
        bars_male = ax.bar(x - 0.2, male_data[grade], width=0.4, bottom=bottom_male, color=colors[i], label=f'WHO CNS Grade {grade}')
        bottom_male += male_data[grade].values
        for bar in bars_male:
            height = bar.get_height()
            if height > 0:  # Only annotate non-zero bars
                ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2),
                            ha='center', va='center', fontsize=9, color='white')

    bottom_female = np.zeros(num_bins)
    for i, grade in enumerate(female_data.columns):
        bars_female = ax.bar(x + 0.2, female_data[grade], width=0.4, bottom=bottom_female, color=colors[i])
        bottom_female += female_data[grade].values
        
        for bar in bars_female:
            height = bar.get_height()
            if height > 0:  # Only annotate non-zero bars
                ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2),
                            ha='center', va='center', fontsize=9, color='white')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{label}' for label in age_bin_labels])
    for i in range(num_bins):
        ax.text(i - 0.2, -0.1, 'M', ha='center', va='center', transform=ax.transData, fontsize=10)
        ax.text(i + 0.2, -0.1, 'F', ha='center', va='center', transform=ax.transData, fontsize=10)
    ax.set_xlabel('Age Bins')
    ax.set_ylabel('Count')
    ax.set_title('Stacked Bar Chart by Pathologic Diagnosis (WHO 2021), Sex (M/F), and Age at MRI')
    ax.legend(title='Pathologic Diagnosis')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/age_sex_pathological_diagnosis.png')



csv = pd.read_csv('USCF-PDGM/UCSF-PDGM-metadata.csv')
output_dir = "USCF-PDGM/statistics"
plot_data(csv, output_dir)
