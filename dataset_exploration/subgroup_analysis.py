import pandas as pd
import numpy as np

class PatientDataAnalysis:
    def __init__(self, file_path):
        # Load the CSV file into a pandas DataFrame
        self.data = pd.read_csv(file_path)
        # filter for split 
        self.data = self.data[self.data['split'] == 'test']
        # only one per patient_id 
        self.data = self.data.drop_duplicates(subset='patient_id')
        
        # Initialize median age and highest OS of dead patients
        self.median_age = self.data['age_at_mri'].median()
        self.highest_os_dead = self.data[self.data['alive'] == 1]['os'].max()

        # Add Age Group column (Old/Young based on median)
        self.data['Age Group'] = self.data['age_at_mri'].apply(lambda x: 'Old' if x >= self.median_age else 'Young')
        
        # Add combined Gender and Age Group column
        self.data['Subgroup'] = self.data['sex'] + '-' + self.data['Age Group']

        # Calculate bin size for OS
        self.bin_size = self.highest_os_dead / 4

        # Assign OS bucket for each patient
        self.data['OS Bucket'] = self.data['os'].apply(self.assign_os_bucket)

    def print_basic_statistics(self):
        print(f"Median age: {self.median_age}")
        print(f"Highest OS of any dead patient: {self.highest_os_dead}")
        print(f"Bin size for OS: {self.bin_size}")
        
    def assign_os_bucket(self, os_value):
        # Calculate the bucket based on the bin size and cap at 3
        bucket = int(np.floor(os_value / self.bin_size))
        return min(bucket, 3)

    def count_subgroups(self):
        # Count how many samples fall into each subgroup
        subgroup_counts = self.data['Subgroup'].value_counts()
        print("Subgroup counts:")
        print(subgroup_counts)
    
    def count_subgroups_with_buckets(self):
        # Count how many patients are in each subgroup and bucket combination
        subgroup_bucket_counts = self.data.groupby(['Subgroup', 'OS Bucket']).size().unstack(fill_value=0)
        print("Subgroup and OS Bucket counts:")
        print(subgroup_bucket_counts)

    def classify_by_column(self, column_name, condition):
        """
        Classify patients based on a column and a condition.
        Condition is a function applied to the column values.
        """
        classified_data = self.data[self.data[column_name].apply(condition)]
        
        # Count classified samples across subgroups
        classified_counts = classified_data['Subgroup'].value_counts()
        print(f"Classification results for {column_name} with given condition:")
        print(classified_counts)

    def classify_by_column_and_bucket(self, column_name, condition):
        """
        Classify patients based on a column and a condition, and show OS bucket distribution.
        Condition is a function applied to the column values.
        """
        classified_data = self.data[self.data[column_name].apply(condition)]
        
        # Count classified samples across subgroups and OS buckets
        classified_counts = classified_data.groupby(['Subgroup', 'OS Bucket']).size().unstack(fill_value=0)
        print(f"Classification results for {column_name} with given condition:")
        print(classified_counts)


# Example of how you can use the class with your CSV file

# Initialize the analysis with the CSV file path
file_path = '../../../data/UCSF-PDGM/metadata.csv'  # Replace with your actual CSV file path
analysis = PatientDataAnalysis(file_path)

# Print basic statistics
analysis.print_basic_statistics()

# Count how many samples are in each of the four subgroups
analysis.count_subgroups()

# Example: classify patients by MGMT status being 'methylated'
analysis.classify_by_column('final_diagnosis', lambda x: x != 'Glioblastoma, IDH-wildtype')
analysis.classify_by_column('who_cns_grade', lambda x: x != 4)
analysis.count_subgroups_with_buckets()

# You can add more classification criteria as needed.

