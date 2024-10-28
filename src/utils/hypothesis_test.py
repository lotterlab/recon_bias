# Define helper functions for percentage difference and t-test

import numpy as np
import pandas as pd
import scipy.stats
from scipy.stats import chi2_contingency, ttest_ind


def hypothesis_test(col1, col2, test_type="auto"):
    """
    Calculate the statistical significance (p-value) between two columns of data.

    Parameters:
    col1 (pd.Series or np.array): The first column of data.
    col2 (np.Series or np.array): The second column of data.
    test_type (str): The type of test to use ('t-test', 'chi2', 'auto'). Default is 'auto'.

    Returns:
    float: The p-value of the significance test. If the test cannot be performed, returns np.nan.
    """

    # Remove NaN or infinite values from both columns
    col1 = pd.Series(col1).replace([np.inf, -np.inf], np.nan).dropna()
    col2 = pd.Series(col2).replace([np.inf, -np.inf], np.nan).dropna()

    # Ensure both columns have the same length after cleaning
    min_length = min(len(col1), len(col2))
    col1 = col1[:min_length]
    col2 = col2[:min_length]

    # Check if the columns have any variation (needed for statistical testing)
    if np.all(col1 == col1.iloc[0]) or np.all(col2 == col2.iloc[0]):
        return 1  # Return NaN if there is no variation

    # Automatically determine the type of test based on data or use provided test type
    if test_type == "auto":
        if (
            col1.dtype == np.int64
            or col2.dtype == np.int64
            or col1.dtype == bool
            or col2.dtype == bool
        ):
            test_type = "chi2"
        else:
            test_type = "t-test"

    try:
        if test_type == "t-test":
            # Perform a two-sample t-test for continuous data
            _, p_value = ttest_ind(col1, col2, equal_var=False)  # Welch's t-test
            return p_value

        elif test_type == "chi2":
            # Perform a chi-square test for categorical data
            contingency_table = pd.crosstab(col1, col2)
            _, p_value, _, _ = chi2_contingency(contingency_table)
            return p_value

    except Exception as e:
        # Return NaN if any error occurs (e.g., invalid data)
        print(f"Error: {e}")
        return 1
