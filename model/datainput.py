import pandas as pd
import os

DATA_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'heart_og.csv')

"""
Loads data from an Excel file and prepares it for machine learning.

Args:
    file_path (str): Path to the Excel file.
    target_column (str): Name of the target column for machine learning.

Returns:
    X (pd.DataFrame): Features for training.
    y (pd.Series): Target variable.
"""

def load_data_from_xlsx(target, features, file_path=DATA_FILE_PATH):
    try:
        # Load the Excel file into a DataFrame
        data = pd.read_csv(file_path)

        # Separate features (X) and target (y)
        X = data[features]
        y = data[target]

        return X, y
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None
