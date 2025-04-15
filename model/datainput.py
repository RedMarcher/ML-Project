import pandas as pd

def load_data_from_xlsx(file_path, target, features):
    """
    Loads data from an Excel file and prepares it for machine learning.

    Args:
        file_path (str): Path to the Excel file.
        target_column (str): Name of the target column for machine learning.

    Returns:
        X (pd.DataFrame): Features for training.
        y (pd.Series): Target variable.
    """
    # Load the Excel file into a DataFrame
    data = pd.read_excel(file_path)

    # Separate features (X) and target (y)
    X = data[features]
    y = data[target]

    return X, y