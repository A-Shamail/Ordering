import numpy as np
import pandas as pd

def generate_redundant_dataset(redundancy_type, n_samples=5000):
    """
    Generates a dataset with redundancy between two features, plus random filler features.

    Parameters:
        redundancy_type (str): Type of redundancy ('linear', 'nonlinear', 'noisy_copy').
        n_samples (int): Number of samples to generate.

    Returns:
        pd.DataFrame: Dataset with redundant features and fillers.
    """
    np.random.seed(42)

    X1 = np.random.uniform(-1, 1, n_samples)

    if redundancy_type == "linear":
        X2 = 2 * X1 + 0.01 * np.random.normal(0, 1, n_samples)  # Small noise

    elif redundancy_type == 'trigonometric':
        X2 = np.cosh(np.sin(X1)) + 0.01 * np.random.normal(0, 1, n_samples)
    elif redundancy_type == 'type3':
        ...
    elif redundancy_type == 'type4':
        ...
    elif redundancy_type == 'type5':
        ...


    Y = X1 + np.random.normal(0, 0.05, n_samples)  # Y depends mainly on X1

    # Generate filler features
    fillers = {
        f'X{i}': np.random.normal(0, 1, n_samples) for i in range(3, 8)
    }

    # Create DataFrame
    df = pd.DataFrame({'X1': X1, 'X2': X2, **fillers, 'Y': Y})
    return df


def generate_dependent_dataset(dependency_type, n_samples=5000):
    """
    Generates a dataset where two features are dependent for predicting Y, plus random filler features.

    Parameters:
        dependency_type (str): Type of dependency ('multiplicative', 'conditional', 'complex').
        n_samples (int): Number of samples to generate.

    Returns:
        pd.DataFrame: Dataset with dependent features and fillers.
    """
    np.random.seed(42)

    X1 = np.random.uniform(-1, 1, n_samples)
    X2 = np.random.uniform(-1, 1, n_samples)

    if dependency_type == "multiplicative":
        Y = X1 * X2 + np.random.normal(0, 0.05, n_samples)

    elif dependency_type == 'type2':
        ...
    elif dependency_type == 'type3':
        ...
    elif dependency_type == 'type4':
        ...
    elif dependency_type == 'type5':
        ...
        
        
        
    fillers = {
        f'X{i}': np.random.normal(0, 1, n_samples) for i in range(3, 8)
    }

    # Create DataFrame
    df = pd.DataFrame({'X1': X1, 'X2': X2, **fillers, 'Y': Y})
    return df



# dep_df = generate_dependent_dataset('multiplicative')

# dep_df.to_csv('dependent_dataset_multiplicative.csv', index=False)

red_df = generate_redundant_dataset('trigonometric')

red_df.to_csv('redundant_dataset_linear.csv', index=False)
