import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import mutual_info_regression

def main():
    parser = argparse.ArgumentParser(description="Calculate Mutual Information between features.")
    parser.add_argument('--csv', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--model', type=str, required=False, help='(Ignored) Placeholder for model name')

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv)
    X = df[[col for col in df.columns if col.startswith('X')]]
    
    # Calculate mutual information between all features
    n_features = len(X.columns)
    mi_matrix = np.zeros((n_features, n_features))
    
    for i, col1 in enumerate(X.columns):
        for j, col2 in enumerate(X.columns):
            if i != j:
                mi_matrix[i, j] = mutual_info_regression(X[[col1]], X[col2], random_state=42)[0]
            else:
                mi_matrix[i, j] = 1.0  # Self-information is 1
    
    # Create DataFrame for better visualization
    mi_df = pd.DataFrame(mi_matrix, columns=X.columns, index=X.columns)
    
    # Print the matrix
    print("\nMutual Information Matrix:")
    print(mi_df.round(4))
    
    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(mi_df, annot=True, cmap='YlOrRd', fmt='.4f')
    plt.title('Feature-to-Feature Mutual Information')
    plt.show()

if __name__ == "__main__":
    main()
