import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_selection import mutual_info_regression
import shap
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from matplotlib.colors import LinearSegmentedColormap

def calculate_shap_interactions(X, y, model):
    # Fit the model with actual target
    model.fit(X, y)
    
    # Calculate SHAP interaction values
    explainer = shap.TreeExplainer(model)
    shap_interaction_values = explainer.shap_interaction_values(X)
    
    # Average the absolute SHAP interaction values across all samples
    mean_interactions = np.abs(shap_interaction_values).mean(0)
    
    return mean_interactions

def main():
    parser = argparse.ArgumentParser(description="Calculate Mutual Information between features.")
    parser.add_argument('--csv', type=str, required=True, help='Path to input CSV file')
    # parser.add_argument('--target', type=str, required=False, help='Name of the target column', default='Y')
    parser.add_argument('--model', type=str, required=False, help='(Ignored) Placeholder for model name')

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv)
    X = df[[col for col in df.columns if col.startswith('X')]]
    
    # y is the last column
    y = df[df.columns[-1]]
    
    # Calculate correlation matrix
    corr_matrix = X.corr()
    print("\nCorrelation Matrix:")
    print(corr_matrix.round(4))
    
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
    
    # Print the matrices
    print("\nMutual Information Matrix:")
    print(mi_df.round(4))
    
    # Calculate SHAP interaction values
    # model = RandomForestRegressor(n_estimators=100, random_state=42)
    model = XGBRegressor(n_estimators=100, random_state=42)
    shap_interactions = calculate_shap_interactions(X, y, model)
    
    # Create DataFrame for SHAP interactions
    shap_df = pd.DataFrame(shap_interactions, columns=X.columns, index=X.columns)
    
    print("\nSHAP Interaction Values Matrix:")
    print(shap_df.round(4))
    
    # Print value ranges for reference
    print("\nValue Ranges:")
    print(f"MI values range: {mi_df.values.min():.4f} to {mi_df.values.max():.4f}")
    print(f"SHAP values range: {shap_df.values.min():.4f} to {shap_df.values.max():.4f}")
    
    # Set diagonal to 0 for MI and correlation matrices
    np.fill_diagonal(mi_matrix, 0)
    np.fill_diagonal(corr_matrix.values, 0)
    
    # Create figure with a 2x2 grid
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Create three subplots in the 2x2 grid
    ax1 = fig.add_subplot(gs[0, 0])  # Correlation plot (top-left)
    ax2 = fig.add_subplot(gs[0, 1])  # MI plot (top-right)
    ax3 = fig.add_subplot(gs[1, 0])  # SHAP plot (bottom-left)
    
    # Custom color maps
    # corr_colors = sns.diverging_palette(140, 140, s=85, l=30, n=256, center="dark")  # Green-Gray-Green
    # corr_colors from light blue to white to light green
    corr_colors = LinearSegmentedColormap.from_list(
        "blue_white_green",
        ["#add8e6", "#ffffff", "#90ee90"],  # light blue → white → light green
        N=256
    )
    mi_colors = sns.light_palette("blue", n_colors=256)  # Gray to blue
    shap_colors = sns.light_palette("yellow", n_colors=256)  # Gray to Yellow
    
    # Calculate significance thresholds for MI and SHAP
    mi_threshold = 0.05  # Values below this are considered insignificant for MI
    shap_threshold = np.percentile(np.abs(shap_df.values), 80)  # Using 25th percentile as minimum threshold
    
    # Plot correlation heatmap
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap=corr_colors, 
                fmt='.4f', 
                ax=ax1, 
                center=0,
                mask=np.eye(len(corr_matrix)),  # Mask diagonal
                )
    ax1.set_title('Feature Correlation Matrix', pad=20)
    
    # Plot MI heatmap with significance threshold
    sns.heatmap(mi_df, 
                annot=True, 
                cmap=mi_colors, 
                fmt='.4f', 
                ax=ax2,
                mask=np.eye(len(mi_df)),  # Mask diagonal
                vmin=mi_threshold,  # Values below this will be gray
                vmax=max(mi_threshold + 0.1, mi_df.values.max())  # Ensure some spread in colors
                )
    ax2.set_title(f'Feature-to-Feature Mutual Information)', pad=20)
    
    # Plot SHAP interactions heatmap with significance threshold
    sns.heatmap(shap_df, 
                annot=True, 
                cmap=shap_colors, 
                fmt='.4f', 
                ax=ax3,
                vmin=shap_threshold,  # Values below this will be gray
                vmax=max(shap_df.values.max(), shap_threshold * 2)  # Ensure good spread of colors
                )
    ax3.set_title(f'SHAP Feature Interaction Values)', pad=20)
    
    plt.show()

if __name__ == "__main__":
    main()
