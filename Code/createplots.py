import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from itertools import combinations, permutations
import matplotlib.patches as mpatches
from sklearn.metrics import mean_squared_error
import random
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# import nbformat
import networkx as nx
import argparse
from datetime import datetime
from tqdm import tqdm
import os

def analyze_feature_impacts(df, feature_list, target='Y', model_type='rf', poly_degree=2):
    """
    Evaluates all feature combinations and analyzes incremental impact of each feature addition using different models.
    
    Parameters:
    - df: DataFrame containing the dataset
    - feature_list: List of feature names to consider
    - target: The target variable (default: 'Y')
    - model_type: The model to use ('linear', 'poly', 'rf', 'mlp')
    - poly_degree: Degree for polynomial regression (default: 2)
    
    Returns:
    - combinations_df: DataFrame containing MSE for each feature combination
    - impact_df: DataFrame analyzing feature importance through MSE reduction
    """
    
    # Calculate total number of combinations
    total_combinations = sum(len(list(combinations(feature_list, r))) for r in range(1, len(feature_list) + 1))
    
    # First get all combinations and their MSEs
    combinations_results = []
    with tqdm(total=total_combinations, desc="Analyzing feature combinations") as pbar:
        for r in range(1, len(feature_list) + 1):
            for feature_combo in combinations(feature_list, r):
                X = df[list(feature_combo)]
                y = df[target]
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                
                # Choose model
                if model_type == 'linear':
                    model = LinearRegression()
                elif model_type == 'poly':
                    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                    X_train = poly.fit_transform(X_train)
                    X_test = poly.transform(X_test)
                    model = LinearRegression()
                elif model_type == 'rf':
                    model = RandomForestRegressor(n_estimators=300, random_state=43 ,max_depth=10)
                elif model_type == 'mlp':
                    model = MLPRegressor(hidden_layer_sizes=(20), activation='relu', max_iter=500, random_state=42)
                else:
                    raise ValueError("Invalid model_type. Choose from 'linear', 'poly', 'rf', or 'mlp'.")

                # Train and evaluate
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                
                combinations_results.append({
                    'features': feature_combo,
                    'num_features': len(feature_combo),
                    'mse': mse,
                    'rmse': np.sqrt(mse)
                })
                pbar.update(1)
    
    # Create impact analysis
    impact_results = []
    with tqdm(total=len(combinations_results), desc="Calculating feature impacts") as pbar:
        for result in combinations_results:
            if len(result['features']) > 1:
                current_mse = result['mse']
                current_features = set(result['features'])
                
                # Find the MSE for each subset with one feature removed
                for feature in result['features']:
                    subset_features = current_features - {feature}
                    
                    # Find the corresponding base MSE
                    base_mse = next(r['mse'] for r in combinations_results 
                                  if set(r['features']) == subset_features)
                    
                    impact_results.append({
                        'base_features': tuple(sorted(subset_features)),
                        'added_feature': feature,
                        'final_features': result['features'],
                        'base_mse': base_mse,
                        'new_mse': current_mse,
                        'mse_reduction': base_mse - current_mse,
                        'relative_improvement': ((base_mse - current_mse) / base_mse) * 100
                    })
            pbar.update(1)
    
    # Convert to DataFrame and sort by MSE reduction
    impact_df = pd.DataFrame(impact_results).sort_values('mse_reduction', ascending=False)
    
    # Convert combinations results to DataFrame
    combinations_df = pd.DataFrame(combinations_results).sort_values('mse')
    
    return combinations_df, impact_df



L_score_dict = {}

# Function to compute PCA metrics with orientation validation
def compute_pca_metrics(data):
    if len(data) < 3:
        return 1, 0, 0, False

    pca = PCA(n_components=2)
    pca.fit(data)

    explained_variance = pca.explained_variance_ratio_
    if explained_variance[1] == 0:
        skinniness = 1  # or 0, but 1 is safe linear fallback
    else:
        skinniness = explained_variance[0] / explained_variance[1]
        skinniness = skinniness / (1 + skinniness)

    first_component = pca.components_[0]
    angle = np.degrees(np.arctan2(first_component[1], first_component[0]))
    horizontalness = np.cos(2 * np.radians(angle))

    centroid = np.mean(data, axis=0)
    # print(centroid)
    # print(horizontalness)
    if horizontalness > 0:
        correct_orientation = centroid[0] > 0
    elif horizontalness < 0:
        correct_orientation = centroid[1] > 0
    else:
        correct_orientation = False

    if np.any(np.isnan([skinniness, horizontalness, angle])):
        return 1, 0, 0, False

    return skinniness, horizontalness, angle, correct_orientation

def create_plot(impact_df, csv_filename):
    # Extract base filename without path and extension
    base_name = csv_filename.split('/')[-1].replace('.csv', '')
    
    # Create main directory for this CSV
    main_dir = f'outputs/{base_name}'
    pairs_dir = f'{main_dir}/pairs'
    
    # Create directories if they don't exist
    os.makedirs(main_dir, exist_ok=True)
    os.makedirs(pairs_dir, exist_ok=True)
    
    features = np.sort(impact_df['added_feature'].unique())
    feature_pairs = [(f1, f2) for i, f1 in enumerate(features) for f2 in features[i+1:]]

    global_min = impact_df['mse_reduction'].min()
    global_max = impact_df['mse_reduction'].max()

    # Create the main figure with all pairs
    fig, axes = plt.subplots(nrows=len(feature_pairs)//2 + 1, ncols=2, figsize=(12, 5 * (len(feature_pairs)//2 + 1)))

    # Initialize L-score matrix
    n_features = len(features)
    L_score_matrix = np.zeros((n_features, n_features))
    feature_to_idx = {feature: idx for idx, feature in enumerate(features)}

    for ax, (feature1, feature2) in zip(axes.flatten(), feature_pairs):
        points = []

        for row in impact_df.itertuples(index=False):
            if feature1 in row.added_feature and feature2 not in row.base_features:
                for row2 in impact_df.itertuples(index=False):
                    if set(row.final_features) <= set(row2.base_features) and feature2 in row2.added_feature:
                        points.append((row.mse_reduction, row2.mse_reduction, 'red'))
            elif feature2 in row.added_feature and feature1 not in row.base_features:
                for row2 in impact_df.itertuples(index=False):
                    if set(row.final_features) <= set(row2.base_features) and feature1 in row2.added_feature:
                        points.append((row2.mse_reduction, row.mse_reduction, 'blue'))
            elif set([row.base_features]) == {feature1}:
                points.append((row.mse_reduction, 0, 'red'))
            elif set([row.base_features]) == {feature2}:
                points.append((0, row.mse_reduction, 'blue'))

        if points:
            red_points = [p for p in points if p[2] == 'red']
            blue_points = [p for p in points if p[2] == 'blue']

            red_points = random.sample(red_points, min(len(red_points), 300))
            blue_points = random.sample(blue_points, min(len(blue_points), 300))

            points = red_points + blue_points
            x_vals, y_vals, colors = zip(*points)
            x_vals, y_vals = np.array(x_vals), np.array(y_vals)

            red_array = np.array([(x, y) for x, y, c in points if c == 'red'])
            blue_array = np.array([(x, y) for x, y, c in points if c == 'blue'])

            red_skinny, red_horiz, red_angle, red_ok = compute_pca_metrics(red_array)
            blue_skinny, blue_horiz, blue_angle, blue_ok = compute_pca_metrics(blue_array)

            if red_ok and blue_ok:
                L_score = (red_skinny * blue_skinny) * (red_horiz - blue_horiz) / 2
                L_score = round(L_score, 3)
                # Store L-score in matrix
                idx1, idx2 = feature_to_idx[feature1], feature_to_idx[feature2]
                L_score_matrix[idx1, idx2] = L_score
                L_score_matrix[idx2, idx1] = L_score  # Make matrix symmetric
            else:
                L_score = None
        else:
            x_vals, y_vals, colors = [], [], []
            L_score = None

        ax.scatter(x_vals, y_vals, c=colors, alpha=0.5)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)

        if L_score is not None:
            ax.set_title(
                f'{feature1} vs {feature2}\n'
                f'Score: {L_score}\n'
                f'Red skinny: {round(red_skinny, 2)}, angle: {round(np.deg2rad(red_angle), 1)}°\n'
                f'Blue skinny: {round(blue_skinny, 2)}, angle: {round(np.deg2rad(blue_angle), 1)}°',
                fontsize=10
            )
            L_score_dict[(feature1, feature2)] = L_score
        else:
            ax.set_title(f'{feature1} vs {feature2} | No Valid Data', fontsize=10)

        ax.set_xlabel(f'{feature1} MSE Reduction', fontsize=10)
        ax.set_ylabel(f'{feature2} MSE Reduction', fontsize=10)
        red_patch = mpatches.Patch(color='red', label=f'{feature1} first')
        blue_patch = mpatches.Patch(color='blue', label=f'{feature2} first')
        ax.legend(handles=[red_patch, blue_patch], loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Create individual pair plot
        pair_fig, pair_ax = plt.subplots(figsize=(8, 8))
        pair_ax.scatter(x_vals, y_vals, c=colors, alpha=0.5)
        pair_ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        pair_ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        pair_ax.set_xlim(global_min, global_max)
        pair_ax.set_ylim(global_min, global_max)
        
        # Set the same title and formatting
        if L_score is not None:
            pair_ax.set_title(
                f'{feature1} vs {feature2}\n'
                f'Score: {L_score}\n'
                f'Red skinny: {round(red_skinny, 2)}, angle: {round(np.deg2rad(red_angle), 1)}°\n'
                f'Blue skinny: {round(blue_skinny, 2)}, angle: {round(np.deg2rad(blue_angle), 1)}°',
                fontsize=10
            )
        else:
            pair_ax.set_title(f'{feature1} vs {feature2} | No Valid Data', fontsize=10)

        pair_ax.set_xlabel(f'{feature1} MSE Reduction', fontsize=10)
        pair_ax.set_ylabel(f'{feature2} MSE Reduction', fontsize=10)
        red_patch = mpatches.Patch(color='red', label=f'{feature1} first')
        blue_patch = mpatches.Patch(color='blue', label=f'{feature2} first')
        pair_ax.legend(handles=[red_patch, blue_patch], loc='upper right')
        pair_ax.grid(True, alpha=0.3)
        
        # Save individual pair plot
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pair_filename = f'{pairs_dir}/{feature1}_vs_{feature2}_{current_time}.png'
        pair_fig.tight_layout()
        pair_fig.savefig(pair_filename)
        plt.close(pair_fig)  # Close the individual pair figure
        
    # Save the main figure with all pairs
    plt.tight_layout()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    main_filename = f'{main_dir}/all_pairs_{current_time}.png'
    fig.savefig(main_filename)
    plt.close(fig)  # Close the main figure
    
    # Create and save L-score matrix heatmap
    plt.figure(figsize=(12, 10))
    # Create custom colormap from green (-1) through white (0) to blue (1)
    cmap = sns.diverging_palette(145, 10, as_cmap=True)
    
    # Create heatmap
    sns.heatmap(L_score_matrix, 
                xticklabels=features,
                yticklabels=features,
                cmap=cmap,
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'L-Score'})
    
    plt.title('L-Score Matrix')
    plt.tight_layout()
    
    # Save L-score matrix
    matrix_filename = f'{main_dir}/L_score_matrix_{current_time}.png'
    plt.savefig(matrix_filename)
    plt.close()
    
    # Save L-score matrix as CSV
    matrix_df = pd.DataFrame(L_score_matrix, index=features, columns=features)
    matrix_df.to_csv(f'{main_dir}/L_score_matrix_{current_time}.csv')
    
    print(f"Individual pair plots saved to: {pairs_dir}")
    print(f"Combined plot saved to: {main_filename}")
    print(f"L-score matrix saved to: {matrix_filename}")
    print(f"L-score matrix data saved to: {main_dir}/L_score_matrix_{current_time}.csv")

# use argparse to get the csv file
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--poly_degree", type=int, required=False, default=2)
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.csv_file)
    features_to_test = df.columns.tolist()[:-1]
    
    
    print("Analyzing feature impacts...")
    combinations_df, impact_df = analyze_feature_impacts(df, features_to_test, model_type=args.model_type, poly_degree=args.poly_degree)
    print("Creating plots...")
    create_plot(impact_df, args.csv_file)