"""
Feature Analysis Tool combining exhaustive combinations and random walks approaches.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from tqdm import tqdm
import os
from datetime import datetime
import argparse
import json
import random

from optimizer import RandomOrderOptimizer

# Add plotly import for 3D plotting
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. 3D plots will be skipped. Install with: pip install plotly")


class FeatureAnalyzer:
    """Main class for feature analysis using different approaches and L-score methods."""
    
    def __init__(self, approach='combos', l_score_method='AS', model_type='rf', 
                 poly_degree=2, n_paths=50, random_seed=42, output_dir=None):
        """
        Initialize the Feature Analyzer.
        
        Args:
            approach: 'combos' for exhaustive combinations, 'walks' for random walks
            l_score_method: 'AS' for PCA-based, 'CDM' for pattern-based
            model_type: 'linear', 'poly', 'rf', 'mlp', 'gam'
            poly_degree: Degree for polynomial regression
            n_paths: Number of random paths (for walks approach)
            random_seed: Random seed for reproducibility
            output_dir: Custom output directory (if None, creates timestamped default)
        """
        self.approach = approach
        self.l_score_method = l_score_method
        self.model_type = model_type
        self.poly_degree = poly_degree
        self.n_paths = n_paths
        self.random_seed = random_seed
        
        # Create output directory
        if output_dir is None:
            # Create timestamped output directory for this run
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.output_dir = f'outputs/run_{approach}_{l_score_method}_{model_type}_{current_time}'
        else:
            self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize model parameters
        self.model_params = self._get_model_params()
        
        # Initialize optimizer for random walks approach
        if approach == 'walks':
            self.optimizer = RandomOrderOptimizer(
                base_model_class=self._get_model_class(),
                model_params=self.model_params,
                n_total_paths=n_paths
            )
    
    def _get_model_class(self):
        """Get the model class based on model_type."""
        if self.model_type == 'rf':
            return RandomForestRegressor
        elif self.model_type == 'linear':
            return LinearRegression
        elif self.model_type == 'mlp':
            return MLPRegressor
        elif self.model_type == 'gam':
            try:
                from pygam import LinearGAM
                return LinearGAM
            except ImportError:
                raise ImportError("pygam is required for GAM models. Install with: pip install pygam")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _get_model_params(self):
        """Get model parameters based on model_type."""
        if self.model_type == 'rf':
            return {'n_estimators': 300, 'random_state': 43, 'max_depth': 10}
        elif self.model_type == 'mlp':
            return {'hidden_layer_sizes': (40, 30, 20), 'activation': 'tanh', 
                   'max_iter': 500, 'random_state': 42, 'early_stopping': True, 'validation_fraction': 0.2,
                   'alpha': 0.01}
        elif self.model_type == 'gam':
            return {'n_splines': 10, 'spline_order': 3, 'lam': 0.6}
        else:
            return {}
    
    def analyze_features(self, df, target='Y'):
        """
        Main method to analyze features using the specified approach.
        
        Args:
            df: DataFrame containing the dataset
            target: Target column name
            
        Returns:
            Dictionary containing analysis results
        """
        feature_list = [col for col in df.columns if col != target]
        
        if self.approach == 'combos':
            return self._analyze_with_combinations(df, feature_list, target)
        elif self.approach == 'walks':
            return self._analyze_with_random_walks(df, feature_list, target)
        else:
            raise ValueError(f"Unknown approach: {self.approach}")
    
    def _analyze_with_combinations(self, df, feature_list, target):
        """Analyze using exhaustive combinations approach (AS method)."""
        print("Using exhaustive combinations approach...")
        
        # Get all combinations and their impacts
        combinations_df, impact_df = self._analyze_feature_impacts_combos(
            df, feature_list, target
        )
        
        # Calculate L-scores based on method
        if self.l_score_method == 'AS':
            l_scores = self._calculate_l_scores_AS_method(impact_df, feature_list)
            detailed_metrics = {}  # AS method doesn't return detailed metrics yet
        else:
            # For CDM method, we need to convert combo data to path-like format
            l_scores, detailed_metrics = self._calculate_l_scores_CDM_from_combos(
                combinations_df, feature_list
            )
        
        return {
            'approach': 'combinations',
            'combinations_df': combinations_df,
            'impact_df': impact_df,
            'l_scores': l_scores,
            'feature_list': feature_list,
            'detailed_metrics': detailed_metrics
        }
    
    def _analyze_with_random_walks(self, df, feature_list, target):
        """Analyze using random walks approach (CDM method)."""
        print("Using random walks approach...")
        
        # Prepare data
        X = df[feature_list]
        y = df[target]
        
        # Scale data
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X), 
            columns=X.columns
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=self.random_seed
        )
        
        # Generate random paths
        print(f"Generating {self.n_paths} random paths...")
        all_paths_data = self.optimizer.generate_paths_only(
            X_train, X_test, y_train, y_test, feature_list
        )
        
        # Calculate L-scores
        if self.l_score_method == 'CDM':
            l_scores, detailed_metrics = self._calculate_l_scores_CDM_method(all_paths_data, feature_list)
        else:
            # Convert paths to AS format for comparison
            l_scores = self._calculate_l_scores_AS_from_paths(
                all_paths_data, feature_list
            )
            detailed_metrics = {}  # AS method doesn't return detailed metrics yet
        
        return {
            'approach': 'random_walks',
            'paths_data': all_paths_data,
            'l_scores': l_scores,
            'feature_list': feature_list,
            'detailed_metrics': detailed_metrics
        }
    
    def _analyze_feature_impacts_combos(self, df, feature_list, target):
        """AS original combination analysis method."""
        total_combinations = sum(len(list(combinations(feature_list, r))) 
                               for r in range(1, len(feature_list) + 1))
        
        combinations_results = []
        with tqdm(total=total_combinations, desc="Analyzing feature combinations") as pbar:
            for r in range(1, len(feature_list) + 1):
                for feature_combo in combinations(feature_list, r):
                    X = df[list(feature_combo)]
                    y = df[target]
                    
                    # scaler = StandardScaler()
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=0.2, random_state=42
                    )
                    
                    model = self._get_trained_model(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    
                    combinations_results.append({
                        'features': feature_combo,
                        'num_features': len(feature_combo),
                        'mse': mse,
                        'rmse': np.sqrt(mse)
                    })
                    pbar.update(1)
        
        combinations_df = pd.DataFrame(combinations_results).sort_values('mse')
        impact_df = self._calculate_impact_from_combinations(combinations_results)
        
        return combinations_df, impact_df
    
    def _get_trained_model(self, X_train, y_train):
        """Get and train a model based on model_type."""
        if self.model_type == 'linear':
            model = LinearRegression()
        elif self.model_type == 'poly':
            poly = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
            X_train = poly.fit_transform(X_train)
            model = LinearRegression()
        elif self.model_type == 'rf':
            model = RandomForestRegressor(**self.model_params)
        elif self.model_type == 'mlp':
            model = MLPRegressor(**self.model_params)
        elif self.model_type == 'gam':
            try:
                from pygam import LinearGAM, s
                # Create smooth terms for each feature
                n_features = X_train.shape[1]
                if n_features == 1:
                    terms = s(0, n_splines=self.model_params['n_splines'], 
                             spline_order=self.model_params['spline_order'])
                else:
                    terms = sum([s(i, n_splines=self.model_params['n_splines'], 
                                   spline_order=self.model_params['spline_order']) 
                                for i in range(n_features)])
                
                model = LinearGAM(terms, lam=self.model_params['lam'])
            except ImportError:
                raise ImportError("pygam is required for GAM models. Install with: pip install pygam")
        else:
            raise ValueError(f"Invalid model_type: {self.model_type}")
        
        model.fit(X_train, y_train)
        return model
    
    def _convert_combinations_to_impacts(self, combinations_df):
        """Convert combinations data to impact format. Used by both AS and CDM methods."""
        impact_results = []
        
        for result in combinations_df.itertuples():
            if len(result.features) > 1:
                current_mse = result.mse
                current_features = set(result.features)
                
                for feature in result.features:
                    subset_features = current_features - {feature}
                    
                    # Find the corresponding base MSE
                    base_result = combinations_df[combinations_df['features'].apply(
                        lambda x: set(x) == subset_features
                    )]
                    
                    if not base_result.empty:
                        base_mse = base_result.iloc[0]['mse']
                        impact_results.append({
                            'base_features': tuple(sorted(subset_features)),
                            'added_feature': feature,
                            'final_features': result.features,
                            'base_mse': base_mse,
                            'new_mse': current_mse,
                            'mse_reduction': base_mse - current_mse,
                            'relative_improvement': ((base_mse - current_mse) / base_mse) * 100
                        })
        
        return pd.DataFrame(impact_results).sort_values('mse_reduction', ascending=False)

    def _calculate_impact_from_combinations(self, combinations_results):
        """Calculate feature impacts from combination results."""
        combinations_df = pd.DataFrame(combinations_results)
        return self._convert_combinations_to_impacts(combinations_df)
    
    def _initialize_l_score_matrix(self, feature_list):
        """Initialize L-score matrix and feature index mapping."""
        n_features = len(feature_list)
        L_score_matrix = np.zeros((n_features, n_features))
        feature_to_idx = {feature: idx for idx, feature in enumerate(feature_list)}
        return L_score_matrix, feature_to_idx, n_features

    def _calculate_l_scores_AS_method(self, impact_df, feature_list):
        """Calculate L-scores using AS PCA-based method."""
        # Import the AS PCA metrics function
        import sys
        sys.path.append('.')
        from createplots_cdm import compute_pca_metrics
        
        features = np.sort(impact_df['added_feature'].unique())
        feature_pairs = [(f1, f2) for i, f1 in enumerate(features) 
                        for f2 in features[i+1:]]
        
        L_score_matrix, feature_to_idx, n_features = self._initialize_l_score_matrix(features)
        
        for feature1, feature2 in feature_pairs:
            points = self._extract_points_for_pair(impact_df, feature1, feature2)
            
            if points:
                red_points = np.array([(x, y) for x, y, c in points if c == 'red'])
                blue_points = np.array([(x, y) for x, y, c in points if c == 'blue'])
                
                if len(red_points) >= 3 and len(blue_points) >= 3:
                    # Add synthetic points at origin for AS method (matching CDM approach)
                    red_synthetic = np.zeros((len(red_points), 2))
                    blue_synthetic = np.zeros((len(blue_points), 2))
                    
                    red_augmented = np.vstack([red_points, red_synthetic])
                    blue_augmented = np.vstack([blue_points, blue_synthetic])
                    
                    red_skinny, red_horiz, _, red_ok = compute_pca_metrics(red_augmented)
                    blue_skinny, blue_horiz, _, blue_ok = compute_pca_metrics(blue_augmented)
                    
                    if red_ok and blue_ok:
                        L_score = (red_skinny * blue_skinny) * (red_horiz - blue_horiz) / 2
                        print(f"AS {feature1} vs {feature2}: red_skinny={red_skinny:.4f}, red_horiz={red_horiz:.4f}, blue_skinny={blue_skinny:.4f}, blue_horiz={blue_horiz:.4f}, L_score={L_score:.4f}")
                        idx1, idx2 = feature_to_idx[feature1], feature_to_idx[feature2]
                        L_score_matrix[idx1, idx2] = L_score
                        L_score_matrix[idx2, idx1] = L_score
        
        return L_score_matrix
    
    def _extract_points_for_pair(self, impact_df, feature1, feature2):
        """Extract points for a feature pair (AS method)."""
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
        
        return points
    
    def create_visualizations(self, results, csv_filename):
        """Create visualizations based on the analysis results."""
        # Save run configuration info to the timestamped folder
        base_name = csv_filename.split('/')[-1].replace('.csv', '')
        config_info = {
            'dataset': base_name,
            'approach': self.approach,
            'l_score_method': self.l_score_method,
            'model_type': self.model_type,
            'poly_degree': self.poly_degree,
            'n_paths': self.n_paths,
            'random_seed': self.random_seed,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save configuration as JSON
        with open(f'{self.output_dir}/run_config.json', 'w') as f:
            json.dump(config_info, f, indent=2)
        
        if results['approach'] == 'combinations':
            self._create_combination_plots(results, self.output_dir)
        else:
            self._create_walks_plots(results, self.output_dir)
        
        # Create L-score matrix heatmap (common for both approaches)
        self._create_l_score_heatmap(results['l_scores'], results['feature_list'], self.output_dir)
        
        # Create 3D scatter plot for feature triplets
        if results['approach'] == 'combinations':
            # For combinations approach, use impact_df
            impact_df = results['impact_df']
        else:
            # For walks approach, convert paths data to impact format
            all_paths_data = results['paths_data']
            impact_results = []
            for path_data in all_paths_data:
                path_features = path_data['order']
                metrics = path_data['metrics']
                improvements = -np.diff(metrics)  # Convert to improvements
                
                # For each consecutive pair of features in the path
                for i in range(len(path_features) - 1):
                    if i < len(improvements):
                        base_features = tuple(path_features[:i+1])
                        added_feature = path_features[i+1]
                        final_features = tuple(path_features[:i+2])
                        
                        # Simulated MSE values (we don't have actual MSE from paths)
                        base_mse = metrics[i]
                        new_mse = metrics[i+1]
                        mse_reduction = improvements[i]
                        
                        impact_results.append({
                            'base_features': base_features[:-1] if len(base_features) > 1 else (),
                            'added_feature': added_feature,
                            'final_features': final_features,
                            'base_mse': base_mse,
                            'new_mse': new_mse,
                            'mse_reduction': mse_reduction,
                            'relative_improvement': (mse_reduction / base_mse) * 100 if base_mse != 0 else 0
                        })
            
            if impact_results:
                impact_df = pd.DataFrame(impact_results)
            else:
                impact_df = None
        
        # Create 3D plot if we have impact data
        if impact_df is not None:
            self._create_3d_scatter_plot(impact_df, results['feature_list'], self.output_dir)
    
    def _create_l_score_heatmap(self, l_scores, feature_list, output_dir):
        """Create L-score matrix heatmap."""
        plt.figure(figsize=(12, 10))
        cmap = sns.diverging_palette(145, 10, as_cmap=True)
        
        sns.heatmap(l_scores, 
                    xticklabels=feature_list,
                    yticklabels=feature_list,
                    cmap=cmap,
                    center=0,
                    vmin=-1,
                    vmax=1,
                    square=True,
                    annot=True,
                    fmt='.2f',
                    cbar_kws={'label': 'L-Score'})
        
        plt.title(f'L-Score Matrix ({self.l_score_method} method, {self.approach} approach)')
        plt.tight_layout()
        
        # Save with simple filenames since the folder is already timestamped
        filename = f'{output_dir}/L_score_matrix.png'
        plt.savefig(filename)
        plt.close()
        
        # Save as CSV
        matrix_df = pd.DataFrame(l_scores, index=feature_list, columns=feature_list)
        csv_filename = f'{output_dir}/L_score_matrix.csv'
        matrix_df.to_csv(csv_filename)
        
        print(f"L-score matrix saved to: {filename}")
        print(f"L-score matrix data saved to: {csv_filename}")
        print(f"All outputs saved to: {output_dir}")
    
    def _create_combination_plots(self, results, output_dir):
        """Create plots for combination approach (AS plots)."""
        impact_df = results['impact_df']
        feature_list = results['feature_list']
        l_scores = results['l_scores']
        
        # Create pairs subdirectory
        pairs_dir = f'{output_dir}/pairs'
        os.makedirs(pairs_dir, exist_ok=True)
        
        # Create individual pair scatterplots and combined plot
        self._create_feature_pair_scatterplots(impact_df, feature_list, output_dir, pairs_dir, l_scores, results['detailed_metrics'])
        
        print(f"Individual pair plots saved to: {pairs_dir}")
        print(f"Combined plot saved to: {output_dir}")
    
    def _create_walks_plots(self, results, output_dir):
        """Create plots for random walks approach."""
        # Convert paths data to impact-like format for plotting
        all_paths_data = results['paths_data']
        feature_list = results['feature_list']
        l_scores = results['l_scores']
        
        # Convert to impact format
        impact_results = []
        for path_data in all_paths_data:
            path_features = path_data['order']
            metrics = path_data['metrics']
            improvements = -np.diff(metrics)  # Convert to improvements
            
            # For each consecutive pair of features in the path
            for i in range(len(path_features) - 1):
                if i < len(improvements):
                    base_features = tuple(path_features[:i+1])
                    added_feature = path_features[i+1]
                    final_features = tuple(path_features[:i+2])
                    
                    # Simulated MSE values (we don't have actual MSE from paths)
                    base_mse = metrics[i]
                    new_mse = metrics[i+1]
                    mse_reduction = improvements[i]
                    
                    impact_results.append({
                        'base_features': base_features[:-1] if len(base_features) > 1 else (),
                        'added_feature': added_feature,
                        'final_features': final_features,
                        'base_mse': base_mse,
                        'new_mse': new_mse,
                        'mse_reduction': mse_reduction,
                        'relative_improvement': (mse_reduction / base_mse) * 100 if base_mse != 0 else 0
                    })
        
        if impact_results:
            impact_df = pd.DataFrame(impact_results)
            
            # Create pairs subdirectory
            pairs_dir = f'{output_dir}/pairs'
            os.makedirs(pairs_dir, exist_ok=True)
            
            # Create scatterplots
            self._create_feature_pair_scatterplots(impact_df, feature_list, output_dir, pairs_dir, l_scores, results['detailed_metrics'])
            
            print(f"Individual pair plots saved to: {pairs_dir}")
            print(f"Combined plot saved to: {output_dir}")
        else:
            print(f"No impact data available for walks plots in: {output_dir}")
    
    def _create_feature_pair_scatterplots(self, impact_df, feature_list, output_dir, pairs_dir, l_scores, detailed_metrics=None):
        """Create scatterplots for each feature pair showing MSE reductions."""
        import matplotlib.patches as mpatches
        import random
        
        features = np.sort(impact_df['added_feature'].unique())
        feature_pairs = [(f1, f2) for i, f1 in enumerate(features) for f2 in features[i+1:]]
        
        if not feature_pairs:
            print("No feature pairs found for plotting.")
            return
        
        # Create feature to index mapping for L-score lookup
        feature_to_idx = {feature: idx for idx, feature in enumerate(feature_list)}
        
        global_min = impact_df['mse_reduction'].min()
        global_max = impact_df['mse_reduction'].max()
        
        # Create the main figure with all pairs
        n_pairs = len(feature_pairs)
        ncols = 2
        nrows = (n_pairs + ncols - 1) // ncols  # Ceiling division
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 5 * nrows))
        
        # Handle single subplot case
        if n_pairs == 1:
            axes = [axes]
        elif nrows == 1:
            axes = [axes] if ncols == 1 else axes
        else:
            axes = axes.flatten()
        
        for idx, (feature1, feature2) in enumerate(feature_pairs):
            ax = axes[idx] if idx < len(axes) else None
            if ax is None:
                continue
                
            points = self._extract_points_for_pair(impact_df, feature1, feature2)
            
            if points:
                # Limit points for performance (similar to original)
                red_points = [p for p in points if p[2] == 'red']
                blue_points = [p for p in points if p[2] == 'blue']
                
                red_points = random.sample(red_points, min(len(red_points), 300))
                blue_points = random.sample(blue_points, min(len(blue_points), 300))
                
                all_points = red_points + blue_points
                if all_points:
                    x_vals, y_vals, colors = zip(*all_points)
                    x_vals, y_vals = np.array(x_vals), np.array(y_vals)
                else:
                    x_vals, y_vals, colors = [], [], []
            else:
                x_vals, y_vals, colors = [], [], []
            
            # Get L-score for this pair
            if feature1 in feature_to_idx and feature2 in feature_to_idx:
                idx1, idx2 = feature_to_idx[feature1], feature_to_idx[feature2]
                l_score = l_scores[idx1, idx2]
                l_score_text = f"L-score: {l_score:.3f}" if l_score != 0 else "L-score: 0.000"
            else:
                l_score_text = "L-score: N/A"
            
            # Calculate detailed metrics for title
            detailed_title = self._calculate_detailed_metrics_for_title(
                points, feature1, feature2, l_score_text, detailed_metrics
            )
            
            # Create scatter plot
            if len(x_vals) > 0:
                ax.scatter(x_vals, y_vals, c=colors, alpha=0.5)
            
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
            ax.set_xlim(global_min, global_max)
            ax.set_ylim(global_min, global_max)
            
            ax.set_title(detailed_title, fontsize=9)
            ax.set_xlabel(f'{feature1} MSE Reduction', fontsize=10)
            ax.set_ylabel(f'{feature2} MSE Reduction', fontsize=10)
            
            # Add legend
            red_patch = mpatches.Patch(color='red', label=f'{feature1} first')
            blue_patch = mpatches.Patch(color='blue', label=f'{feature2} first')
            ax.legend(handles=[red_patch, blue_patch], loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Create individual pair plot
            self._create_individual_pair_plot(
                x_vals, y_vals, colors, feature1, feature2, 
                global_min, global_max, pairs_dir, detailed_title
            )
        
        # Hide any unused subplots
        for idx in range(len(feature_pairs), len(axes)):
            axes[idx].set_visible(False)
        
        # Save the main figure with all pairs
        plt.tight_layout()
        main_filename = f'{output_dir}/all_pairs_scatterplot.png'
        fig.savefig(main_filename)
        plt.close(fig)
        
        print(f"Combined scatterplot saved to: {main_filename}")
    
    def _calculate_detailed_metrics_for_title(self, points, feature1, feature2, l_score_text, detailed_metrics=None):
        """Calculate detailed skinnyness and horizontalness metrics for plot title."""
        # If we have stored detailed metrics, use those instead of recalculating
        pair_key = (feature1, feature2)
        reverse_pair_key = (feature2, feature1)
        
        if detailed_metrics and (pair_key in detailed_metrics or reverse_pair_key in detailed_metrics):
            # Use the stored metrics from L-score calculation
            if pair_key in detailed_metrics:
                metrics = detailed_metrics[pair_key]
                red_skinny = metrics['red_skinny']
                red_horiz = metrics['red_horiz']
                blue_skinny = metrics['blue_skinny']
                blue_horiz = metrics['blue_horiz']
            else:
                # Reverse pair - swap red and blue
                metrics = detailed_metrics[reverse_pair_key]
                red_skinny = metrics['blue_skinny']
                red_horiz = metrics['blue_horiz']
                blue_skinny = metrics['red_skinny']
                blue_horiz = metrics['red_horiz']
                
            return (f'{feature1} vs {feature2}\n{l_score_text}\n'
                   f'Red: S={red_skinny:.3f}, H={red_horiz:.3f}\n'
                   f'Blue: S={blue_skinny:.3f}, H={blue_horiz:.3f}')
        
        # Fallback to original calculation if no stored metrics
        if not points:
            return f'{feature1} vs {feature2}\n{l_score_text}\nNo data points'
        
        red_points = np.array([(x, y) for x, y, c in points if c == 'red'])
        blue_points = np.array([(x, y) for x, y, c in points if c == 'blue'])
        
        if len(red_points) < 3 or len(blue_points) < 3:
            return f'{feature1} vs {feature2}\n{l_score_text}\nInsufficient data'
        
        try:
            if self.l_score_method == 'AS':
                # Use AS method with synthetic points
                import sys
                sys.path.append('.')
                from createplots_cdm import compute_pca_metrics
                
                # Add synthetic points at origin
                red_synthetic = np.zeros((len(red_points), 2))
                blue_synthetic = np.zeros((len(blue_points), 2))
                
                red_augmented = np.vstack([red_points, red_synthetic])
                blue_augmented = np.vstack([blue_points, blue_synthetic])
                
                red_skinny, red_horiz, _, red_ok = compute_pca_metrics(red_augmented)
                blue_skinny, blue_horiz, _, blue_ok = compute_pca_metrics(blue_augmented)
                
                if red_ok and blue_ok:
                    return (f'{feature1} vs {feature2}\n{l_score_text}\n'
                           f'Red: S={red_skinny:.3f}, H={red_horiz:.3f}\n'
                           f'Blue: S={blue_skinny:.3f}, H={blue_horiz:.3f}')
                else:
                    return f'{feature1} vs {feature2}\n{l_score_text}\nPCA failed'
            
            else:  # CDM method
                red_skinny, red_horiz, red_centroid = self._get_cloud_characteristics(red_points, feature1, feature2, "red")
                blue_skinny, blue_horiz, blue_centroid = self._get_cloud_characteristics(blue_points, feature1, feature2, "blue")
                
                # Calculate separation factor for CDM enhancement (Cloud Radius Method)
                separation = np.linalg.norm(red_centroid - blue_centroid)
                
                # Calculate each cloud's radius (maximum distance from centroid)
                red_distances = np.linalg.norm(red_points - red_centroid, axis=1)
                blue_distances = np.linalg.norm(blue_points - blue_centroid, axis=1)
                red_radius = np.max(red_distances)
                blue_radius = np.max(blue_distances)
                
                # Separation factor: how far apart vs combined cloud sizes
                combined_radius = red_radius + blue_radius
                separation_factor = min(separation / combined_radius, 1.0) if combined_radius > 0 else 0
                
                # Calculate base L-score using CDM formula
                base_l_score = (red_skinny * blue_skinny) * (red_horiz - blue_horiz) / 2
                
                # Apply separation enhancement
                l_score = base_l_score * separation_factor
                
                print(f"CDM {feature1} vs {feature2}: red_skinny={red_skinny:.4f}, red_horiz={red_horiz:.4f}, blue_skinny={blue_skinny:.4f}, blue_horiz={blue_horiz:.4f}")
                print(f"  separation={separation:.4f}, red_radius={red_radius:.4f}, blue_radius={blue_radius:.4f}, sep_factor={separation_factor:.4f}")
                print(f"  base_L_score={base_l_score:.4f}, enhanced_L_score={l_score:.4f}")
                
                return (f'{feature1} vs {feature2}\n{l_score_text}\n'
                       f'Red: S={red_skinny:.3f}, H={red_horiz:.3f}\n'
                       f'Blue: S={blue_skinny:.3f}, H={blue_horiz:.3f}')
        
        except Exception as e:
            return f'{feature1} vs {feature2}\n{l_score_text}\nError: {str(e)[:20]}'
    
    def _create_individual_pair_plot(self, x_vals, y_vals, colors, feature1, feature2, 
                                   global_min, global_max, pairs_dir, l_score_text):
        """Create individual scatterplot for a feature pair."""
        import matplotlib.patches as mpatches
        
        pair_fig, pair_ax = plt.subplots(figsize=(8, 8))
        
        if len(x_vals) > 0:
            pair_ax.scatter(x_vals, y_vals, c=colors, alpha=0.5)
        
        pair_ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        pair_ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        pair_ax.set_xlim(global_min, global_max)
        pair_ax.set_ylim(global_min, global_max)
        
        pair_ax.set_title(f'{feature1} vs {feature2}\n{l_score_text}', fontsize=12)
        pair_ax.set_xlabel(f'{feature1} MSE Reduction', fontsize=12)
        pair_ax.set_ylabel(f'{feature2} MSE Reduction', fontsize=12)
        
        # Add legend
        red_patch = mpatches.Patch(color='red', label=f'{feature1} first')
        blue_patch = mpatches.Patch(color='blue', label=f'{feature2} first')
        pair_ax.legend(handles=[red_patch, blue_patch], loc='upper right')
        pair_ax.grid(True, alpha=0.3)
        
        # Save individual pair plot
        pair_filename = f'{pairs_dir}/{feature1}_vs_{feature2}.pdf'
        pair_fig.tight_layout()
        pair_fig.savefig(pair_filename, format='pdf', bbox_inches='tight')
        plt.close(pair_fig)
    
    def _calculate_l_scores_CDM_from_combos(self, combinations_df, feature_list):
        """Calculate L-scores using CDM method from combination data."""
        L_score_matrix, _, _ = self._initialize_l_score_matrix(feature_list)
        
        # Store detailed metrics for each pair
        detailed_metrics = {}
        
        # Convert combinations data to impact data using consolidated method
        impact_df = self._convert_combinations_to_impacts(combinations_df)
        
        # Calculate CDM L-scores for all pairs
        for i, feature1 in enumerate(feature_list):
            for j, feature2 in enumerate(feature_list):
                if i < j:  # Only calculate upper triangle, then make symmetric
                    # Extract points for this feature pair
                    points = self._extract_points_for_pair(impact_df, feature1, feature2)
                    
                    if points and len(points) > 4:  # Need some minimum points
                        # Extract red and blue points the same way as plot title calculation
                        red_points = np.array([(x, y) for x, y, c in points if c == 'red'])
                        blue_points = np.array([(x, y) for x, y, c in points if c == 'blue'])
                        
                        if len(red_points) >= 2 and len(blue_points) >= 2:
                            red_skinny, red_horizontalness, red_centroid = self._get_cloud_characteristics(red_points, feature1, feature2, "red")
                            blue_skinny, blue_horizontalness, blue_centroid = self._get_cloud_characteristics(blue_points, feature1, feature2, "blue")
                            
                            # Calculate separation factor for CDM enhancement (Cloud Radius Method)
                            separation = np.linalg.norm(red_centroid - blue_centroid)
                            
                            # Calculate each cloud's radius (maximum distance from centroid)
                            red_distances = np.linalg.norm(red_points - red_centroid, axis=1)
                            blue_distances = np.linalg.norm(blue_points - blue_centroid, axis=1)
                            red_radius = np.max(red_distances)
                            blue_radius = np.max(blue_distances)
                            
                            # Separation factor: how far apart vs combined cloud sizes
                            combined_radius = red_radius + blue_radius
                            separation_factor = min(separation / combined_radius, 1.0) if combined_radius > 0 else 0
                            
                            # Calculate base L-score using CDM formula
                            base_l_score = (red_skinny * blue_skinny) * (red_horizontalness - blue_horizontalness) / 2
                            
                            # Apply separation enhancement
                            l_score = base_l_score * separation_factor
                            
                            print(f"CDM {feature1} vs {feature2}: red_skinny={red_skinny:.4f}, red_horiz={red_horizontalness:.4f}, blue_skinny={blue_skinny:.4f}, blue_horiz={blue_horizontalness:.4f}")
                            print(f"  separation={separation:.4f}, red_radius={red_radius:.4f}, blue_radius={blue_radius:.4f}, sep_factor={separation_factor:.4f}")
                            print(f"  base_L_score={base_l_score:.4f}, enhanced_L_score={l_score:.4f}")
                            
                            # Store metrics for plotting
                            pair_key = (feature1, feature2)
                            detailed_metrics[pair_key] = {
                                'red_skinny': red_skinny,
                                'red_horiz': red_horizontalness,
                                'blue_skinny': blue_skinny,
                                'blue_horiz': blue_horizontalness,
                                'l_score': l_score,
                                'separation_factor': separation_factor,
                                'base_l_score': base_l_score
                            }
                            
                            # Store in matrix (symmetric)
                            L_score_matrix[i, j] = l_score
                            L_score_matrix[j, i] = l_score
        
        return L_score_matrix, detailed_metrics
    
    def _calculate_l_scores_AS_from_paths(self, all_paths_data, feature_list):
        """Calculate L-scores using AS method from random paths data."""
        L_score_matrix, _, _ = self._initialize_l_score_matrix(feature_list)
        
        # Convert path data to impact-like format for AS method
        impact_results = []
        
        for path_data in all_paths_data:
            path_features = path_data['order']
            metrics = path_data['metrics']
            improvements = -np.diff(metrics)  # Convert to improvements
            
            # For each consecutive pair of features in the path
            for i in range(len(path_features) - 1):
                if i < len(improvements):
                    base_features = tuple(path_features[:i+1])
                    added_feature = path_features[i+1]
                    final_features = tuple(path_features[:i+2])
                    
                    # Simulated MSE values (we don't have actual MSE from paths)
                    base_mse = metrics[i]
                    new_mse = metrics[i+1]
                    mse_reduction = improvements[i]
                    
                    impact_results.append({
                        'base_features': base_features[:-1] if len(base_features) > 1 else (),
                        'added_feature': added_feature,
                        'final_features': final_features,
                        'base_mse': base_mse,
                        'new_mse': new_mse,
                        'mse_reduction': mse_reduction,
                        'relative_improvement': (mse_reduction / base_mse) * 100 if base_mse != 0 else 0
                    })
        
        if not impact_results:
            return L_score_matrix
            
        impact_df = pd.DataFrame(impact_results)
        
        # Use the existing AS method calculation
        return self._calculate_l_scores_AS_method(impact_df, feature_list)

    def _calculate_l_scores_CDM_method(self, all_paths_data, feature_list):
        """Calculate L-scores using CDM pattern-based method from path data."""
        L_score_matrix, _, _ = self._initialize_l_score_matrix(feature_list)
        
        # Store detailed metrics for each pair
        detailed_metrics = {}
        
        # Convert paths data to impact format (same as used for plotting)
        impact_results = []
        for path_data in all_paths_data:
            path_features = path_data['order']
            metrics = path_data['metrics']
            improvements = -np.diff(metrics)  # Convert to improvements
            
            # For each consecutive pair of features in the path
            for i in range(len(path_features) - 1):
                if i < len(improvements):
                    base_features = tuple(path_features[:i+1])
                    added_feature = path_features[i+1]
                    final_features = tuple(path_features[:i+2])
                    
                    # Simulated MSE values (we don't have actual MSE from paths)
                    base_mse = metrics[i]
                    new_mse = metrics[i+1]
                    mse_reduction = improvements[i]
                    
                    impact_results.append({
                        'base_features': base_features[:-1] if len(base_features) > 1 else (),
                        'added_feature': added_feature,
                        'final_features': final_features,
                        'base_mse': base_mse,
                        'new_mse': new_mse,
                        'mse_reduction': mse_reduction,
                        'relative_improvement': (mse_reduction / base_mse) * 100 if base_mse != 0 else 0
                    })
        
        if not impact_results:
            return L_score_matrix, {}
            
        impact_df = pd.DataFrame(impact_results)
        
        # Now calculate CDM L-scores using the same impact_df data that plots use
        for i, feature1 in enumerate(feature_list):
            for j, feature2 in enumerate(feature_list):
                if i < j:  # Only calculate upper triangle, then make symmetric
                    # Extract points for this feature pair (same as plotting)
                    points = self._extract_points_for_pair(impact_df, feature1, feature2)
                    
                    if points and len(points) > 4:  # Need some minimum points
                        # Extract red and blue points the same way as plot title calculation
                        red_points = np.array([(x, y) for x, y, c in points if c == 'red'])
                        blue_points = np.array([(x, y) for x, y, c in points if c == 'blue'])
                        
                        if len(red_points) >= 2 and len(blue_points) >= 2:
                            red_skinny, red_horizontalness, red_centroid = self._get_cloud_characteristics(red_points, feature1, feature2, "red")
                            blue_skinny, blue_horizontalness, blue_centroid = self._get_cloud_characteristics(blue_points, feature1, feature2, "blue")
                            
                            # Calculate separation factor for CDM enhancement (Cloud Radius Method)
                            separation = np.linalg.norm(red_centroid - blue_centroid)
                            
                            # Calculate each cloud's radius (maximum distance from centroid)
                            red_distances = np.linalg.norm(red_points - red_centroid, axis=1)
                            blue_distances = np.linalg.norm(blue_points - blue_centroid, axis=1)
                            red_radius = np.max(red_distances)
                            blue_radius = np.max(blue_distances)
                            
                            # Separation factor: how far apart vs combined cloud sizes
                            combined_radius = red_radius + blue_radius
                            separation_factor = min(separation / combined_radius, 1.0) if combined_radius > 0 else 0
                            
                            # Calculate base L-score using CDM formula
                            base_l_score = (red_skinny * blue_skinny) * (red_horizontalness - blue_horizontalness) / 2
                            
                            # Apply separation enhancement
                            l_score = base_l_score * separation_factor
                            
                            print(f"CDM {feature1} vs {feature2}: red_skinny={red_skinny:.4f}, red_horiz={red_horizontalness:.4f}, blue_skinny={blue_skinny:.4f}, blue_horiz={blue_horizontalness:.4f}")
                            print(f"  separation={separation:.4f}, red_radius={red_radius:.4f}, blue_radius={blue_radius:.4f}, sep_factor={separation_factor:.4f}")
                            print(f"  base_L_score={base_l_score:.4f}, enhanced_L_score={l_score:.4f}")
                            
                            # Store metrics for plotting
                            pair_key = (feature1, feature2)
                            detailed_metrics[pair_key] = {
                                'red_skinny': red_skinny,
                                'red_horiz': red_horizontalness,
                                'blue_skinny': blue_skinny,
                                'blue_horiz': blue_horizontalness,
                                'l_score': l_score,
                                'separation_factor': separation_factor,
                                'base_l_score': base_l_score
                            }
                            
                            # Store in matrix (symmetric)
                            L_score_matrix[i, j] = l_score
                            L_score_matrix[j, i] = l_score
        
        return L_score_matrix, detailed_metrics
    
    def _get_cloud_characteristics(self, points, feature1=None, feature2=None, color=None):
        """Calculate cloud characteristics (skinnyness and horizontalness) for a set of points."""
        # Calculate centroid
        centroid = np.mean(points, axis=0)
        
        # Calculate angle from origin to centroid (remove % 360 to preserve sign information)
        centroid_angle = np.degrees(np.arctan2(centroid[1], centroid[0]))
        
        # Calculate horizontalness from centroid angle
        horizontalness = np.cos(2 * np.radians(centroid_angle))
        
        # Debug print statements with feature and color info
        feature_info = f"{feature1} vs {feature2}" if feature1 and feature2 else "Unknown features"
        color_info = f"{color} cloud" if color else "cloud"
        print(f"  DEBUG [{feature_info}] {color_info}: points mean: {centroid}, centroid_angle: {centroid_angle:.4f}°, horizontalness: {horizontalness:.4f}")
        
        # Add synthetic points at origin
        synthetic_points = np.zeros((len(points), 2))  # Create points at (0,0)
        augmented_points = np.vstack([points, synthetic_points])
        
        # Calculate skinnyness using augmented points with PCA
        pca = PCA()
        pca.fit(augmented_points)
        skinnyness = pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_[1]
        skinnyness = skinnyness / (1 + skinnyness)
        
        return skinnyness, horizontalness, centroid

    def _extract_points_for_triplet_correct(self, impact_df, feature1, feature2, feature3):
        """Extract 3D points where each point represents MSE reductions for all three features in the same scenario."""
        points = []
        
        # Find all scenarios where all three features are involved
        for row in impact_df.itertuples(index=False):
            final_features = set(row.final_features)
            
            # We want scenarios that include all three features
            if {feature1, feature2, feature3}.issubset(final_features):
                # This row represents adding one feature to a base containing the other features
                # We need to trace back to get individual MSE reductions for each feature
                
                added_feature = row.added_feature
                base_features = set(row.base_features) if row.base_features else set()
                
                # Initialize MSE reductions
                f1_reduction = f2_reduction = f3_reduction = 0
                
                # Get the reduction for the feature being added in this row
                if added_feature == feature1:
                    f1_reduction = row.mse_reduction
                elif added_feature == feature2:
                    f2_reduction = row.mse_reduction
                elif added_feature == feature3:
                    f3_reduction = row.mse_reduction
                
                # Now find the MSE reductions for the other features by looking at the base
                # The base should contain the other two features
                if len(base_features) >= 2:
                    # Find how we got to this base state
                    for base_row in impact_df.itertuples(index=False):
                        if set(base_row.final_features) == base_features:
                            base_added = base_row.added_feature
                            if base_added == feature1 and f1_reduction == 0:
                                f1_reduction = base_row.mse_reduction
                            elif base_added == feature2 and f2_reduction == 0:
                                f2_reduction = base_row.mse_reduction
                            elif base_added == feature3 and f3_reduction == 0:
                                f3_reduction = base_row.mse_reduction
                            
                            # If this base was built from a single feature, find that too
                            base_base_features = set(base_row.base_features) if base_row.base_features else set()
                            if len(base_base_features) == 1:
                                for single_row in impact_df.itertuples(index=False):
                                    if set(single_row.final_features) == base_base_features:
                                        single_added = single_row.added_feature
                                        if single_added == feature1 and f1_reduction == 0:
                                            f1_reduction = single_row.mse_reduction
                                        elif single_added == feature2 and f2_reduction == 0:
                                            f2_reduction = single_row.mse_reduction
                                        elif single_added == feature3 and f3_reduction == 0:
                                            f3_reduction = single_row.mse_reduction
                            elif len(base_base_features) == 0:
                                # The base was built from a single feature addition
                                single_added = base_row.added_feature
                                # Find the other feature's single addition
                                for single_row in impact_df.itertuples(index=False):
                                    if (len(single_row.base_features) == 0 and 
                                        single_row.added_feature in {feature1, feature2, feature3} and
                                        single_row.added_feature != single_added):
                                        other_added = single_row.added_feature
                                        if other_added == feature1 and f1_reduction == 0:
                                            f1_reduction = single_row.mse_reduction
                                        elif other_added == feature2 and f2_reduction == 0:
                                            f2_reduction = single_row.mse_reduction
                                        elif other_added == feature3 and f3_reduction == 0:
                                            f3_reduction = single_row.mse_reduction
                
                # Also check for single feature additions directly
                for single_row in impact_df.itertuples(index=False):
                    if len(single_row.base_features) == 0:  # Single feature addition
                        single_added = single_row.added_feature
                        if single_added == feature1 and f1_reduction == 0:
                            f1_reduction = single_row.mse_reduction
                        elif single_added == feature2 and f2_reduction == 0:
                            f2_reduction = single_row.mse_reduction
                        elif single_added == feature3 and f3_reduction == 0:
                            f3_reduction = single_row.mse_reduction
                
                # Only add point if we have reductions for all three features
                if f1_reduction != 0 and f2_reduction != 0 and f3_reduction != 0:
                    # Determine color based on which feature was added last (in this row)
                    if added_feature == feature1:
                        color = 'red'
                    elif added_feature == feature2:
                        color = 'blue'
                    else:
                        color = 'green'
                    
                    points.append((f1_reduction, f2_reduction, f3_reduction, color))
        
        return points

    def _create_3d_scatter_plot(self, impact_df, feature_list, output_dir):
        """Create a true 3D scatter plot where each point has MSE reductions for all three features."""
        if not PLOTLY_AVAILABLE:
            print("Skipping 3D plot - plotly not available")
            return
        
        # Check if we have at least 3 features
        if len(feature_list) < 3:
            print("Skipping 3D plot - need at least 3 features")
            return
        
        # Use first 3 features for 3D plot
        feature1, feature2, feature3 = feature_list[:3]
        
        print(f"Creating TRUE 3D scatter plot for features: {feature1}, {feature2}, {feature3}")
        print(f"Each point shows MSE reductions for all three features in the same scenario")
        print(f"X-axis: {feature1} MSE reduction")
        print(f"Y-axis: {feature2} MSE reduction") 
        print(f"Z-axis: {feature3} MSE reduction")
        
        # Extract true 3D points where each point has all three coordinates
        points_3d = self._extract_points_for_triplet_correct(impact_df, feature1, feature2, feature3)
        
        if not points_3d:
            print("No valid 3D points found - trying alternative approach...")
            # Fallback to the previous approach if the correct method doesn't find points
            points_3d = []
            points_12 = self._extract_points_for_pair(impact_df, feature1, feature2)
            points_13 = self._extract_points_for_pair(impact_df, feature1, feature3)
            
            for x1_val, x2_val, color_12 in points_12:
                best_x3_val = 0
                for x1_val_2, x3_val, color_13 in points_13:
                    if abs(x1_val - x1_val_2) < 1e-6:
                        best_x3_val = x3_val
                        break
                points_3d.append((x1_val, x2_val, best_x3_val, color_12))
        
        if not points_3d:
            print("No valid 3D points found with either method - skipping 3D plot")
            return
        
        print(f"Found {len(points_3d)} true 3D points")
        
        # Create the 3D plot
        fig = go.Figure()
        
        # Separate by color and create traces
        red_points = [(x, y, z) for x, y, z, c in points_3d if c == 'red']
        blue_points = [(x, y, z) for x, y, z, c in points_3d if c == 'blue']
        green_points = [(x, y, z) for x, y, z, c in points_3d if c == 'green']
        
        # Sample points for performance
        red_points = random.sample(red_points, min(len(red_points), 300))
        blue_points = random.sample(blue_points, min(len(blue_points), 300))
        green_points = random.sample(green_points, min(len(green_points), 300))
        
        if red_points:
            x_red, y_red, z_red = zip(*red_points)
            fig.add_trace(go.Scatter3d(
                x=x_red, y=y_red, z=z_red,
                mode='markers',
                marker=dict(size=4, color='red', opacity=0.6),
                name=f'{feature1} added last'
            ))
        
        if blue_points:
            x_blue, y_blue, z_blue = zip(*blue_points)
            fig.add_trace(go.Scatter3d(
                x=x_blue, y=y_blue, z=z_blue,
                mode='markers',
                marker=dict(size=4, color='blue', opacity=0.6),
                name=f'{feature2} added last'
            ))
        
        if green_points:
            x_green, y_green, z_green = zip(*green_points)
            fig.add_trace(go.Scatter3d(
                x=x_green, y=y_green, z=z_green,
                mode='markers',
                marker=dict(size=4, color='green', opacity=0.6),
                name=f'{feature3} added last'
            ))
        
        # Calculate 3D variance for display
        all_coords = [(x, y, z) for x, y, z, c in points_3d]
        if len(all_coords) > 10:
            coords_array = np.array(all_coords)
            try:
                variance_3d = np.var(coords_array, axis=0)
                total_variance = np.sum(variance_3d)
                L_score_3d = round(total_variance, 3)
            except:
                L_score_3d = 0
        else:
            L_score_3d = 0
        
        print(f"3D analysis variance: {L_score_3d:.4f}")
        
        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title=f"{feature1} MSE Reduction",
                yaxis_title=f"{feature2} MSE Reduction",
                zaxis_title=f"{feature3} MSE Reduction"
            ),
            height=600,
            showlegend=True,
            margin=dict(l=0, r=0, b=0, t=40),
            title=f"True 3D Feature Analysis: {feature1}, {feature2}, {feature3}<br>Each point shows all three MSE reductions from same scenario"
        )
        
        # Save as HTML file (interactive) - this always works
        html_filename = f'{output_dir}/3d_scatter_plot.html'
        fig.write_html(html_filename)
        print(f"3D scatter plot saved to: {html_filename}")
        
        # Try to save as PNG file (static) - optional
        try:
            png_filename = f'{output_dir}/3d_scatter_plot.png'
            fig.write_image(png_filename)
            print(f"3D scatter plot (PNG) saved to: {png_filename}")
        except Exception as e:
            print(f"Could not save PNG file (kaleido not installed): {e}")
            print("Install kaleido with: pip install -U kaleido")
            print("HTML file is still available for interactive viewing")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Feature Analysis Tool')
    parser.add_argument("--csv_file", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--approach", type=str, choices=['combos', 'walks'], 
                       default='combos', help="Analysis approach")
    parser.add_argument("--l_score_method", type=str, choices=['AS', 'CDM'], 
                       default='AS', help="L-score calculation method")
    parser.add_argument("--model_type", type=str, choices=['linear', 'poly', 'rf', 'mlp', 'gam'],
                       default='rf', help="Model type")
    parser.add_argument("--poly_degree", type=int, default=2, help="Polynomial degree")
    parser.add_argument("--n_paths", type=int, default=50, help="Number of random paths")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.csv_file)
    
    # Initialize analyzer
    analyzer = FeatureAnalyzer(
        approach=args.approach,
        l_score_method=args.l_score_method,
        model_type=args.model_type,
        poly_degree=args.poly_degree,
        n_paths=args.n_paths,
        random_seed=args.random_seed
    )
    
    # Analyze features
    print(f"Starting analysis with {args.approach} approach and {args.l_score_method} L-score method...")
    results = analyzer.analyze_features(df)
    
    # Create visualizations
    analyzer.create_visualizations(results, args.csv_file)
    
    print("Analysis complete!")


if __name__ == "__main__":
    main() 