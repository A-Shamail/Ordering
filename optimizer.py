"""Module for Random Order Optimization through multiple path sampling."""

import numpy as np
import pandas as pd
from sklearn.metrics import auc, mean_squared_error
import multiprocessing as mp
from functools import partial

class RandomOrderOptimizer:
    """Handles generation and analysis of multiple random paths for discovering optimal orderings."""
    
    def __init__(self, base_model_class, model_params, n_total_paths=50, n_subsamples=5, subsample_size=15):
        """Initialize RandomOrderOptimizer.
        
        Args:
            base_model_class: The model class to use (e.g., RandomForestRegressor)
            model_params: Dictionary of parameters for model initialization
            n_total_paths: Number of total random paths to generate
            n_subsamples: Number of subsamples to create
            subsample_size: Size of each subsample
        """
        self.base_model_class = base_model_class
        self.model_params = model_params
        self.n_total_paths = n_total_paths
        self.n_subsamples = n_subsamples
        self.subsample_size = subsample_size

    def evaluate_element_set(self, elements, X_train, X_test, y_train, y_test):
        """Evaluate performance for a set of elements using the full dataset.
        
        Args:
            elements: List of element names to use
            X_train: Training data
            X_test: Test data
            y_train: Training labels
            y_test: Test labels
            
        Returns:
            float: Performance metric for the element set
        """
        
        model = self.base_model_class(**self.model_params)
        print(model)
        model.fit(X_train[elements], y_train)
        y_pred = model.predict(X_test[elements])
        return mean_squared_error(y_test, y_pred)

    def evaluate_ranking(self, feature_order, X_train, X_test, y_train, y_test):
        """Evaluate a feature ranking by progressively adding features.
        
        Args:
            feature_order: List of features in order to evaluate
            X_train: Training data
            X_test: Test data
            y_train: Training labels
            y_test: Test labels
            
        Returns:
            list: MSE scores for each progressive feature set
        """
        mses = []
        current_features = []
        for feature in feature_order:
            current_features.append(feature)
            mse = self.evaluate_element_set(current_features, X_train, X_test, y_train, y_test)
            mses.append(mse)
        return mses

    def _evaluate_single_path(self, _, elements, X_train, X_test, y_train, y_test):
        """Generate and evaluate a single random path.
        
        Args:
            _: Unused iteration index (required for multiprocessing)
            elements: List of element names
            X_train: Training data
            X_test: Test data
            y_train: Training labels
            y_test: Test labels
            
        Returns:
            dict: Path information with order and metrics
        """
        element_order = np.random.permutation(elements)
        path_metrics = []
        current_elements = []
        
        for element in element_order:
            current_elements.append(element)
            metric = self.evaluate_element_set(
                current_elements, X_train, X_test, y_train, y_test
            )
            path_metrics.append(metric)
        
        return {
            'order': element_order,
            'metrics': path_metrics
        }

    def generate_paths(self, X_train, X_test, y_train, y_test, elements):
        """Generate multiple random paths and evaluate performance.
        
        Args:
            X_train: Training data
            X_test: Test data
            y_train: Training labels
            y_test: Test labels
            elements: List of element names
            
        Returns:
            List of dictionaries containing path information
        """
        worker_fn = partial(
            self._evaluate_single_path,
            elements=elements,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )
        
        n_cores = mp.cpu_count()
        
        with mp.Pool(processes=n_cores) as pool:
            all_paths_data = pool.map(worker_fn, range(self.n_total_paths))
            
        return all_paths_data

    def calculate_cumulative_ranking(self, paths_subset):
        """Calculate element ranking based on cumulative metric improvements.
        
        Args:
            paths_subset: List of path dictionaries
            
        Returns:
            pandas.Series of element rankings
        """
        element_improvements = {element: [] for element in paths_subset[0]['order']}
        
        for path_data in paths_subset:
            path_order = path_data['order']
            path_metrics = path_data['metrics']
            
            element_improvements[path_order[0]].append(0)
            for i in range(1, len(path_order)):
                element = path_order[i]
                metric_improvement = path_metrics[i-1] - path_metrics[i]
                element_improvements[element].append(metric_improvement)
        
        cumulative_improvements = {}
        for element in element_improvements:
            cumulative_improvements[element] = np.mean(element_improvements[element])
            
        return pd.Series(cumulative_improvements).sort_values(ascending=False)

    def create_subsamples(self, all_paths_data, X_train, X_test, y_train, y_test):
        """Create and evaluate path subsamples.
        
        Args:
            all_paths_data: List of all path dictionaries
            X_train: Training data
            X_test: Test data
            y_train: Training labels
            y_test: Test labels
            
        Returns:
            List of subsample results
        """
        subsample_results = []
        
        for i in range(self.n_subsamples):
            subsample = np.random.choice(len(all_paths_data), size=self.subsample_size, replace=False)
            subsample_paths = [all_paths_data[idx] for idx in subsample]
            
            ranking = self.calculate_cumulative_ranking(subsample_paths)
            
            metrics = []
            current_elements = []
            for element in ranking.index:
                current_elements.append(element)
                metric = self.evaluate_element_set(
                    current_elements, X_train, X_test, y_train, y_test
                )
                metrics.append(metric)
            
            auc_score = np.trapz(metrics) / len(metrics)
            
            subsample_results.append({
                'ranking': ranking,
                'metrics': metrics,
                'auc': auc_score,
                'paths_indices': subsample,
                'id': i
            })
            
        return subsample_results

    def get_best_subsample(self, subsample_results, metric='auc'):
        """Select best subsample based on metric.
        
        Args:
            subsample_results: List of subsample result dictionaries
            metric: Metric to use for selection (default: 'auc')
            
        Returns:
            Dictionary containing best subsample results
        """
        return min(subsample_results, key=lambda x: x[metric])

    def optimize(self, X_train, X_test, y_train, y_test, elements):
        """Main method to discover optimal ordering through random path sampling.
        
        Args:
            X_train: Training data
            X_test: Test data
            y_train: Training labels
            y_test: Test labels
            elements: List of element names
            
        Returns:
            Dictionary containing:
            - all_paths: All generated paths with metrics
            - subsamples: All subsample results
            - best_subsample: Best performing subsample
            - final_ranking: Element ranking from best subsample
        """
        all_paths = self.generate_paths(X_train, X_test, y_train, y_test, elements)
        subsamples = self.create_subsamples(all_paths, X_train, X_test, y_train, y_test)
        best_subsample = self.get_best_subsample(subsamples)
        
        return {
            'all_paths': all_paths,
            'subsamples': subsamples,
            'best_subsample': best_subsample,
            'final_ranking': best_subsample['ranking']
        }

    def generate_paths_only(self, X_train, X_test, y_train, y_test, elements):
        """Generate multiple random paths and their performance metrics without ranking.
        
        Args:
            X_train: Training data
            X_test: Test data
            y_train: Training labels
            y_test: Test labels
            elements: List of element names
            
        Returns:
            List of dictionaries, each containing:
            - order: numpy array of element ordering
            - metrics: list of performance metrics for progressive feature sets
        """
        return self.generate_paths(X_train, X_test, y_train, y_test, elements)