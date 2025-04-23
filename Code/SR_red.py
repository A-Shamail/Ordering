import numpy as np
import pandas as pd
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import argparse
from itertools import combinations

def setup_symbolic_regression(iterations=10, processes=4, max_size=10):
    """
    Setup the symbolic regression model with specified parameters
    """
    unary_operators = [
        "abs", "sin", "cos", "tan",
        "asin", "acos", "atan",
        "sinh", "cosh", "tanh",
        "sqrt", "exp", "log",
        "sign"
    ]

    binary_operators = [
        "+", "-", "*", "/", "^",
        "max", "min"
    ]

    model = PySRRegressor(
        niterations=iterations,
        unary_operators=unary_operators,
        binary_operators=binary_operators,
        model_selection="best",
        elementwise_loss="loss(x, y) = (x - y)^2",
        verbosity=1,
        maxsize=max_size,
        procs=processes,
        constraints={"^": (-1, 1)},
        temp_equation_file = True,
        delete_tempfiles = True
    )
    
    return model

def analyze_feature_relationships(df, features, model, test_size=0.2, random_state=42):
    """
    Analyze relationships between all pairs of features
    """
    results = []
    
    # Analyze all pairs of features
    feature_pairs = list(combinations(features, 2))
    print(f"\nAnalyzing {len(feature_pairs)} feature pairs...")
    print("=" * 50)
    
    for feature1, feature2 in feature_pairs:
        print(f"\nAnalyzing pair: {feature1} and {feature2}")
        print("-" * 30)
        
        # First direction: feature1 → feature2
        X = df[feature1].values.reshape(-1, 1)
        y = df[feature2].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        model.fit(X_train, y_train)
        
        eqs = model.equations_
        eqs["test_mse"] = eqs["lambda_format"].apply(
            lambda f: mean_squared_error(y_test, f(X_test))
        )
        best_eq_forward = eqs.sort_values("test_mse").iloc[0]
        
        # Second direction: feature2 → feature1
        X = df[feature2].values.reshape(-1, 1)
        y = df[feature1].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        model.fit(X_train, y_train)
        
        eqs = model.equations_
        eqs["test_mse"] = eqs["lambda_format"].apply(
            lambda f: mean_squared_error(y_test, f(X_test))
        )
        best_eq_reverse = eqs.sort_values("test_mse").iloc[0]
        
        results.extend([
            {
                'feature1': feature1,
                'feature2': feature2,
                'direction': '→',
                'equation': best_eq_forward['equation'],
                'complexity': best_eq_forward['complexity'],
                'rmse': np.sqrt(best_eq_forward['test_mse'])
            },
            {
                'feature1': feature2,
                'feature2': feature1,
                'direction': '→',
                'equation': best_eq_reverse['equation'],
                'complexity': best_eq_reverse['complexity'],
                'rmse': np.sqrt(best_eq_reverse['test_mse'])
            }
        ])
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze Feature Relationships using Symbolic Regression')
    parser.add_argument("--csv_file", type=str, required=True, help='Path to CSV file')
    parser.add_argument("--features", type=str, nargs='+', required=True, help='Feature columns to analyze')
    parser.add_argument("--iterations", type=int, default=10, help='Number of iterations')
    parser.add_argument("--processes", type=int, default=4, help='Number of processes')
    parser.add_argument("--max_size", type=int, default=10, help='Maximum size of equations')
    parser.add_argument("--test_size", type=float, default=0.2, help='Test set size')
    parser.add_argument("--random_state", type=int, default=42, help='Random state')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.csv_file}")
    df = pd.read_csv(args.csv_file)
    
    # Verify columns exist in the dataset
    missing_cols = [col for col in args.features if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataset: {missing_cols}")
    
    # Setup symbolic regression
    print("Setting up symbolic regression model...")
    model = setup_symbolic_regression(
        iterations=args.iterations,
        processes=args.processes,
        max_size=args.max_size
    )
    
    # Analyze relationships
    results_df = analyze_feature_relationships(
        df,
        features=args.features,
        model=model,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Print final summary
    print("\n" + "=" * 50)
    print("SUMMARY OF BEST RELATIONSHIPS")
    print("=" * 50)
    for _, row in results_df.sort_values('rmse').iterrows():
        print(f"\n{row['feature1']} {row['direction']} {row['feature2']}")
        print(f"Equation: {row['equation']}")
        print(f"RMSE: {row['rmse']:.6f}")