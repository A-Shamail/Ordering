import argparse
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def load_model(model_name):
    if model_name == 'linear':
        return LinearRegression()
    elif model_name == 'random_forest':
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
    elif model_name == 'xgboost':
        return XGBRegressor()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def plot_interaction_matrix(interaction_values, feature_names):
    # Calculate mean absolute interaction values
    mean_interactions = np.abs(interaction_values).mean(0)
    
    # Create a DataFrame for the interaction matrix
    interaction_df = pd.DataFrame(
        mean_interactions,
        columns=feature_names,
        index=feature_names
    )
    
    # Create a heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(interaction_df, annot=True, cmap='coolwarm', center=0)
    plt.title('SHAP Interaction Values Matrix')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Train a model and display SHAP interaction values.")
    parser.add_argument('--csv', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--model', type=str, required=True, choices=['linear', 'random_forest', 'xgboost'], help='Model to train')

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv)
    X = df[[col for col in df.columns if col.startswith('X')]]
    y = df['Y']

    # Print data characteristics
    print("\nData Overview:")
    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {len(X.columns)}")
    print("\nFeature statistics:")
    print(X.describe())
    print("\nTarget variable statistics:")
    print(y.describe())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load and train model
    model = load_model(args.model)
    model.fit(X_train, y_train)

    # Print model performance
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"\nModel R² score on training set: {train_score:.4f}")
    print(f"Model R² score on test set: {test_score:.4f}")

    # Calculate predictions to verify model is working
    y_pred = model.predict(X_test)
    print("\nSample of predictions vs actual:")
    for pred, actual in zip(y_pred[:5], y_test[:5]):
        print(f"Predicted: {pred:.4f}, Actual: {actual:.4f}")

    # Calculate SHAP interaction values
    print("\nCalculating SHAP interaction values...")
    explainer = shap.Explainer(model, X_train)
    shap_interaction_values = explainer.shap_interaction_values(X_test)
    
    # Print shape and sample of interaction values
    print(f"\nShape of interaction values: {shap_interaction_values.shape}")
    print("\nSample of non-zero interaction values (if any):")
    non_zero = np.where(shap_interaction_values != 0)
    if len(non_zero[0]) > 0:
        for i in range(min(5, len(non_zero[0]))):
            idx = non_zero[0][i]
            feat1 = non_zero[1][i]
            feat2 = non_zero[2][i]
            val = shap_interaction_values[idx, feat1, feat2]
            print(f"Sample {idx}, Features {X.columns[feat1]}-{X.columns[feat2]}: {val}")
    else:
        print("All interaction values are zero!")

    # Print feature names
    print("\nFeatures:", list(X_test.columns))
    
    # Print mean absolute interaction values
    print("\nMean Absolute SHAP Interaction Values Matrix:")
    mean_interactions = np.abs(shap_interaction_values).mean(0)
    interaction_df = pd.DataFrame(
        mean_interactions,
        columns=X_test.columns,
        index=X_test.columns
    )
    print(interaction_df.round(4))

    # Plot interaction matrix
    plot_interaction_matrix(shap_interaction_values, X_test.columns)

if __name__ == "__main__":
    main()
