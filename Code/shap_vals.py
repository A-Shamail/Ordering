import argparse
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def load_model(model_name):
    if model_name == 'linear':
        return LinearRegression()
    elif model_name == 'random_forest':
        return RandomForestRegressor()
    elif model_name == 'xgboost':
        return XGBRegressor()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def main():
    parser = argparse.ArgumentParser(description="Train a model and display SHAP values.")
    parser.add_argument('--csv', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--model', type=str, required=True, choices=['linear', 'random_forest', 'xgboost'], help='Model to train')

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv)
    X = df[[col for col in df.columns if col.startswith('X')]]
    y = df['Y']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load and train model
    model = load_model(args.model)
    model.fit(X_train, y_train)

    # Explain with SHAP
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # Print mean absolute SHAP values for each feature
    mean_shap = pd.DataFrame({
        'Feature': X_test.columns,
        'Mean |SHAP|': np.abs(shap_values.values).mean(axis=0)
    })
    # Use sigmoid-like function to map to (0, -1)
    mean_shap['Importance'] = -1 * (2 / (1 + np.exp(-mean_shap['Mean |SHAP|'])) - 1)
    mean_shap = mean_shap.sort_values('Mean |SHAP|', ascending=False)
    
    print("\nMean Absolute SHAP Values by Feature:")
    print(mean_shap.to_string(index=False))

    # Plot summary
    shap.summary_plot(shap_values, X_test)

if __name__ == "__main__":
    main()
