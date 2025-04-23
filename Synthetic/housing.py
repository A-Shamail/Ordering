import numpy as np
import pandas as pd
from pysr import PySRRegressor
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# --- Load and Prepare Housing Data ---
housing = fetch_california_housing()
X = housing.data
feature_names = housing.feature_names

# Create DataFrame with feature names
df = pd.DataFrame(X, columns=feature_names)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=feature_names)

# --- Define Supported Operator Sets ---
unary_operators = [
    "abs", "sin", "cos", 
    "exp", "log",
    "sqrt"
]

binary_operators = [
    "+", "-", "*", "/",
    "max", "min"
]

# Function to test relationships
def test_feature_relationships(df_scaled, target_feature, other_features):
    X = df_scaled[other_features].values
    y = df_scaled[target_feature].values
    
    model = PySRRegressor(
        niterations=10,
        unary_operators=unary_operators,
        binary_operators=binary_operators,
        model_selection="best",
        loss="loss(x, y) = (x - y)^2",
        verbosity=1,
        maxsize=15,
        procs=4,
        populations=5,  # Run multiple populations in parallel
        fraction_validation=0.2,  # Use 20% of data for validation
        early_stop_condition=10,  # Stop if no improvement after 10 iterations
        constraints={
            "^": (-1, 1),
        }
    )
    
    model.fit(X, y)
    
    print(f"\nTesting {target_feature} as a function of other features:")
    print("Best equation found:")
    print(model.get_best())
    
    # Plot results
    # y_pred = model.predict(X)
    # plt.figure(figsize=(10, 6))
    # plt.scatter(y, y_pred, alpha=0.5)
    # plt.plot([-3, 3], [-3, 3], 'r--')
    # plt.xlabel(f'True {target_feature} Values (scaled)')
    # plt.ylabel(f'Predicted {target_feature} Values (scaled)')
    # plt.title(f'True vs Predicted Values for {target_feature}')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    return model.get_best()

# Test each feature as a function of others
results = {}
for feature in feature_names:
    other_features = [f for f in feature_names if f != feature]
    print(f"\nFeature being predicted: {feature}")
    print("Using features:", other_features)
    results[feature] = test_feature_relationships(df_scaled, feature, other_features)

# Print summary of all relationships found
print("\nSummary of relationships found:")
print("--------------------------------")
for feature, equation in results.items():
    print(f"\n{feature}:")
    print(equation)