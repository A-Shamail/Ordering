import numpy as np
import pandas as pd
from pysr import PySRRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Generate Data ---
np.random.seed(42)
n_samples = 1500

X1 = np.random.uniform(-1, 1, n_samples)
X2 = np.sin(X1) + np.random.normal(0, 0.05, n_samples)
# X2 = np.random.uniform(-1, 1, n_samples)
# X3 = np.sin(X1) + np.random.normal(0, 0.05, n_samples)
# Y = X1 * X2 + np.random.normal(0, 0.05, n_samples)

df = pd.DataFrame({
    "X1": X1,
    "X2": X2,
    # "X3": X3,
        # "Y": Y
})

X = df[["X1"]].values
y = df["X2"].values

# --- Split into Train/Validation ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Define Operators ---
unary_operators = [
    "abs", "sin", "cos", "tan",
    "asin", "acos", "atan",
    "sinh", "cosh", "tanh",
    "sqrt", "exp", "log",
    "sign", "floor", "ceil"
]

binary_operators = [
    "+", "-", "*", "/", "^",
    "max", "min"
]

# --- Initialize Model ---
model = PySRRegressor(
    niterations=10,
    unary_operators=unary_operators,
    binary_operators=binary_operators,
    model_selection="best",  # This selects best by internal loss
    loss="loss(x, y) = (x - y)^2",
    verbosity=1,
    maxsize=40,
    procs=4,
    constraints={"^": (-1, 1)}
)

# --- Train on Training Set ---
model.fit(X_train, y_train)

# --- Evaluate All Discovered Models on Validation Set ---
eqs = model.equations_
print("Columns in equations_ DataFrame:", model.equations_.columns.tolist())
eqs["val_mse"] = eqs["lambda_format"].apply(lambda f: mean_squared_error(y_val, f(X_val)))
eqs_sorted = eqs.sort_values("val_mse")

# --- Report Best Validating Equation ---
print("\nBest Equation by Validation MSE:")
print(eqs_sorted.iloc[0][["equation", "val_mse"]])

# --- Predict using Best Generalizing Model ---
best_model = eqs_sorted.iloc[0]["lambda"]
y_pred = best_model(X_val)

# --- Plot ---
plt.figure(figsize=(6, 4))
plt.scatter(y_val, y_pred, alpha=0.5, label="Predicted vs True")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
plt.xlabel("True Y")
plt.ylabel("Predicted Y")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
