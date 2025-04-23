import numpy as np
import pandas as pd

np.random.seed(43)

n_samples = 1000

X1 = np.random.uniform(-1, 1, n_samples)  
X2 = np.random.uniform(-1, 1, n_samples)
X3 = np.random.uniform(-1, 1, n_samples)

Z = np.random.uniform(-1, 1, n_samples)

X8 = 0.5 * Z + np.random.normal(0, 0.3, n_samples) 
Y = X1 * X2 + (0.7 * X3) + (0.5 * X8) + np.random.normal(0, 0.1, n_samples) 

X6 = np.sin(X3) + np.random.normal(0, 0.05, n_samples)

X5 = np.random.uniform(-1, 1, n_samples)
X4 = 0.7 * X5 + np.random.normal(0, 0.3, n_samples) 

X7 = 0.9 * X6 + np.random.uniform(-1, 1, n_samples)

df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5, 'X6': X6, 'X7': X7, 'X8': X8, 'Y': Y})


print(df.head())

# df.to_csv("synthetic_data.csv", index=False)