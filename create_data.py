import numpy as np
import pandas as pd

# Generate synthetic dataset
np.random.seed(42)  # For reproducibility
num_samples = 1000
X = np.random.rand(num_samples, 2)  # 1000 samples with 2 features each
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Target labels

# Create a DataFrame and save to CSV
data = pd.DataFrame({'feature1': X[:, 0], 'feature2': X[:, 1], 'label': y})
data.to_csv('dataset/synthetic_data.csv', index=False)
