import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the embeddings data from poems_data.csv, skipping the first row
data = pd.read_csv(r'G:\My Drive\Google Drive Documents\Sneha Saragadam\Sneha Engineering Plan 2022-2025\Engineering Preparation\2nd Year\machinelearning\project\poems_embeddings_bert.csv', skiprows=1)

# Select the embedding columns (1st to 768th)
X_selected = data.iloc[:, :768].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Plot the explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='-')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Elbow Curve for PCA')
plt.grid(True)
plt.show()
