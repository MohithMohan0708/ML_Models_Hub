import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Step 1: Load the dataset
data = pd.read_csv('Energy_dataset.csv')

# Select features for clustering
features = ['Heating Load', 'Cooling Load']
X = data[features].values

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply the Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=3, random_state=42)  # Using 3 clusters


# Step 4: Predict clusters
clusters = gmm.fit_predict(X_scaled)

# Add cluster labels to the dataset
data['Cluster'] = clusters

# Step 5: Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', label='Data Points')

# Mark the centroids (means of each Gaussian component)
centroids = gmm.means_
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

plt.title("Gaussian Mixture Model Clustering (Heating Load vs Cooling Load)")
plt.xlabel("Heating Load (Standardized)")
plt.ylabel("Cooling Load (Standardized)")
plt.colorbar(label='Cluster')
plt.legend()
plt.show()

# Step 6: Print the cluster centers and analysis
print("Cluster Centers (means in standardized scale):")
print(gmm.means_)

print("\nCluster Analysis (Mean of each feature per cluster):")
print(data.groupby('Cluster')[features].mean())
