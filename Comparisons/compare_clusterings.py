import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
data = pd.read_csv('Energy_dataset.csv')

# Select features for clustering
features = ['Heating Load', 'Cooling Load']
X = data[features].values

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
data['KMeans Cluster'] = kmeans.fit_predict(X_scaled)

# Step 4: Apply DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
data['DBSCAN Cluster'] = dbscan.fit_predict(X_scaled)

# Step 5: Apply Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=3, random_state=42)
data['GMM Cluster'] = gmm.fit_predict(X_scaled)

# Visualization function
def plot_clusters(X_scaled, labels, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    plt.title(title)
    plt.xlabel('Heating Load (Standardized)')
    plt.ylabel('Cooling Load (Standardized)')
    plt.colorbar(label='Cluster')
    plt.show()

# Step 6: Visualize the clusters for each algorithm
plot_clusters(X_scaled, data['KMeans Cluster'], "K-Means Clustering")
plot_clusters(X_scaled, data['DBSCAN Cluster'], "DBSCAN Clustering")
plot_clusters(X_scaled, data['GMM Cluster'], "Gaussian Mixture Model Clustering")

# Step 7: Compare cluster means for K-Means and GMM (DBSCAN may have noise points labeled as -1)
print("\nK-Means Cluster Analysis:")
print(data.groupby('KMeans Cluster')[features].mean())

print("\nGMM Cluster Analysis:")
print(data.groupby('GMM Cluster')[features].mean())

# DBSCAN does not guarantee a cluster for all points
print("\nDBSCAN Cluster Distribution:")
print(data['DBSCAN Cluster'].value_counts())
