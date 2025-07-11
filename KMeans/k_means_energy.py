import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load your custom dataset
data = pd.read_csv('Energy_dataset.csv')

# Define features (using 'Heating Load' and 'Cooling Load')
X = data[['Heating Load', 'Cooling Load']]

# Define the number of clusters (adjust as needed)
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)


# Predict cluster labels
cluster_labels = kmeans.fit_predict(X)

# Plot the clusters
plt.figure(figsize=(10, 7))
plt.scatter(data['Heating Load'], data['Cooling Load'], c=cluster_labels, cmap='viridis', s=50, alpha=0.7, label='Data Points')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X',s=200, edgecolor='k', label='Centroids')
plt.title('K-Means Clustering on Energy Dataset (Using Heating Load and Cooling Load)')
plt.xlabel('Heating Load')
plt.ylabel('Cooling Load')
plt.legend()
plt.show()
