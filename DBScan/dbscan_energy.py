import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Load your custom dataset
data = pd.read_csv('Energy_dataset.csv')

# Define features (using 'Relative Compactness' and 'Heating Load')
X = data[['Heating Load','Cooling Load']]

# Initialize DBSCAN model
model_dbscan = DBSCAN(eps=0.1, min_samples=2)

# Fit the model and assign cluster labels
data['DBSCAN Cluster'] = model_dbscan.fit_predict(X)

# Plot the clusters
plt.figure(figsize=(10, 7))
plt.scatter(data['Heating Load'], data['Cooling Load'], c=data['DBSCAN Cluster'], cmap='viridis', marker='o', s=50)
plt.xlabel('Heating Load')
plt.ylabel('Cooling Load')
plt.title('DBSCAN Clustering on Selected Features')
plt.colorbar(label='Cluster Label')
plt.show()
