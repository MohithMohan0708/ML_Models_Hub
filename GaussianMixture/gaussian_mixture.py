import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv('synthetic_dataset.csv')

features = ['Age', 'Income', 'Spending Score']
X = data[features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

gmm = GaussianMixture(n_components=3,random_state=42)
gmm.fit(X_scaled)

clusters = gmm.predict(X_scaled)

data['clusters'] = clusters

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10,6))
plt.scatter(X_pca[:,0],X_pca[:,1],c=clusters,cmap='viridis',label='Data Points')

centroids = pca.transform(gmm.means_)

plt.scatter(centroids[:,0],centroids[:,1],s=200,c='red',marker='X',label='Centroids')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Gaussian Mixture Model')
plt.colorbar(label='Clusters')
plt.show()

print("Cluster Centres:")
print(gmm.means_)

print("Cluster Analysis:")
print(data.groupby('clusters')[features].mean())