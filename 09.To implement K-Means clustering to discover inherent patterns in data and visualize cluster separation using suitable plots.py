import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

print("\n" + "=" * 65)
print("AIM 9 : K-MEANS CLUSTERING")
print("=" * 65)

iris = load_iris()
X = iris.data[:, :2]

kmeans = KMeans(n_clusters=3, random_state=25)
clusters = kmeans.fit_predict(X)

print("\nCluster Centers:\n")
print(kmeans.cluster_centers_)

print(f"\nInertia (WCSS): {kmeans.inertia_:.4f}")

inertia_values = []
k_values = range(1, 11)

for k in k_values:
    model = KMeans(n_clusters=k, random_state=25)
    model.fit(X)
    inertia_values.append(model.inertia_)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(k_values, inertia_values, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True)

plt.subplot(1, 2, 2)
for i in range(3):
    plt.scatter(X[clusters == i, 0], X[clusters == i, 1], label=f"Cluster {i}")

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    marker='X',
    label="Centroids"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering Result")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("aim9_kmeans_clustering.png")

print("\nK-Means clustering plots saved as 'aim9_kmeans_clustering.png'")
