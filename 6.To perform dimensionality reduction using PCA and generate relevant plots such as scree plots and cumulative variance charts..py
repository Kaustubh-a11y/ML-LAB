import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

print("\n" + "=" * 65)
print("AIM 6 : PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("=" * 65)

iris = load_iris()
X = iris.data
y = iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print("\nExplained Variance Ratio:\n")
print(pca.explained_variance_ratio_)

print("\nCumulative Explained Variance:\n")
print(np.cumsum(pca.explained_variance_ratio_))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(
    range(1, len(pca.explained_variance_ratio_) + 1),
    pca.explained_variance_ratio_
)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(
    range(1, len(pca.explained_variance_ratio_) + 1),
    np.cumsum(pca.explained_variance_ratio_),
    marker='o'
)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Variance Plot")
plt.grid(True)

plt.tight_layout()
plt.savefig("aim6_pca_variance_plots.png")

print("\nVariance plots saved as 'aim6_pca_variance_plots.png'")

pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for i in range(3):
    plt.scatter(
        X_pca_2d[y == i, 0],
        X_pca_2d[y == i, 1],
        label=iris.target_names[i]
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("2D PCA Projection of Iris Dataset")
plt.legend()
plt.grid(True)
plt.savefig("aim6_pca_2d_projection.png")

print("2D PCA plot saved as 'aim6_pca_2d_projection.png'")
