import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

print("\n" + "=" * 65)
print("AIM 7 : K-NEAREST NEIGHBORS (KNN) CLASSIFIER")
print("=" * 65)

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=15
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nKNN Results (k = 7):\n")
print(f"Accuracy : {accuracy:.4f}")

print("\nConfusion Matrix:\n")
print(conf_matrix)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

k_range = range(1, 21)
accuracy_scores = []

for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    accuracy_scores.append(accuracy_score(y_test, predictions))

plt.figure(figsize=(10, 6))
plt.plot(k_range, accuracy_scores, marker='o')
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs k Value")
plt.grid(True)
plt.savefig("aim7_knn_k_analysis.png")

print("\nKNN k-value analysis plot saved as 'aim7_knn_k_analysis.png'")
