import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

print("\n" + "=" * 65)
print("AIM 8 : NAÏVE BAYES CLASSIFIER")
print("=" * 65)

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=18
)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nNaïve Bayes Results:\n")
print(f"Accuracy : {accuracy:.4f}")

print("\nConfusion Matrix:\n")
print(conf_matrix)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

y_pred_proba = model.predict_proba(X_test)

print("\nSample Prediction Probabilities (First 5 Samples):\n")
print(y_pred_proba[:5])
