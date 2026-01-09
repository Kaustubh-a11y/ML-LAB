import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

print("\n" + "=" * 65)
print("AIM 5 : LOGISTIC REGRESSION USING SCIKIT-LEARN")
print("=" * 65)

X, y = make_classification(
    n_samples=600,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=21
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=21
)

model = LogisticRegression(random_state=21)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nLogistic Regression Results:\n")
print(f"Accuracy : {accuracy:.4f}")

print("\nConfusion Matrix:\n")
print(conf_matrix)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
