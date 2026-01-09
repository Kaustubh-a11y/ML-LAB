import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("\n" + "=" * 65)
print("AIM 4 : LINEAR REGRESSION USING SCIKIT-LEARN")
print("=" * 65)

np.random.seed(7)
X = np.random.rand(120, 1) * 12
y = 3.2 * X + 4 + np.random.randn(120, 1) * 1.8

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Results:\n")
print(f"Coefficient : {model.coef_[0][0]:.4f}")
print(f"Intercept   : {model.intercept_[0]:.4f}")
print(f"MSE         : {mse:.4f}")
print(f"R² Score    : {r2:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label='Actual Values')
plt.plot(X_test, y_pred, linewidth=2, label='Predicted Line')
plt.xlabel("Input Feature (X)")
plt.ylabel("Target Value (y)")
plt.title("Linear Regression Model Performance")
plt.legend()
plt.grid(True)
plt.savefig("aim4_linear_regression_output.png")

print("\nGraph saved as 'aim4_linear_regression_output.png'")
