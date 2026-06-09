import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


class LinearRegressionGD:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            loss = np.mean((y - y_pred) ** 2)
            self.losses.append(loss)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


class LogisticRegressionGD:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            linear = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear)
        return [1 if i > 0.5 else 0 for i in y_pred]


print("\n===== LINEAR REGRESSION =====")

X, y = make_regression(
    n_samples=200,
    n_features=1,
    noise=20,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = LinearRegressionGD(lr=0.01, epochs=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("MSE :", mean_squared_error(y_test, predictions))
print("R2 Score :", r2_score(y_test, predictions))

plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, label="Actual")
plt.plot(X_test, predictions, color="red", label="Predicted")
plt.title("Linear Regression")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(model.losses)
plt.title("Linear Regression Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


print("\n===== LOGISTIC REGRESSION =====")

X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

clf = LogisticRegressionGD(lr=0.1, epochs=1000)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print("Accuracy :", accuracy_score(y_test, pred))

plt.figure(figsize=(8, 5))
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=pred,
    cmap="coolwarm"
)
plt.title("Logistic Regression Classification")
plt.show()