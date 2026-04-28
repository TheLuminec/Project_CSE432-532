from minilearn.models.base import Classifier
import numpy as np

class LinearRegression(Classifier):
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0

        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.coef_) + self.intercept_
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.coef_) + self.intercept_
        return linear_model

    def score(self, X, y):
        y_predicted = self.predict(X)
        return np.mean((y_predicted - y) ** 2)

class LogisticRegression(Classifier):
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)

        self.coef_ = np.zeros((n_features, n_classes))
        self.intercept_ = np.zeros(n_classes)

        y_onehot = np.zeros((n_samples, n_classes))
        y_onehot[np.arange(n_samples), y_idx] = 1

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.coef_) + self.intercept_
            exp_model = np.exp(linear_model - np.max(linear_model, axis=1, keepdims=True))
            y_predicted = exp_model / np.sum(exp_model, axis=1, keepdims=True)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y_onehot))
            db = (1 / n_samples) * np.sum(y_predicted - y_onehot, axis=0)
            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.coef_) + self.intercept_
        return self.classes_[np.argmax(linear_model, axis=1)]

    def score(self, X, y):
        y_predicted = self.predict(X)
        return np.mean(y_predicted == y)