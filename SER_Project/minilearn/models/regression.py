from minilearn.models.base import Classifier
import numpy as np

class LogisticRegression(Classifier):
    def __init__(self, learning_rate=0.01, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.n_features_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0
        self.classes_ = np.unique(y)
        self.n_features_ = n_features

        if len(self.classes_) != 2:
            raise ValueError("Logistic Regression is only implemented for binary classification.")

        for _ in range(self.n_iterations):
            y_predicted = self.predict(X)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.coef_) + self.intercept_
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = np.where(y_predicted >= 0.5, self.classes_[1], self.classes_[0])
        return y_predicted_cls

    def score(self, X, y):
        y_predicted = self.predict(X)
        return np.mean((y_predicted - y) ** 2)