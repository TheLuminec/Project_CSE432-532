from minilearn.models.base import Classifier
import numpy as np
from typing import Literal


class SVM(Classifier):
    def __init__(self, kernel='linear', learning_rate=0.01, n_iterations=100, lambda_param=0.01):
        self.kernel = kernel
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_param = lambda_param
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.n_features_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, self.n_features_ = X.shape

        self.classes_ = np.unique(y)

        if len(self.classes_) != 2:
            raise ValueError("SVM is only implemented for binary classification.")
        if self.kernel != 'linear':
            raise ValueError("This SVM currently supports only the linear kernel.")

        self.coef_ = np.zeros(self.n_features_)
        self.intercept_ = 0.0

        y_transformed = np.where(y == self.classes_[0], -1, 1)

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                margin = y_transformed[idx] * (np.dot(x_i, self.coef_) + self.intercept_)

                if margin >= 1:
                    self.coef_ -= self.learning_rate * (2 * self.lambda_param * self.coef_)
                else:
                    self.coef_ -= self.learning_rate * (2 * self.lambda_param * self.coef_ - y_transformed[idx] * x_i)
                    self.intercept_ += self.learning_rate * y_transformed[idx]

    def predict(self, X):
        X = np.array(X)
        linear_output = np.dot(X, self.coef_) + self.intercept_
        y_predicted = np.where(linear_output >= 0, self.classes_[1], self.classes_[0])
        return y_predicted

    def score(self, X, y):
        y_predicted = self.predict(X)
        return np.mean(y_predicted == y)
