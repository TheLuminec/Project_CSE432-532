from minilearn.models.base import Classifier
import numpy as np

class GaussianNaiveBayes(Classifier):
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = None
        self.n_features_ = None
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y) 
        n_samples, self.n_features_ = X.shape

        self.classes_ = np.unique(y)
        self.mean_ = np.zeros((len(self.classes_), self.n_features_))
        self.var_ = np.zeros((len(self.classes_), self.n_features_))

        for i, cls in enumerate(self.classes_):
            X_cls = X[y == cls]
            self.mean_[i] = np.mean(X_cls, axis=0)
            self.var_[i] = np.var(X_cls, axis=0)

    def predict(self, X):
        X = np.array(X)
        y_predicted = []

        for x in X:
            posteriors = []
            for i, cls in enumerate(self.classes_):
                prior = 1 / len(self.classes_)
                log_likelihood = np.sum(np.log(self._gaussian_likelihood(i, x) + self.var_smoothing))
                posterior = prior * log_likelihood
                posteriors.append(posterior)

            y_predicted.append(self.classes_[np.argmax(posteriors)])

        return np.array(y_predicted)
    
    def _gaussian_likelihood(self, class_idx, x):
        mean = self.mean_[class_idx]
        var = self.var_[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)