from minilearn.models.base import Classifier
import numpy as np

class KNN(Classifier):
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.classes_ = None
        self.n_features_ = None
        self.data_ = None
        self.labels_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        
        self.classes_ = np.unique(y)
        self.n_features_ = n_features
        self.data_ = X
        self.labels_ = y

    def predict(self, X):
        X = np.array(X)
        y_predicted = []
        
        for x in X:
            distances = self._dist_from_data(x)
            neighbor_indices = np.argsort(distances)[:self.n_neighbors]
            neighbor_labels = self.labels_[neighbor_indices]
            most_common = np.bincount(neighbor_labels).argmax()
            y_predicted.append(most_common)
        
        return np.array(y_predicted)
    
    def _dist_from_data(self, x):
        return np.sqrt(np.sum((self.data_ - x) ** 2, axis=1))

    def score(self, X, y):
        y_predicted = self.predict(X)
        return np.mean(y_predicted == y)