import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..')))

from minilearn.models.base import Classifier
from minilearn.metrics import accuracy_score

class KMeans(Classifier):
    def __init__(self, n_clusters=8, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels_ = None
        self.cluster_centers_ = None
        
    def fit(self, X, y=None):
        np.random.seed(self.random_state)
        X = np.array(X)
        if y is not None:
            y_unique = np.unique(np.array(y))
            self.labels_ = {label: i for i, label in enumerate(y_unique)}
            self.n_clusters = len(self.labels_)
        else:
            self.labels_ = np.arange(self.n_clusters)

        # choose centers
        if y is None:
            idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            self.cluster_centers_ = X[idx]
        else:
            self.cluster_centers_ = np.array([X[y == label].mean(axis=0) for label in self.labels_])

        # train
        for _ in range(self.max_iter):
            # assign labels
            y_pred = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2), axis=1)
            
            # update centers
            for i in range(self.n_clusters):
                self.cluster_centers_[i] = np.mean(X[y_pred == i], axis=0)

    def predict(self, X):
        X = np.array(X)
        return np.argmin(np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2), axis=1)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.predict(X)

    def predict_proba(self, X):
        X = np.array(X)
        return np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)

    def score(self, X, y=None):
        if y is None:
            return np.sum(self.predict(X) == self.labels_) / len(self.predict(X))
        return accuracy_score(y, self.predict(X))
    
    def get_cluster_centers(self):
        return self.cluster_centers_


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    kmeans = KMeans(3)
    kmeans.fit(X, y)
    # kmeans.fit(X)

    print(kmeans.labels_)
    print(kmeans.predict(X))
    print(kmeans.predict_proba(X))
    print(kmeans.score(X, y))
    print(kmeans.get_cluster_centers())