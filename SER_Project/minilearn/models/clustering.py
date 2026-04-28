import numpy as np
import matplotlib.pyplot as plt

from minilearn.models.base import Classifier

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

    def score(self, X, y):
        return adjusted_rand_index(y, self.predict(X))
    
    def get_cluster_centers(self):
        return self.cluster_centers_


def adjusted_rand_index(labels_true, labels_pred):
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    contingency = np.zeros((classes.size, clusters.size), dtype=int)
    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            contingency[i, j] = np.sum((labels_true == c) & (labels_pred == k))
    
    def comb2(n): return n * (n - 1) / 2
    
    sum_comb_c = sum(comb2(n_c) for n_c in contingency.sum(axis=1))
    sum_comb_k = sum(comb2(n_k) for n_k in contingency.sum(axis=0))
    sum_comb = sum(comb2(n_ij) for n_ij in contingency.flatten())
    
    n = labels_true.size
    expected_comb = (sum_comb_c * sum_comb_k) / comb2(n) if comb2(n) != 0 else 0
    max_comb = (sum_comb_c + sum_comb_k) / 2
    
    if max_comb == expected_comb:
        return 1.0
    return (sum_comb - expected_comb) / (max_comb - expected_comb)


def normalized_mutual_info_score(labels_true, labels_pred):
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    n = labels_true.size
    
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    
    entropy_true = 0
    for c in classes:
        p = np.sum(labels_true == c) / n
        if p > 0: entropy_true -= p * np.log(p)
        
    entropy_pred = 0
    for k in clusters:
        p = np.sum(labels_pred == k) / n
        if p > 0: entropy_pred -= p * np.log(p)
        
    mi = 0
    for c in classes:
        for k in clusters:
            p_ij = np.sum((labels_true == c) & (labels_pred == k)) / n
            if p_ij > 0:
                p_i = np.sum(labels_true == c) / n
                p_j = np.sum(labels_pred == k) / n
                mi += p_ij * np.log(p_ij / (p_i * p_j))
                
    if entropy_true + entropy_pred == 0:
        return 1.0
    return 2 * mi / (entropy_true + entropy_pred)


def pca(X, n_components=2):
    X = np.asarray(X)
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    if cov_matrix.ndim == 0:
        cov_matrix = np.array([[cov_matrix]])
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    top_eigenvectors = eigenvectors[:, :n_components]
    return np.dot(X_centered, top_eigenvectors)


def visualize_clusters(X, labels):        
    X_trans = pca(X, n_components=2)
        
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        plt.scatter(X_trans[labels == label, 0], X_trans[labels == label, 1], label=f'Cluster {label}')
    plt.title("Cluster Visualization using PCA")
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    kmeans = KMeans(3)
    kmeans.fit(X)

    y_pred = kmeans.predict(X)
    print(adjusted_rand_index(y, y_pred))
    print(normalized_mutual_info_score(y, y_pred))
    visualize_clusters(X, y_pred)