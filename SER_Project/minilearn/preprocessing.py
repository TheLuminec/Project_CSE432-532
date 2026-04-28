import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split arrays or matrices into random train and test subsets.
    """
    if random_state is not None:
        np.random.seed(random_state)

    X = np.array(X)
    y = np.array(y)

    indices = np.random.permutation(len(X))
    split_index = int(test_size * len(X))
    train_indices = indices[split_index:]
    test_indices = indices[:split_index]
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)

    def transform(self, X):
        X_transformed = X - self.mean_
        scale = np.where(self.scale_ == 0, 1.0, self.scale_)
        X_transformed /= scale
        return X_transformed

    def fit_transform(self, X):
        X = np.array(X)
        self.fit(X)
        return self.transform(X)