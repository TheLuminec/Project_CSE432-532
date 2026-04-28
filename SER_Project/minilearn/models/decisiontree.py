from minilearn.models.base import Classifier
import numpy as np

class DecisionTreeClassifier(Classifier):
    class Node:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

        def forward(self, x):
            if self.value is not None:
                return self.value
            if x[self.feature_index] <= self.threshold:
                return self.left.forward(x)
            else:
                return self.right.forward(x)
            
    def __init__(self, criterion="gini", max_depth=4, min_samples_split=2, min_samples_leaf=1):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        
        self.tree_ = self._build_tree(X, y_idx, depth=0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            leaf_value = np.bincount(y).argmax()
            return self.Node(value=leaf_value)
        
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            leaf_value = np.bincount(y).argmax()
            return self.Node(value=leaf_value)
        
        left_indices = np.where(X[:, best_feature] <= best_threshold)[0]
        right_indices = np.where(X[:, best_feature] > best_threshold)[0]
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        return self.Node(feature_index=best_feature, threshold=best_threshold, left=left_subtree, right=right_subtree)

    def _information_gain(self, X, y, feature_index, threshold):
        parent_loss = self._loss(y)
        
        left_indices = np.where(X[:, feature_index] <= threshold)[0]
        right_indices = np.where(X[:, feature_index] > threshold)[0]
        
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0
        
        left_loss = self._loss(y[left_indices])
        right_loss = self._loss(y[right_indices])
        
        n_left = len(left_indices)
        n_right = len(right_indices)
        n_total = n_left + n_right
        
        weighted_loss = (n_left / n_total) * left_loss + (n_right / n_total) * right_loss
        
        gain = parent_loss - weighted_loss
        return gain
    
    def _loss(self, y):
        if self.criterion == "gini":
            return 1 - np.sum((np.bincount(y) / len(y)) ** 2)
        elif self.criterion == "entropy":
            probabilities = np.bincount(y) / len(y)
            return -np.sum(probabilities * np.log2(probabilities + 1e-10))
        else:
            raise ValueError("Criterion must be 'gini' or 'entropy'.")

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature_index in range(self.n_features_):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature_index, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold
                    
        return best_feature, best_threshold

    def predict(self, X):
        X = np.array(X)
        return np.array([self.classes_[self.tree_.forward(x)] for x in X])

    def score(self, X, y):
        y = np.array(y)
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    