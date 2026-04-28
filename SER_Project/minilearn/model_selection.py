import numpy as np
from typing import Collection
from itertools import product
from abc import ABC, abstractmethod

from minilearn.models.base import Classifier
from minilearn.classifiers import SVM
from minilearn.metrics import accuracy_score

class Split(ABC):
    """Split base class."""
    def __init__(self, n_splits=5, random_state=None):
        self.random_state = random_state
        self.n_splits = n_splits

    @abstractmethod
    def split(self, X, y):
        pass

    def get_n_splits(self):
        return self.n_splits

class CV(Classifier, ABC):
    """Cross-validation base class."""
    @abstractmethod
    def get_cv_results_(self):
        pass

class KFold(Split):
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits, random_state)
        self.shuffle = shuffle

    def split(self, X, y):
        classes, counts = np.unique(y, return_counts=True)
        self.n_classes = len(classes)
        
        base_n_per_class = counts // self.n_splits
        remainder = counts % self.n_splits
        
        folds = [[] for _ in range(self.n_splits)]
        for i, cls in enumerate(classes):
            cls_indices = np.where(y == cls)[0]

            for fold_idx in range(self.n_splits):
                start = fold_idx * base_n_per_class[i]
                end = start + base_n_per_class[i]
                folds[fold_idx].extend(cls_indices[start:end])
        
        # Distribute remainder samples one by one to folds
        for cls_idx, cls in enumerate(classes):
            remainder_count = remainder[cls_idx]
            cls_indices = np.where(y == cls)[0]
            
            start = (self.n_splits * base_n_per_class[cls_idx])
            remaining_cls_indices = cls_indices[start:]
            for i in range(remainder_count):
                fold_idx = i
                folds[fold_idx].append(remaining_cls_indices[i])
        
        train_test_splits = []
        for fold in folds:
            fold_indices = np.array(fold, dtype=int)
            if len(fold_indices) == 0:
                continue
            train_indices = fold_indices
            test_indices = np.setdiff1d(np.arange(len(y)), fold_indices)
            train_test_splits.append((train_indices, test_indices))

        return train_test_splits

class StratifiedKFold(Split):
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits, random_state)
        self.shuffle = shuffle

    def split(self, X, y):
        classes, counts = np.unique(y, return_counts=True)
        self.n_classes = len(classes)
        
        base_n_per_class = counts // self.n_splits
        remainder = counts % self.n_splits
        
        folds = [[] for _ in range(self.n_splits)]
        for i, cls in enumerate(classes):
            cls_indices = np.where(y == cls)[0]

            for fold_idx in range(self.n_splits):
                start = fold_idx * base_n_per_class[i]
                end = start + base_n_per_class[i]
                folds[fold_idx].extend(cls_indices[start:end])
        
        # Distribute remainder samples one by one to folds
        for cls_idx, cls in enumerate(classes):
            remainder_count = remainder[cls_idx]
            cls_indices = np.where(y == cls)[0]
            
            start = (self.n_splits * base_n_per_class[cls_idx])
            remaining_cls_indices = cls_indices[start:]
            for i in range(remainder_count):
                fold_idx = i
                folds[fold_idx].append(remaining_cls_indices[i])
        
        train_test_splits = []
        for fold in folds:
            fold_indices = np.array(fold, dtype=int)
            if len(fold_indices) == 0:
                continue
            train_indices = fold_indices
            test_indices = np.setdiff1d(np.arange(len(y)), fold_indices)
            train_test_splits.append((train_indices, test_indices))

        return train_test_splits


class GridSearchCV(CV):
    def __init__(self, estimator: Classifier, param_grid: dict, cv: Split = None, scoring=accuracy_score):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring

        if self.cv is None:
            self.cv = StratifiedKFold(n_splits=5)
        
        self.n_params = len(self.param_grid)
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.cv_results_ = None

    def _extract_parameters(self):
        default_params = {}
        params = {}
        for key, values in self.param_grid.items():
            if not isinstance(values, Collection):
                default_params[key] = values
            else:
                params[key] = values

        param_list = [dict(zip(params.keys(), v)) for v in product(*params.values())]
        param_list = [{**default_params, **param} for param in param_list]
        return param_list

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        train_test_split = self.cv.split(X, y)

        estimators = []
        self.cv_results_ = []
        param_list = self._extract_parameters()
        for param in param_list:
            for train_indices, test_indices in train_test_split:
                X_train, X_test = X[train_indices], X[test_indices]
                y_train, y_test = y[train_indices], y[test_indices]
                
                estimators.append(self.estimator(**param))
                estimators[-1].fit(X_train, y_train)
                y_pred = estimators[-1].predict(X_test)
                score = self.scoring(y_test, y_pred)
                self.cv_results_.append({
                    "params": param,
                    "mean_test_score": score
                })
        
        self.best_params_ = max(self.cv_results_, key=lambda x: x["mean_test_score"])["params"]
        self.best_score_ = max(self.cv_results_, key=lambda x: x["mean_test_score"])["mean_test_score"]
        self.best_estimator_ = self.estimator(**self.best_params_)
        self.best_estimator_.fit(X, y)
        
    def predict(self, X):
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        return self.best_estimator_.score(X, y)

    def get_cv_results_(self):
        return self.cv_results_

class RandomizedSearchCV(CV):
    def __init__(self, estimator: Classifier, param_distributions: dict, n_iter=10, cv: Split = None, scoring=accuracy_score):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring

        if self.cv is None:
            self.cv = StratifiedKFold(n_splits=5)

        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.cv_results_ = None

    def _extract_parameters(self):
        param = {}
        for key, value in self.param_distributions.items():
            if not isinstance(value, Collection):
                param[key] = value
            else:
                param[key] = np.random.choice(value)
        return param
    
    def fit(self, X, y):
        train_test_split = self.cv.split(X, y)

        estimators = []
        self.cv_results_ = []
        for train_indices, test_indices in train_test_split:
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            
            param = self._extract_parameters()
            estimators.append(self.estimator(**param))
            estimators[-1].fit(X_train, y_train)
            y_pred = estimators[-1].predict(X_test)
            score = self.scoring(y_test, y_pred)
            self.cv_results_.append({
                "params": param,
                "mean_test_score": score
            })
        
        self.best_params_ = max(self.cv_results_, key=lambda x: x["mean_test_score"])["params"]
        self.best_score_ = max(self.cv_results_, key=lambda x: x["mean_test_score"])["mean_test_score"]
        self.best_estimator_ = self.estimator(**self.best_params_)
        self.best_estimator_.fit(X, y)

    def predict(self, X):
        return self.best_estimator_.predict(X)

    def score(self, X, y):
        return self.best_estimator_.score(X, y)

    def get_cv_results_(self):
        return self.cv_results_

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)
    param_grid = {
        "learning_rate": [0.01, 0.001, 0.0001],
        "kernel": ["linear", "poly", "rbf"],
        "degree": [2, 3, 4],
        "coef0": [1, 2, 3],
        "tolerance": [1e-3, 1e-4, 1e-5],
        "random_state": 42
    }
    param_dist = {
        "learning_rate": [0.01, 0.001, 0.0001],
        "kernel": ["linear", "poly", "rbf"],
        "degree": [2, 3, 4],
        "coef0": [1, 2, 3],
        "tolerance": [1e-3, 1e-4, 1e-5],
        "random_state": 42
    }
    randomized_search = RandomizedSearchCV(estimator=SVM, param_distributions=param_dist, n_iter=10)
    randomized_search.fit(X, y)
    print(randomized_search.best_params_)
    print(randomized_search.best_score_)
    print(randomized_search.best_estimator_)
    
    grid_search = GridSearchCV(estimator=SVM, param_grid=param_grid)
    grid_search.fit(X, y)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    print(grid_search.best_estimator_)
    #print(grid_search.cv_results_)