import numpy as np
from typing import Collection
from itertools import product
from abc import ABC, abstractmethod

from minilearn.models.base import Classifier
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
        n_samples = len(X)
        indices = np.arange(n_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.random_state)
            rng.shuffle(indices)

        folds = np.array_split(indices, self.n_splits)
        train_test_splits = []
        for fold in folds:
            test_indices = np.array(fold, dtype=int)
            if len(test_indices) == 0:
                continue
            train_indices = np.setdiff1d(indices, test_indices, assume_unique=False)
            train_test_splits.append((train_indices, test_indices))

        return train_test_splits

class StratifiedKFold(Split):
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        super().__init__(n_splits, random_state)
        self.shuffle = shuffle

    def split(self, X, y):
        y = np.asarray(y)
        classes = np.unique(y)
        folds = [[] for _ in range(self.n_splits)]
        rng = np.random.default_rng(self.random_state)

        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            if self.shuffle:
                rng.shuffle(cls_indices)
            cls_folds = np.array_split(cls_indices, self.n_splits)
            for fold_idx, cls_fold in enumerate(cls_folds):
                folds[fold_idx].extend(cls_fold.tolist())

        all_indices = np.arange(len(y))
        train_test_splits = []
        for fold in folds:
            test_indices = np.array(fold, dtype=int)
            if len(test_indices) == 0:
                continue
            train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=False)
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
            if not isinstance(values, Collection) or isinstance(values, str):
                default_params[key] = values
            else:
                params[key] = values

        param_list = [dict(zip(params.keys(), v)) for v in product(*params.values())]
        param_list = [{**default_params, **param} for param in param_list]
        return param_list

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        train_test_splits = self.cv.split(X, y)

        self.cv_results_ = []
        param_list = self._extract_parameters()
        for param in param_list:
            scores = []
            for train_indices, test_indices in train_test_splits:
                X_train, X_test = X[train_indices], X[test_indices]
                y_train, y_test = y[train_indices], y[test_indices]

                model = self.estimator(**param)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                scores.append(self.scoring(y_test, y_pred))

            self.cv_results_.append({
                "params": param,
                "mean_test_score": float(np.mean(scores)),
                "split_test_scores": scores,
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
    def __init__(self, estimator: Classifier, param_distributions: dict, n_iter=10, cv: Split = None, scoring=accuracy_score, random_state=None):
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state

        if self.cv is None:
            self.cv = StratifiedKFold(n_splits=5)

        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.cv_results_ = None
        self._rng = np.random.default_rng(self.random_state)

    def _extract_parameters(self):
        param = {}
        for key, value in self.param_distributions.items():
            if not isinstance(value, Collection) or isinstance(value, str):
                param[key] = value
            else:
                choices = list(value)
                param[key] = choices[self._rng.integers(len(choices))]
        return param
    
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        train_test_splits = self.cv.split(X, y)

        self.cv_results_ = []
        for _ in range(self.n_iter):
            param = self._extract_parameters()
            scores = []
            for train_indices, test_indices in train_test_splits:
                X_train, X_test = X[train_indices], X[test_indices]
                y_train, y_test = y[train_indices], y[test_indices]
                
                model = self.estimator(**param)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                scores.append(self.scoring(y_test, y_pred))
                
            self.cv_results_.append({
                "params": param,
                "mean_test_score": np.mean(scores)
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
    from minilearn.classifiers import SVM
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
