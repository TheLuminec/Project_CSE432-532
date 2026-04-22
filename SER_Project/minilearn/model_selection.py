import numpy as np


class StratifiedKFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y):
        pass

    def get_n_splits(self):
        return self.n_splits


class GridSearchCV:
    def __init__(self, estimator, param_grid, cv=None, scoring=None):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def score(self, X, y):
        pass

    def get_best_params(self):
        pass

    def get_best_score(self):
        pass

    def get_best_estimator(self):
        pass

    def get_cv_results(self):
        pass
