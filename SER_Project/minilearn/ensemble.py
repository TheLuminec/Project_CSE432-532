from minilearn.models.base import Classifier
from minilearn.metrics import accuracy_score
import copy
import numpy as np

class VotingClassifier(Classifier):
    def __init__(self, estimator: Classifier, n_estimators: int=5):
        self.estimators = []
        self.n_estimators = n_estimators

        for _ in range(n_estimators):
            est = copy.deepcopy(estimator)
            self.estimators.append(est)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        for e in self.estimators:
            indices = np.random.choice(len(X), size=len(X), replace=True)
            e.fit(X[indices], y[indices])

    def predict(self, X):
        X = np.asarray(X)

        predictions = np.asarray([e.predict(X) for e in self.estimators])

        y_pred = []
        for sample_preds in predictions.T:
            labels, counts = np.unique(sample_preds, return_counts=True)
            y_pred.append(labels[np.argmax(counts)])

        return np.asarray(y_pred)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
    

class AdaBoostClassifier(Classifier):
    def __init__(self, estimator: Classifier, n_estimators: int=5, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = len(X)
    
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        sample_weights = np.full(n_samples, 1 / n_samples, dtype=float)
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []

        max_error = 1 - (1 / n_classes)
        eps = 1e-12

        for _ in range(self.n_estimators):
            estimator = copy.deepcopy(self.estimator)
            bootstrap_indices = rng.choice(n_samples, size=n_samples, replace=True, p=sample_weights)
            estimator.fit(X[bootstrap_indices], y[bootstrap_indices])

            y_pred = estimator.predict(X)
            incorrect = y_pred != y
            error = float(np.dot(sample_weights, incorrect))
            error = min(max(error, eps), 1 - eps)

            if error >= max_error:
                continue

            if n_classes == 2:
                estimator_weight = np.log((1 - error) / error)
            else:
                estimator_weight = np.log((1 - error) / error) + np.log(n_classes - 1)

            self.estimators_.append(estimator)
            self.estimator_weights_.append(float(estimator_weight))
            self.estimator_errors_.append(error)

            if error <= eps:
                break

            sample_weights *= np.exp(estimator_weight * incorrect.astype(float))
            sample_weights_sum = sample_weights.sum()
            if sample_weights_sum <= 0:
                break
            sample_weights /= sample_weights_sum

        self.estimator_weights_ = np.asarray(self.estimator_weights_, dtype=float)
        self.estimator_errors_ = np.asarray(self.estimator_errors_, dtype=float)
        return self

    def predict(self, X):
        X = np.asarray(X)

        class_scores = np.zeros((len(X), len(self.classes_)), dtype=float)
        for estimator, estimator_weight in zip(self.estimators_, self.estimator_weights_):
            predictions = estimator.predict(X)
            for class_index, class_label in enumerate(self.classes_):
                class_scores[:, class_index] += estimator_weight * (predictions == class_label)

        return self.classes_[np.argmax(class_scores, axis=1)]

    def score(self, X, y):
        return accuracy_score(np.asarray(y), self.predict(X))

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from minilearn.preprocessing import train_test_split

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from minilearn.classifiers import DecisionTreeClassifier

    base_estimator = DecisionTreeClassifier(max_depth=3)
    ensemble = VotingClassifier(estimator=base_estimator, n_estimators=10)
    ensemble.fit(X_train, y_train)
    print("Ensemble Accuracy:", ensemble.score(X_test, y_test))

    stump = DecisionTreeClassifier(max_depth=1)
    boost = AdaBoostClassifier(estimator=stump, n_estimators=10, random_state=42)
    boost.fit(X_train, y_train)
    print("AdaBoost Accuracy:", boost.score(X_test, y_test))
