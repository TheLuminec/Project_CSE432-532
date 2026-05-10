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