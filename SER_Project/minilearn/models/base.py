from abc import ABC, abstractmethod
import numpy as np

class Classifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def score(self, X, y):
        pass
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    