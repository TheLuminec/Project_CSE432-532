import numpy as np
import torch.nn as nn
import torch

from minilearn.models.base import Classifier
from minilearn.metrics import accuracy_score


class ANN(nn.Module, Classifier):
    def __init__(self, 
                 layers: list[nn.Module],
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 loss_fn: torch.nn.Module = nn.CrossEntropyLoss,
                 metric: callable = accuracy_score,
                 epochs: int = 10,
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 random_state: int = 42):
        """
        Args:
            layers (list[nn.Module]): List of layers in the network.
            optimizer (torch.optim.Optimizer): Optimizer to use for training.
            loss_fn (torch.nn.Module): Loss function to use for training.
            metric (callable): Metric to use for evaluation.
            epochs (int): Number of epochs to train for.
            batch_size (int): Batch size to use for training.
            learning_rate (float): Learning rate to use for training.
        """
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.loss_fn = loss_fn()
        self.metric = metric
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        torch.random.manual_seed(random_state)

    def _transform_labels(self, y):
        """Transform labels to be 0-indexed."""
        y = np.asarray(y)

        unique_labels = np.unique(y)
        if np.min(unique_labels) != 0 or np.max(unique_labels) != len(unique_labels) - 1:
            label_to_int = {label: i for i, label in enumerate(unique_labels)}
            y = np.array([label_to_int[label] for label in y], dtype=np.int64)
        return y


    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        X = torch.tensor(X, dtype=torch.float32)
        y = self._transform_labels(y)
        y = torch.tensor(y, dtype=torch.long)

        self.train()
        for epoch in range(self.epochs):
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i+self.batch_size]
                batch_y = y[i:i+self.batch_size]
                self.optimizer.zero_grad()
                y_pred = self.forward(batch_X)
                loss = self.loss_fn(y_pred, batch_y)
                loss.backward()
                self.optimizer.step()
        
    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        y_pred = []
        self.eval()
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch_X = torch.tensor(X[i:i+self.batch_size], dtype=torch.float32)
                y_pred.extend(np.argmax(self.forward(batch_X).detach().numpy(), axis=1))
        return np.array(y_pred)

    def score(self, X, y):
        y = self._transform_labels(y)
        return self.metric(y, self.predict(X))

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

class DenseANN(ANN):
    def __init__(self, input_dim: int, hidden_layers: list[int], num_classes: int,
                 activation = nn.ReLU,
                 optimizer: torch.optim.Optimizer = torch.optim.Adam,
                 loss_fn: torch.nn.Module = nn.CrossEntropyLoss,
                 metric: callable = accuracy_score,
                 epochs: int = 10,
                 batch_size: int = 32,
                 learning_rate: float = 0.001):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(activation())
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(activation())
        layers.append(nn.Linear(hidden_layers[-1], num_classes))
        super().__init__(layers, optimizer, loss_fn, metric, epochs, batch_size, learning_rate)


if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    X, y = load_digits(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    layers = [nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 10)]
    ann = ANN(layers)
    ann.fit(X_train, y_train)
    print("Predict:",ann.predict(X_test))
    print("Score:",ann.score(X_test, y_test))
