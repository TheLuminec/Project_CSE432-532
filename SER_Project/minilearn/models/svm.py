from minilearn.models.base import Classifier
import numpy as np


class SVM(Classifier):
    def __init__(
        self,
        kernel='linear',
        learning_rate=0.01,
        n_iterations=100,
        lambda_param=0.01,
        degree=3,
        gamma='scale',
        coef0=1.0,
        tolerance=1e-3,
        random_state=42,
    ):
        kernel = kernel.lower()
        self.kernel_types = ('linear', 'rbf', 'poly', 'polynomial')
        if kernel not in self.kernel_types:
            raise ValueError("Supported kernels are 'linear', 'rbf', 'poly', and 'polynomial'.")
        
        self.kernel = kernel
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.lambda_param = lambda_param
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tolerance = tolerance
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = None
        self.n_features_ = None
        self.alphas_ = None
        self.support_vectors_ = None
        self.support_vector_labels_ = None
        self.gamma_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, self.n_features_ = X.shape

        self.classes_ = np.unique(y)

        if self.kernel in ('poly', 'polynomial') and self.degree <= 0:
            raise ValueError("degree must be positive for the polynomial kernel.")

        if len(self.classes_) > 2:
            self._is_multiclass = True
            self.models_ = []
            for c in self.classes_:
                y_binary = np.where(y == c, 1, -1)
                model = SVM(kernel=self.kernel, learning_rate=self.learning_rate,
                            n_iterations=self.n_iterations, lambda_param=self.lambda_param,
                            degree=self.degree, gamma=self.gamma, coef0=self.coef0,
                            tolerance=self.tolerance, random_state=self.random_state)
                model.classes_ = np.array([-1, 1])
                model.n_features_ = self.n_features_
                if self.kernel == 'linear':
                    model._fit_linear(X, y_binary)
                else:
                    model._fit_kernelized(X, y_binary)
                self.models_.append(model)
            return

        self._is_multiclass = False
        y_transformed = np.where(y == self.classes_[0], -1, 1)

        if self.kernel == 'linear':
            self._fit_linear(X, y_transformed)
        else:
            self._fit_kernelized(X, y_transformed)

    def _fit_linear(self, X, y):
        self.coef_ = np.zeros(self.n_features_)
        self.intercept_ = 0.0
        self.alphas_ = None
        self.support_vectors_ = None
        self.support_vector_labels_ = None

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                margin = y[idx] * (np.dot(x_i, self.coef_) + self.intercept_)

                if margin >= 1:
                    self.coef_ -= self.learning_rate * (2 * self.lambda_param * self.coef_)
                else:
                    self.coef_ -= self.learning_rate * (2 * self.lambda_param * self.coef_ - y[idx] * x_i)
                    self.intercept_ += self.learning_rate * y[idx]

    def _fit_kernelized(self, X, y):
        self.gamma_ = self._resolve_gamma(X)
        kernel_matrix = self._kernel(X, X)
        c = self._regularization_strength()
        rng = np.random.default_rng(self.random_state)

        alphas = np.zeros(len(X))
        self.intercept_ = 0.0

        for _ in range(self.n_iterations):
            num_changed = 0

            for i in range(len(X)):
                error_i = self._decision_from_kernel_column(kernel_matrix[:, i], alphas, y) - y[i]

                violates_lower_bound = y[i] * error_i < -self.tolerance and alphas[i] < c
                violates_upper_bound = y[i] * error_i > self.tolerance and alphas[i] > 0
                if not (violates_lower_bound or violates_upper_bound):
                    continue

                j = self._select_second_alpha(i, y, rng)
                error_j = self._decision_from_kernel_column(kernel_matrix[:, j], alphas, y) - y[j]

                old_alpha_i = alphas[i]
                old_alpha_j = alphas[j]

                if y[i] != y[j]:
                    lower = max(0.0, old_alpha_j - old_alpha_i)
                    upper = min(c, c + old_alpha_j - old_alpha_i)
                else:
                    lower = max(0.0, old_alpha_i + old_alpha_j - c)
                    upper = min(c, old_alpha_i + old_alpha_j)

                if lower == upper:
                    continue

                eta = 2 * kernel_matrix[i, j] - kernel_matrix[i, i] - kernel_matrix[j, j]
                if eta >= 0:
                    continue

                alphas[j] -= y[j] * (error_i - error_j) / eta
                alphas[j] = np.clip(alphas[j], lower, upper)

                if abs(alphas[j] - old_alpha_j) < 1e-5:
                    continue

                alphas[i] += y[i] * y[j] * (old_alpha_j - alphas[j])

                b1 = (
                    self.intercept_
                    - error_i
                    - y[i] * (alphas[i] - old_alpha_i) * kernel_matrix[i, i]
                    - y[j] * (alphas[j] - old_alpha_j) * kernel_matrix[i, j]
                )
                b2 = (
                    self.intercept_
                    - error_j
                    - y[i] * (alphas[i] - old_alpha_i) * kernel_matrix[i, j]
                    - y[j] * (alphas[j] - old_alpha_j) * kernel_matrix[j, j]
                )

                if 0 < alphas[i] < c:
                    self.intercept_ = b1
                elif 0 < alphas[j] < c:
                    self.intercept_ = b2
                else:
                    self.intercept_ = (b1 + b2) / 2

                num_changed += 1

            if num_changed == 0:
                break

        support_vector_mask = alphas > 1e-5
        self.alphas_ = alphas[support_vector_mask]
        self.support_vectors_ = X[support_vector_mask]
        self.support_vector_labels_ = y[support_vector_mask]
        self.coef_ = None

    def _regularization_strength(self):
        if self.lambda_param <= 0:
            return 1.0
        return 1 / (2 * self.lambda_param)

    def _resolve_gamma(self, X):
        if isinstance(self.gamma, str):
            if self.gamma == 'scale':
                variance = X.var()
                return 1 / (self.n_features_ * variance) if variance > 0 else 1.0
            if self.gamma == 'auto':
                return 1 / self.n_features_
            raise ValueError("gamma must be positive, 'scale', or 'auto'.")

        if self.gamma <= 0:
            raise ValueError("gamma must be positive, 'scale', or 'auto'.")
        return float(self.gamma)

    def _kernel(self, X, Y):
        if self.kernel == 'linear':
            return np.dot(X, Y.T)
        if self.kernel == 'rbf':
            x_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
            y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
            distances = x_norm + y_norm - 2 * np.dot(X, Y.T)
            distances = np.maximum(distances, 0)
            return np.exp(-self.gamma_ * distances)
        if self.kernel in ('poly', 'polynomial'):
            return (self.gamma_ * np.dot(X, Y.T) + self.coef0) ** self.degree

        raise ValueError("Supported kernels are 'linear', 'rbf', 'poly', and 'polynomial'.")

    def _decision_from_kernel_column(self, kernel_column, alphas, y):
        return np.sum(alphas * y * kernel_column) + self.intercept_

    def _select_second_alpha(self, first_index, y, rng):
        candidates = np.where(y != y[first_index])[0]
        if len(candidates) == 0:
            candidates = np.delete(np.arange(len(y)), first_index)
        return rng.choice(candidates)

    def _decision_function(self, X):
        if self.kernel == 'linear':
            return np.dot(X, self.coef_) + self.intercept_

        if len(self.support_vectors_) == 0:
            return np.full(X.shape[0], self.intercept_)

        kernel_values = self._kernel(X, self.support_vectors_)
        return np.dot(kernel_values, self.alphas_ * self.support_vector_labels_) + self.intercept_

    def predict(self, X):
        X = np.array(X)
        if getattr(self, '_is_multiclass', False):
            decisions = np.column_stack([model._decision_function(X) for model in self.models_])
            return self.classes_[np.argmax(decisions, axis=1)]
            
        output = self._decision_function(X)
        y_predicted = np.where(output >= 0, self.classes_[1], self.classes_[0])
        return y_predicted

    def score(self, X, y):
        y_predicted = self.predict(X)
        return np.mean(y_predicted == y)
