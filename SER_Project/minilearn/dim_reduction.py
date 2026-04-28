import numpy as np

def pca(X, n_components=2):
    """
    Performs Principal Component Analysis (PCA) for dimensionality reduction.
    
    Args:
        X: np.array of shape (n_samples, n_features)
        n_components: int, number of components to reduce to
    Returns:
        np.array of shape (n_samples, n_components)
    """
    X = np.asarray(X)
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    if cov_matrix.ndim == 0:
        cov_matrix = np.array([[cov_matrix]])
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    top_eigenvectors = eigenvectors[:, :n_components]
    return np.dot(X_centered, top_eigenvectors)

