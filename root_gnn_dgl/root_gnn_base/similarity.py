import numpy as np
import scipy
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

from scipy.stats import wasserstein_distance

def cka(rep_a, rep_b, size=None):
    """
    Computes the Centered Kernel Alignment (CKA) between two large representation matrices rep_a and rep_b.
    If size is provided, it performs CKA on a randomly selected subset of the data.
    
    Parameters:
    rep_a : np.ndarray
        First representation matrix of size (n_samples, n_features_a).
    rep_b : np.ndarray
        Second representation matrix of size (n_samples, n_features_b).
    size : int, optional
        Number of samples to use for the CKA calculation. If None, use the full dataset.
        
    Returns:
    float
        CKA similarity between rep_a and rep_b.
    """
    
    def gram_linear(x):
        """Compute the Gram (kernel) matrix using a linear kernel."""
        return x @ x.T
    
    def center_gram(gram):
        """Center the Gram matrix."""
        n = gram.shape[0]
        identity = np.eye(n)
        ones = np.ones((n, n)) / n
        return gram - ones @ gram - gram @ ones + ones @ gram @ ones
    
    # If sample_size is specified, randomly sample a subset of the data
    if size is not None and size < rep_a.shape[0]:
        indices = np.random.choice(rep_a.shape[0], size, replace=False)
        rep_a = rep_a[indices]
        rep_b = rep_b[indices]
    
    # Compute the Gram matrices
    gram_a = gram_linear(rep_a)
    gram_b = gram_linear(rep_b)

    # Center the Gram matrices
    centered_gram_a = center_gram(gram_a)
    centered_gram_b = center_gram(gram_b)
    
    # Compute the CKA similarity
    numerator = np.sum(centered_gram_a * centered_gram_b)
    denominator = np.sqrt(np.sum(centered_gram_a**2) * np.sum(centered_gram_b**2))
    
    return numerator / denominator if denominator != 0 else 0

def cca(X, Y, size = None, num_components=10):
    """
    Perform Canonical Correlation Analysis (CCA) between two datasets.
    
    Parameters:
    X : np.ndarray
        First dataset, shape (n_samples, n_features_X).
    Y : np.ndarray
        Second dataset, shape (n_samples, n_features_Y).
    num_components : int
        Number of CCA components to return.
        
    Returns:
    w_X : np.ndarray
        Canonical weights for the first dataset, shape (n_features_X, num_components).
    w_Y : np.ndarray
        Canonical weights for the second dataset, shape (n_features_Y, num_components).
    corrs : np.ndarray
        Array of canonical correlations for each component.
    """
    
    # If sample size is specified, randomly sample a subset of the data
    if size is not None and size < X.shape[0]:
        indices = np.random.choice(X.shape[0], size, replace=False)
        X = X[indices]
        Y = Y[indices]

    # Standardize both datasets (mean = 0, variance = 1)
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    
    X = scaler_X.fit_transform(X)
    Y = scaler_Y.fit_transform(Y)
    
    # Covariance matrices
    C_XX = np.cov(X, rowvar=False)  # Covariance of X
    C_YY = np.cov(Y, rowvar=False)  # Covariance of Y
    C_XY = np.cov(X, Y, rowvar=False)[:X.shape[1], X.shape[1]:]  # Cross-covariance of X and Y

    # Regularization term to avoid singular matrices
    reg = 1e-6
    inv_C_XX = np.linalg.inv(C_XX + reg * np.eye(C_XX.shape[0]))
    inv_C_YY = np.linalg.inv(C_YY + reg * np.eye(C_YY.shape[0]))

    # Solve the generalized eigenvalue problem for CCA
    # (inv_C_XX @ C_XY @ inv_C_YY @ C_XY.T) and vice versa for Y
    A = inv_C_XX @ C_XY @ inv_C_YY @ C_XY.T
    B = inv_C_YY @ C_XY.T @ inv_C_XX @ C_XY

    # Perform eigenvalue decomposition
    eigvals_X, eigvecs_X = np.linalg.eigh(A)
    eigvals_Y, eigvecs_Y = np.linalg.eigh(B)

    # Sort the eigenvalues and eigenvectors in descending order
    idx_X = np.argsort(eigvals_X)[::-1]
    idx_Y = np.argsort(eigvals_Y)[::-1]
    
    eigvecs_X = eigvecs_X[:, idx_X]
    eigvecs_Y = eigvecs_Y[:, idx_Y]

    # Canonical weights (the first `num_components` components)
    w_X = eigvecs_X[:, :num_components]
    w_Y = eigvecs_Y[:, :num_components]
    
    # Canonical correlations (square root of the eigenvalues, constrained to [0,1])
    corrs = np.sqrt(np.clip(eigvals_X[:num_components], 0, 1))

    return np.mean(corrs)
    return w_X, w_Y, corrs

def pca(X, Y, size=1000, n_components=3, bins=30):

    pca_X = PCA(n_components=n_components)
    X_pca = pca_X.fit_transform(X)

    pca_Y = PCA(n_components=n_components)
    Y_pca = pca_Y.fit_transform(Y)

    # Step 2: Determine common bin edges based on the range of PCA components
    min_value = min(X_pca.min(), Y_pca.min())
    max_value = max(X_pca.max(), Y_pca.max())
    bin_edges = np.linspace(min_value, max_value, bins + 1)

    # Step 3: Calculate histograms for each PCA component using the same bins
    histograms_X = [np.histogram(X_pca[:, i], bins=bin_edges, density=True)[0] for i in range(n_components)]
    histograms_Y = [np.histogram(Y_pca[:, i], bins=bin_edges, density=True)[0] for i in range(n_components)]

    # Step 4: Calculate Wasserstein distance between corresponding histograms
    total_distance = 0
    for i in range(n_components):
        total_distance += wasserstein_distance(histograms_X[i], histograms_Y[i])

    # Step 5: Normalize the total distance for a similarity score
    # Calculate the maximum possible distance (theoretical max could be based on histogram size)
    # This could be replaced with a more complex calculation if necessary.
    max_distance = 1.0  # Replace this with a suitable maximum based on your dataset properties.
    
    similarity_score = 1 - (total_distance / max_distance)

    return max(0, min(1, similarity_score))  # Ensure the score stays in [0, 1]
