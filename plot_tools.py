import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import issparse, coo_matrix

def save_sparsity_plot(K, title, markersize=0.2, max_points=2_000_000):
    """
    Save sparsity pattern of a (possibly sparse) matrix to ./plots/{title}.png.

    Parameters
    ----------
    K : array-like or sparse matrix
        Matrix to visualize.
    title : str
        Title and filename (without extension).
    markersize : float, optional
        Marker size for plotting (smaller for larger matrices).
    max_points : int, optional
        Maximum number of points to plot (downsampling applied if needed).
    """

    # Ensure output directory exists
    os.makedirs("./plots", exist_ok=True)

    # Convert to COO (best format for plotting coordinates)
    if issparse(K):
        Kcoo = K.tocoo()
        rows, cols = Kcoo.row, Kcoo.col
    else:
        rows, cols = np.nonzero(K)

    nnz = len(rows)

    # Downsample if too large
    if nnz > max_points:
        idx = np.random.choice(nnz, size=max_points, replace=False)
        rows = rows[idx]
        cols = cols[idx]

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(cols, rows, s=markersize)
    plt.gca().invert_yaxis()
    plt.title(f"{title} (nnz={nnz})")

    # Save
    filepath = f"./plots/{title}.png"
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    return filepath