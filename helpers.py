import pickle
import numpy as np

def save_variable(x, filename):
    """
    Save a Python object to a file using pickle.
    
    Parameters:
        x: Any Python object
        filename (str): File path (e.g. "data.pkl")
    """
    x_dense = x.toarray()
    
    n = x_dense.shape[0]

    free_dofs = list(range(6, n - 6))

    x_FF = x_dense[np.ix_(free_dofs, free_dofs)]
    print(x_FF.__class__)
    print(x_FF.shape)

    with open(filename, "wb") as f:
        pickle.dump(x_FF, f)


def load_variable(filename):
    """
    Load a Python object from a pickle file.
    
    Parameters:
        filename (str): File path
    
    Returns:
        The loaded Python object
    """
    with open(filename, "rb") as f:
        return pickle.load(f)