import numpy as np

def eigenfreq(M_FF, K_FF):
    """
    Takes Mass and stiffness matrices 
    to compute eigenfrequencies and eigenvectors
    """
    mat = np.linalg.inv(M_FF).dot(K_FF)
    w2, vr = np.linalg.eig(mat)
    w = np.sqrt(w2.real)
    f = w/2/np.pi

    return f, vr