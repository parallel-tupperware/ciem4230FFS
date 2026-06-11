import numpy as np

import numpy as np

def apply_clamped_bc(M, K):
    """
    Apply clamped boundary conditions at node 1 and node N
    for a 3D beam (6 DOFs per node).
    """

    M = M.copy()
    K = K.copy()

    ndof = M.shape[0]
    dof_per_node = 6

    # --- sanity check ---
    assert ndof % dof_per_node == 0, "ndof must be divisible by 6"

    N = ndof // dof_per_node

    # --- Identify constrained DOFs ---
    fixed_dofs = list(range(0, 6)) + list(range(dof_per_node*(N-1), dof_per_node*N))

    # --- Apply row/column replacement ---
    for i in fixed_dofs:
        # Zero rows
        M[i, :] = 0.0
        K[i, :] = 0.0

        # Zero columns
        M[:, i] = 0.0
        K[:, i] = 0.0

        # ✅ CRITICAL: restore diagonal
        M[i, i] = 1.0
        K[i, i] = 1.0

    return M, K, fixed_dofs

def apply_clamped_bc_forcing(q):
    ar = q.copy()
    ar[:6] = 0
    ar[-6:] = 0
    return ar

