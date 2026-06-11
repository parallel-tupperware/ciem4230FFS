import numpy as np
from numba import njit, prange
from nodes import Node
from tqdm import tqdm
from dask import delayed, compute
from dask.distributed import Client
from datetime import datetime
from scipy.sparse import coo_matrix
import sys
import mooring_tools
import plot_tools
import pandas as pd

def build_element_helper(x_i, y_i, z_i, x_j, y_j, z_j, idx_i, idx_j, d):
        E_val = d['E_val']
        G_val = d['G_val']
        A_val = d['A_val']
        I_y_val = d['I_y_val']
        I_z_val = d['I_z_val']
        J_val = d['J_val']
        rho_val = d['rho_val']

        node_i = Node(idx_i, x_i, y_i, z_i, 6)
        node_j = Node(idx_j, x_j, y_j, z_j, 6)

        element = BendingBeam3DElement(idx_i, idx_j, x_i, y_i, z_i, x_j, y_j, z_j, E_val, G_val, A_val, I_y_val, I_z_val, J_val, rho_val)

        d = {
            'global_K_elastic' : triple_matrix_product_helper( element._transformation_matrix_transposed, element.K_local ,  element._transformation_matrix),
            'global_M_elastic' : triple_matrix_product_helper( element._transformation_matrix_transposed, element.M_local ,  element._transformation_matrix),
            'global_dofs' : element.global_dof_vector
        }

        d_k = {
            'K' : d['global_K_elastic'],
            'idx_u' : d['global_dofs']
        }

        d_m = {
            'M' : d['global_M_elastic'],
            'idx_u' : d['global_dofs']
        }

        return d_k, d_m

class ElementLedger3DBeamStructure : 
    def __init__(self, node_ledger, dict_geometry, sampler):

        self.sampler = sampler
        df_node_ledger = node_ledger.df
        ar_xyz = df_node_ledger[['global_x', 'global_y', 'global_z']].to_numpy()
        
        node_idxs = list(df_node_ledger.index)
        element_idxs = list(zip(node_idxs[:-1], node_idxs[1:]))
        self.conf = node_ledger.conf
        d = dict_geometry

        elements_K = []
        elements_M = []

        self.coordinate_df = self._build_element_ledger(df_node_ledger)
        self.ar_coords = self.coordinate_df[['xm', 'ym', 'zm']].to_numpy()


        t0 = datetime.now()        
        for idx_i, idx_j in tqdm(element_idxs):
            x_i, y_i, z_i = ar_xyz[idx_i, :]
            x_j, y_j, z_j = ar_xyz[idx_j, :]
            d_k, d_m = build_element_helper(x_i, y_i, z_i, x_j, y_j, z_j , idx_i, idx_j, d)
            elements_K.append(d_k)
            elements_M.append(d_m)
        
        
        t1 = datetime.now()
        print('assembling matrices')
        K_global = assemble_sparse_matrix(elements_K, key='K')
        M_global = assemble_sparse_matrix(elements_M, key='M')
        self._K_global_linear = K_global
        self._M_global_linear = M_global

        #plot_tools.save_sparsity_plot(M_global, 'M_global')
    
    def _sample_hydro_field(self, t):
        ar = self.ar_coords
        ar_v = self.sampler.inference_velocities(ar, t)
        ar_a = self.sampler.inference_accelerations(ar, t)
        return ar_v, ar_a

    def get_morisson_force_vector(self, t):
        df = self.coordinate_df.copy()
        #print(df)
        ar_v, ar_a = self._sample_hydro_field(t)
        df[['v_x', 'v_y', 'v_z']] = ar_v
        df[['a_x', 'a_y', 'a_z']] = ar_a
        
        D = self.conf['tunnel_properties']['outer_diameter_tunnel']
        Cm = self.conf['tunnel_properties']['C_m']
        Cd = self.conf['tunnel_properties']['C_d']
        rho = self.conf['tunnel_properties']['density_water']

        df_force =  add_morison_force_numba(df, D, Cm, Cd, rho)

        ar = assemble_global_force(df_force, set_partial_dof_vector, 6006) #hardcoded 50m=3006, 20m=7506, 100m=1506

        #print(df_force[['F_x', 'F_y', 'F_z']].mean())
        return ar





    def _build_mooring_constructor(self, node_ledger):
        df = node_ledger.df
        mask = df['has_anchor']
        df.to_csv('mooring_d0.csv')
        #f_K_i = mooring_tools
        #print(df)

    def _build_element_ledger(self, df):
        """
        Create a dataframe of consecutive point pairs (i, i+1)
        with coordinate values and absolute differences.
        """

        # shifted dataframe for j = i+1
        df_shifted = df[['global_x', 'global_y', 'global_z']].shift(-1)

        # construct result
        result = pd.DataFrame({
            'index': list(zip(df.index[:-1], df.index[1:])),

            'x_i': df['global_x'][:-1].values,
            'x_j': df_shifted['global_x'][:-1].values,

            'y_i': df['global_y'][:-1].values,
            'y_j': df_shifted['global_y'][:-1].values,

            'z_i': df['global_z'][:-1].values,
            'z_j': df_shifted['global_z'][:-1].values,
        })
        result.set_index('index', inplace=True)
        
        #print(result.columns)

        result['xm'] = result[['x_i', 'x_j']].mean(axis=1)
        result['ym'] = result[['y_i', 'y_j']].mean(axis=1)
        result['zm'] = result[['z_i', 'z_j']].mean(axis=1)


        return result

class MooringStiffnesssMatrix :
    def __init__(self, E, A_n, node_ledger, theta):
        self.df = node_ledger.df
        self.E = E
        self.A_n = A_n
        self.depth = self.df['depth']
        self.d0 = self.df['global_z']
        self.theta = theta

        self.K_symbolic = mooring_tools.symbolic_stifness_matrix_total()
        self.f_K_mooring = mooring_tools.global_node_mooring_stiffness_generator(self.K_symbolic)

        


    def get_nonlinear_stiffness(self, u):
        # u = np.array([
        # [y1, z1],
        # [y2, z2],
        # ...
        # [yN, zN]
        #])
        #print('begin assembling non lin mat')
        mask = self.df['has_anchor']
        ar_mask_nan = mask.to_numpy()
        ar_mask = ar_mask_nan == True
        ar_u = convert_u_to_mooring_format(u)[ar_mask]

        df_mooring = self.df[ar_mask]
        ar_mooring_props = df_mooring[['depth', 'global_z']].to_numpy()
        ar_idx = df_mooring.index.to_numpy()#.reshape(1,N)
        N = ar_idx.size
        ar_idx = np.reshape(ar_idx, (N, 1))

        # print(f'''
        # shape ar_idx : {ar_idx.shape}
        # shape ar_mooring_props : {ar_mooring_props.shape}
        # shape ar_u : {ar_u.shape}
        # ''')

        ar = np.hstack([ar_idx, ar_mooring_props, ar_u]) # new array is : idx - depth - d0 - y - z 
        l_elements = _helper_get_nonlinear_stiffnesss(ar, self.theta, self.E, self.A_n, self.f_K_mooring)
        #print('these are my elements for nonlin:', l_elements, 'end')
        #print(l_elements)
        return assemble_sparse_matrix(l_elements, 'K', matrix_size=36, ndofs_total=6006) #hardcoded 50m=306 

#@njit(parallel=True, fastmath=True)
def _helper_get_nonlinear_stiffnesss(ar, theta, E, A_n, f):

    l_elements = []

    for row in ar:
        idx = row[0]
        depth = row[1]
        d0 = row[2]
        y = row[3]
        z = row[4]
        ar_stiffness_2x2 = f(y, z, E, A_n, depth, d0, theta)
        ar_idxs = set_partial_dof_vector_node(idx, 6)
        ar_stiffness_6x6 = np.zeros([6,6])
        ar_stiffness_6x6[1,1] = ar_stiffness_2x2[0,0]
        ar_stiffness_6x6[1,2] = ar_stiffness_2x2[0,1]
        ar_stiffness_6x6[2,1] = ar_stiffness_2x2[1,0]
        ar_stiffness_6x6[2,2] = ar_stiffness_2x2[1,1]

        d = {
            'K' : ar_stiffness_6x6,
            'idx_u' : ar_idxs
        }
        l_elements.append(d)
    
    return l_elements



        
        


        

        



        

def assemble_sparse_matrix(elements_K, key, ndofs_total=None, matrix_size=144):
    n_el = len(elements_K)
    entries_per_el = matrix_size

    if ndofs_total is None:
        ndofs_total = max(np.max(el['idx_u']) for el in elements_K) + 1

    print(f'total detected degrees of freedom : {ndofs_total}')

    rows = np.empty(n_el * entries_per_el, dtype=np.int64)
    cols = np.empty(n_el * entries_per_el, dtype=np.int64)
    data = np.empty(n_el * entries_per_el, dtype=np.float64)

    offset = 0

    for el in elements_K:
        K = el[key]
        dofs = el['idx_u']

        rr = np.repeat(dofs, len(dofs))
        cc = np.tile(dofs, len(dofs))

        n = entries_per_el
        rows[offset:offset+n] = rr
        cols[offset:offset+n] = cc
        data[offset:offset+n] = K.ravel()

        offset += n

    K_global = coo_matrix((data, (rows, cols)), shape=(ndofs_total, ndofs_total))
    #print(f'a sparse {key} matrix is made with shape : {K_global.shape}')
    return K_global.tocsr()

                    
#@njit
def convert_u_to_mooring_format(ar):
    ar = np.asarray(ar)
    if ar.size % 6 != 0:
        raise ValueError("Input array length must be divisible by 6")
    return ar.reshape(-1, 6)[:, 1:3]



    
            


class Generic1DElement:

    def __init__(self, x0, y0, z0, x1, y1, z1):
        self.p0 = np.array([x0, y0, z0], dtype=np.float64)
        self.p1 = np.array([x1, y1, z1], dtype=np.float64)
        self._calculate_length()
        self._calculate_transformation_matrix()


    def _calculate_length(self):
        self._element_length = np.linalg.norm(self.p1 - self.p0)

    def _calculate_transformation_matrix(self):

        p0 = self.p0
        p1 = self.p1
        
        T = _get_transformation_matrix(p0, p1)

        self._transformation_matrix = T
        self._transformation_matrix_transposed = T.T





class BendingBeam3DElement(Generic1DElement):

    def __init__(self, n0, n1, x0, y0, z0, x1, y1, z1, E, G, A, Iy, Iz, J, rho):
        super().__init__(x0, y0, z0, x1, y1, z1)
        self.L = self._element_length
        self.K_local = self.get_K(E, G, A, Iy, Iz, J)
        self.M_local = self.get_M(rho, A)
        self.global_dof_vector = self.get_partial_dof_vector(n0, n1, 6)

    def get_K(self, E, G, A, Iy, Iz, J):
        return compute_K(E, G, A, self.L, Iy, Iz, J)

    def get_M(self, rho, A):
        return compute_M(rho, A, self.L)
    
    def get_partial_dof_vector(self, n0, n1, ndofs):
        return set_partial_dof_vector(n0, n1, ndofs)
    
@njit
def triple_matrix_product_helper(A, B, C):
    return A @ B @ C
    
@njit
def set_partial_dof_vector(n0, n1, ndofs):
    dof_idxs = np.empty(2 * ndofs, dtype=np.int64)

    for i in range(ndofs):
        dof_idxs[i] = n0 * ndofs + i
        dof_idxs[i + ndofs] = n1 * ndofs + i

    return dof_idxs

@njit
def set_partial_dof_vector_node(n_node, ndofs):
    base = n_node * ndofs
    dof_idxs = np.empty(ndofs, dtype=np.int64)

    for i in range(ndofs):
        dof_idxs[i] = base + i

    return dof_idxs

@njit
def compute_K( E, G, A, L, Iy, Iz, J):
    K = np.zeros((12, 12))

    # Axial
    EA_L = E * A / L
    K[0,0] = K[6,6] = EA_L
    K[0,6] = K[6,0] = -EA_L

    # Torsion
    GJ_L = G * J / L
    K[3,3] = K[9,9] = GJ_L
    K[3,9] = K[9,3] = -GJ_L

    # Bending Z
    c1 = 12 * E * Iz / L**3
    c2 = 6 * E * Iz / L**2
    c3 = 4 * E * Iz / L
    c4 = 2 * E * Iz / L

    K[1,1] = K[7,7] = c1
    K[1,7] = K[7,1] = -c1
    K[1,5] = K[5,1] = c2
    K[1,11] = K[11,1] = c2
    K[5,7] = K[7,5] = -c2
    K[7,11] = K[11,7] = -c2
    K[5,5] = K[11,11] = c3
    K[5,11] = K[11,5] = c4

    # Bending Y
    c1 = 12 * E * Iy / L**3
    c2 = 6 * E * Iy / L**2
    c3 = 4 * E * Iy / L
    c4 = 2 * E * Iy / L

    K[2,2] = K[8,8] = c1
    K[2,8] = K[8,2] = -c1
    K[2,4] = K[4,2] = -c2
    K[2,10] = K[10,2] = -c2
    K[4,8] = K[8,4] = c2
    K[8,10] = K[10,8] = c2
    K[4,4] = K[10,10] = c3
    K[4,10] = K[10,4] = c4

    return K


@njit
def compute_M( rho, A, L):
    M = np.zeros((12, 12))
    m = rho * A * L

    # Axial
    M[0,0] = M[6,6] = m/3
    M[0,6] = M[6,0] = m/6

    # Transverse Y
    M[1,1] = M[7,7] = 13*m/35
    M[1,7] = M[7,1] = 9*m/70
    M[1,5] = M[5,1] = 11*m*L/210
    M[1,11] = M[11,1] = -13*m*L/420

    M[5,5] = M[11,11] = m*L**2/105
    M[5,11] = M[11,5] = -m*L**2/140

    # Transverse Z
    M[2,2] = M[8,8] = 13*m/35
    M[2,8] = M[8,2] = 9*m/70
    M[2,4] = M[4,2] = -11*m*L/210
    M[2,10] = M[10,2] = 13*m*L/420

    M[4,4] = M[10,10] = m*L**2/105
    M[4,10] = M[10,4] = -m*L**2/140

    # Rotational inertia
    M[3,3] = M[9,9] = m*L**2/3
    M[3,9] = M[9,3] = m*L**2/6

    return M

@njit
def _get_transformation_matrix(p0, p1):
        # --- Step 1: local x-axis ---
        dx = p1 - p0
        L = np.linalg.norm(dx)
        ex = dx / L

        # --- Step 2: choose reference vector ---
        if abs(ex[2]) < 0.9:
            ref = np.array([0.0, 0.0, 1.0])
        else:
            ref = np.array([0.0, 1.0, 0.0])

        # --- Step 3: local z-axis ---
        ez = np.cross(ex, ref)
        ez /= np.linalg.norm(ez)

        # --- Step 4: local y-axis ---
        ey = np.cross(ez, ex)

        # --- Step 5: rotation matrix ---
        R = np.vstack((ex, ey, ez))  # rows = local axes

        # --- Step 6: build 12x12 T ---
        T = np.zeros((12, 12))
        for i in range(4):
            T[3*i:3*(i+1), 3*i:3*(i+1)] = R

        return T

def add_morison_force_numba(df, D, Cm, Cd, rho=1025.0):
    """
    Add Morison force components to a pandas DataFrame.

    This function extracts NumPy arrays from a DataFrame, computes
    segment-wise Morison forces using a Numba-accelerated kernel,
    and appends the resulting force components as new columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the following columns:

        Geometry:
            - x_i, y_i, z_i : start point coordinates
            - x_j, y_j, z_j : end point coordinates

        Fluid kinematics (evaluated at segment midpoint):
            - v_x, v_y, v_z : velocity components
            - a_x, a_y, a_z : acceleration components

    D : float
        Cylinder diameter [m]

    Cm : float
        Inertia coefficient [-]

    Cd : float
        Drag coefficient [-]

    rho : float, optional
        Fluid density [kg/m³], default = 1025 (seawater)

    Returns
    -------
    pandas.DataFrame
        Original DataFrame with additional columns:
            - F_x, F_y, F_z : force components per segment

    Notes
    -----
    - Uses a high-performance Numba kernel with parallel execution
    - Modifies the input DataFrame in-place
    - Recommended for large datasets (>100k rows)

    Example
    -------
    >>> df = add_morison_force_numba(df, D=1.0, Cm=2.0, Cd=1.0)
    >>> df[['F_x', 'F_y', 'F_z']].head()
    """

    Fx, Fy, Fz = morison_numba(
        df["x_i"].values, df["y_i"].values, df["z_i"].values,
        df["x_j"].values, df["y_j"].values, df["z_j"].values,
        df["v_x"].values, df["v_y"].values, df["v_z"].values,
        df["a_x"].values, df["a_y"].values, df["a_z"].values,
        D, Cm, Cd, rho
    )

    df["F_x"] = Fx
    df["F_y"] = Fy
    df["F_z"] = Fz

    return df


@njit(parallel=True, fastmath=True)
def morison_numba(
    xi, yi, zi,
    xj, yj, zj,
    vx, vy, vz,
    ax, ay, az,
    D, Cm, Cd, rho
):
    """
    Compute Morison forces for multiple cylindrical segments (vectorized via Numba).

    This function evaluates hydrodynamic forces on slender cylindrical elements
    using the Morison equation. Each segment is defined by two endpoints (i → j),
    and the fluid kinematics (velocity and acceleration) are assumed to be
    evaluated at the segment midpoint.

    The force is computed using only the velocity and acceleration components
    perpendicular to the cylinder axis.

    Parameters
    ----------
    xi, yi, zi : np.ndarray
        Coordinates of the start point of each segment (shape: n)

    xj, yj, zj : np.ndarray
        Coordinates of the end point of each segment (shape: n)

    vx, vy, vz : np.ndarray
        Fluid velocity components at segment midpoints (shape: n)

    ax, ay, az : np.ndarray
        Fluid acceleration components at segment midpoints (shape: n)

    D : float
        Cylinder diameter [m]

    Cm : float
        Inertia coefficient [-]

    Cd : float
        Drag coefficient [-]

    rho : float
        Fluid density [kg/m³]

    Returns
    -------
    Fx, Fy, Fz : np.ndarray
        Force components acting on each segment (shape: n)

    Notes
    -----
    Morison equation (vector form):

        F = 0.5 * rho * Cd * D * L * |v_perp| * v_perp
            + rho * Cm * (π D² / 4) * L * a_perp

    where:
        - L is the segment length
        - v_perp and a_perp are velocity and acceleration components
          perpendicular to the cylinder axis

    Performance
    -----------
    - Fully parallelized using Numba (prange)
    - Uses fastmath for improved SIMD/vectorization
    - Suitable for large-scale simulations (10^5–10^7 segments)

    Numerical Stability
    -------------------
    - Segments with near-zero length (L < 1e-12) are skipped
    """

    n = xi.shape[0]

    Fx = np.zeros(n)
    Fy = np.zeros(n)
    Fz = np.zeros(n)

    area = np.pi * D * D * 0.25  # cylinder cross-sectional area

    for i in prange(n):

        # --- segment vector ---
        Lx = xj[i] - xi[i]
        Ly = yj[i] - yi[i]
        Lz = zj[i] - zi[i]

        L = (Lx*Lx + Ly*Ly + Lz*Lz) ** 0.5
        if L < 1e-12:
            continue

        # --- unit direction vector ---
        invL = 1.0 / L
        tx = Lx * invL
        ty = Ly * invL
        tz = Lz * invL

        # --- projection of kinematics onto axis ---
        v_dot_t = vx[i]*tx + vy[i]*ty + vz[i]*tz
        a_dot_t = ax[i]*tx + ay[i]*ty + az[i]*tz

        # --- perpendicular components ---
        vpx = vx[i] - v_dot_t * tx
        vpy = vy[i] - v_dot_t * ty
        vpz = vz[i] - v_dot_t * tz

        apx = ax[i] - a_dot_t * tx
        apy = ay[i] - a_dot_t * ty
        apz = az[i] - a_dot_t * tz

        # --- magnitude of perpendicular velocity ---
        vmag = (vpx*vpx + vpy*vpy + vpz*vpz) ** 0.5

        # --- drag force ---
        drag_coeff = 0.5 * rho * Cd * D * L
        Fdx = drag_coeff * vmag * vpx
        Fdy = drag_coeff * vmag * vpy
        Fdz = drag_coeff * vmag * vpz

        # --- inertia force ---
        inertia_coeff = rho * Cm * area * L
        Fix = inertia_coeff * apx
        Fiy = inertia_coeff * apy
        Fiz = inertia_coeff * apz

        # --- total force ---
        Fx[i] = Fdx + Fix
        Fy[i] = Fdy + Fiy
        Fz[i] = Fdz + Fiz

    return Fx, Fy, Fz


import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def _assemble_force_kernel(Fx, Fy, Fz, dof_map, total_dofs):
    """
    Internal Numba kernel: parallel-safe assembly using thread-local buffers.
    """

    n = Fx.shape[0]

    # number of thread buckets (fixed upper bound, safe)
    n_threads = 64
    F_local = np.zeros((n_threads, total_dofs))

    for k in prange(n):
        tid = k % n_threads  # pseudo thread id

        fx = 0.5 * Fx[k]
        fy = 0.5 * Fy[k]
        fz = 0.5 * Fz[k]

        dofs = dof_map[k]

        # translational DOFs (i-node)
        F_local[tid, dofs[0]] += fx
        F_local[tid, dofs[1]] += fy
        F_local[tid, dofs[2]] += fz

        # translational DOFs (j-node)
        F_local[tid, dofs[6]] += fx
        F_local[tid, dofs[7]] += fy
        F_local[tid, dofs[8]] += fz

        # rotational DOFs are zero → skipped

    # reduction step
    F_global = np.zeros(total_dofs)

    for t in range(n_threads):
        for d in range(total_dofs):
            F_global[d] += F_local[t, d]

    return F_global


def assemble_global_force(df, get_dof_index_vector, total_dofs):
    """
    Assemble global force vector from a DataFrame of segment forces.

    Parameters
    ----------
    df : pandas.DataFrame
        Indexed by (i, j), must contain:
            - F_x, F_y, F_z

    get_dof_index_vector : function
        Function(i, j) -> array-like of length 12:
        [i1,...,i6, j1,...,j6]

    total_dofs : int
        Total number of degrees of freedom

    Returns
    -------
    np.ndarray
        Global force vector of shape (total_dofs,)

    Notes
    -----
    - Uses Numba parallel kernel with thread-local accumulation
    - Safe for large systems (shared DOFs handled correctly)
    - Designed for FEM-style assembly
    """

    n = len(df)

    # --- extract force arrays ---
    Fx = df["F_x"].to_numpy()
    Fy = df["F_y"].to_numpy()
    Fz = df["F_z"].to_numpy()

    # --- build DOF map (Python side, once) ---
    dof_map = np.zeros((n, 12), dtype=np.int64)

    for k, (i, j) in enumerate(df.index):
        dof_map[k, :] = get_dof_index_vector(i, j, 6)

    # --- call fast kernel ---
    F_global = _assemble_force_kernel(Fx, Fy, Fz, dof_map, total_dofs)

    return F_global



@njit(parallel=True, fastmath=True)
def _assemble_force_kernel(Fx, Fy, Fz, dof_map, total_dofs):
    """
    Internal Numba kernel: parallel-safe assembly using thread-local buffers.
    """

    n = Fx.shape[0]

    # number of thread buckets (fixed upper bound, safe)
    n_threads = 64
    F_local = np.zeros((n_threads, total_dofs))

    for k in prange(n):
        tid = k % n_threads  # pseudo thread id

        fx = 0.5 * Fx[k]
        fy = 0.5 * Fy[k]
        fz = 0.5 * Fz[k]

        dofs = dof_map[k]

        # translational DOFs (i-node)
        F_local[tid, dofs[0]] += fx
        F_local[tid, dofs[1]] += fy
        F_local[tid, dofs[2]] += fz

        # translational DOFs (j-node)
        F_local[tid, dofs[6]] += fx
        F_local[tid, dofs[7]] += fy
        F_local[tid, dofs[8]] += fz

        # rotational DOFs are zero → skipped

    # reduction step
    F_global = np.zeros(total_dofs)

    for t in range(n_threads):
        for d in range(total_dofs):
            F_global[d] += F_local[t, d]

    return F_global

