import sympy as sp
import numpy as np
from scipy.optimize import root_scalar
from numba import njit

def symbolic_tension_vector_expression_minus(tension_check=True):
    E, A_N, d, d0, theta0 = sp.symbols(['E', 'A_N', 'd', 'd_0' , 'theta_0'])
    y, z = sp.symbols(['y', 'z'])
    L = sp.sqrt((y+(d-d0)*sp.cot(theta0))**2+(z+d)**2)
    L0 = (d-d0)/sp.sin(theta0)
    T_scalar = E*A_N*(L-L0)

    r_y = y+(d-d0)*sp.cot(theta0)
    r_z = z+d

    r_vec = sp.Matrix([r_y, r_z])
    r_unit_vec = r_vec / L
    H = sp.functions.special.delta_functions.Heaviside
    if tension_check:
        T_vec = T_scalar * r_unit_vec * H(L-L0)
    else :
        T_vec = T_scalar * r_unit_vec
    return T_vec, (y,z)

def symbolic_tension_vector_expression_plus(tension_check=True):
    E, A_N, d, d0, theta0 = sp.symbols(['E', 'A_N', 'd', 'd_0' , 'theta_0'])
    y, z = sp.symbols(['y', 'z'])

    L = sp.sqrt((-y+(d-d0)*sp.cot(theta0))**2+(z+d)**2)
    
    L0 = (d-d0)/sp.sin(theta0)
    T_scalar = E*A_N*(L-L0)

    r_y = -1*(-y+(d-d0)*sp.cot(theta0))
    r_z = z+d

    r_vec = sp.Matrix([r_y, r_z])
    r_unit_vec = r_vec / L
    H = sp.functions.special.delta_functions.Heaviside
    if tension_check:
        T_vec = T_scalar * r_unit_vec * H(L-L0)
    else : 
        T_vec = T_scalar * r_unit_vec
    return T_vec, (y,z)

def symbolic_stifness_matrix_minus(simplify=False):
    T_vec, variables = symbolic_tension_vector_expression_minus()
    K_matrix = T_vec.jacobian(variables)
    if simplify:
        return K_matrix.applyfunc(sp.simplify)
    else:
        return K_matrix
    
def symbolic_stifness_matrix_plus(simplify=False):
    T_vec, variables = symbolic_tension_vector_expression_plus()
    K_matrix = T_vec.jacobian(variables)
    if simplify:
        return K_matrix.applyfunc(sp.simplify)
    else:
        return K_matrix
    
def symbolic_tension_vector_total():
    T_vec_min, variables = symbolic_tension_vector_expression_minus()
    T_vec_plus, variables = symbolic_tension_vector_expression_plus()
    return T_vec_min + T_vec_plus, variables

def symbolic_stifness_matrix_total():
    T_vec, variables = symbolic_tension_vector_total()
    K_matrix = T_vec.jacobian(variables)
    return K_matrix

def dirac_delta(x):
    if x==0:
        return np.inf
    else:
        return 0
    
dirac_delta_vec = np.vectorize(dirac_delta)

def global_node_mooring_stiffness_generator(K_total ):#, d, d0, theta, E_val, A_N_val):

    # variable_dict = {
    #     'E' : E_val,
    #     'A_N' : A_N_val,
    #     'd' : d,
    #     'd_0': d0,
    #     'theta_0': theta
    # }

    # spatial_vars = sp.symbols(['y', 'z', ])

    all_vars = sp.symbols(['y', 'z', 'E', 'A_N', 'd', 'd_0', 'theta_0'])

    f_K_total = sp.lambdify(all_vars, K_total , modules=[{'DiracDelta' : dirac_delta_vec}, 'numpy'])
    return f_K_total

def get_z1(d, d0, theta_val, E, A , s, q):

    h_0 = d - d0
    b_0 = h_0 / np.tan(theta_val)
    L_0 = np.sqrt(h_0**2 + b_0**2)

    def f_theta(y,z):

        return np.atan((z - h_0)/(y - b_0))

    def f_L(y,z):
        return np.sqrt((y+(d-d0)/np.tan(f_theta(0,z)))**2 + (z+d)**2)
                       
    def f_root_pretension_equation(z):
        return (q * s) / (2 * np.sin(f_theta(0, z))) - E * A * (f_L(0, z) - L_0) / L_0

    sol = root_scalar(f_root_pretension_equation, bracket=[-d0, 0], method='brentq')
    return sol.root


def max_mooring_tension(y, z, E, A_N, d, d0, theta0):
    # Precompute constants
    c = (d - d0) / np.tan(theta0)
    L0 = (d - d0) / np.sin(theta0)
    
    # --- Minus case ---
    L_minus = np.sqrt((y + c)**2 + (z + d)**2)
    T_minus = E * A_N * np.maximum(L_minus - L0, 0.0)
    
    # --- Plus case ---
    L_plus = np.sqrt((-y + c)**2 + (z + d)**2)
    T_plus = E * A_N * np.maximum(L_plus - L0, 0.0)
    
    # Stack and take row-wise max
    T_combined = np.column_stack((T_minus, T_plus))
    T_max = np.max(T_combined)
    
    return T_max