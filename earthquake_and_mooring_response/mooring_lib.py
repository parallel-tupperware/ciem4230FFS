import sympy as sp
import numpy as np

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

def construct_section_mooring_stifness(K, d, d0, theta0):
    K

