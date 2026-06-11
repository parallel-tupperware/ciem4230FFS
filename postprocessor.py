import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import json 
import geometry
import mooring_tools 
import pandas as pd



#print(ds)
# uy = ds["u"].values.reshape(ds.dims["time"], -1, 6)[:, :, 1]
# ar = uy[:, -1]

def get_displacements(ds):
    n_time = ds.sizes["time"]
    n_dof = ds.sizes["dof"]

    # Reshape to (time, node, dof_per_node)
    u = ds["u"].values.reshape(n_time, n_dof // 6, 6)

    # Select node 125 (index 124)
    node_idx = 124

    # Extract 2nd and 3rd DOF (indices 1 and 2)
    ar_ux = u[:, node_idx, 0]   # DOF 1
    ar_uy = u[:, node_idx, 1]   # DOF 2
    ar_uz = u[:, node_idx, 2]   # DOF 3
    return np.column_stack([ar_ux, ar_uy, ar_uz])

def get_meta_data(ds):
    conf = json.loads(ds.attrs["metadata_json"])
    return conf

def get_geometry_center(d, idx=125):

    x0 = d['x0']
    x1 = d['x1']

    z0 = d['z0']
    z1 = d['z1']

    r_y = d['Ry']
    r_z = d['Rz']
    d_s = d['ds']



    ar = geometry.create_double_arc_geometry(x0=x0,
                                                x1=x1,
                                                z0=z0,
                                                z1=z1,
                                                R_y=r_y,
                                                R_z=r_z,
                                                ds=d_s)
    


    ar_center_coords = ar[idx,:]
    return ar_center_coords

def get_global_center_coordinates(ds):
    ar_displacements = get_displacements(ds)
    conf = get_meta_data(ds)
    d = conf['geometric_properties']

    ar_center_0 = get_geometry_center(d)
    ar_dofs_global = ar_center_0 + ar_displacements
    
    ar_y = ar_dofs_global[:,1]
    ar_z = ar_dofs_global[:,2]
    return ar_y, ar_z


def get_E_and_A(d):
    # Extract values from your dict `d`
    E_aramid = d["mooring_system_properties"]["youngs_modulus_aramid"]
    n_strands = d["mooring_system_properties"]["n_strands_per_anchor"]
    diameter = d["mooring_system_properties"]["diameter_mooring_line"]

    # Cross-sectional area of a single circular line
    A_single = np.pi * (diameter**2) / 4.0

    # Total cross-sectional area (all strands)
    A_total = n_strands * A_single
    return E_aramid, A_total

def get_max_tension_for_file(file):
    ds = xr.open_dataset(file)
    par_DEPTH_0 = 108
    par_DEPTH   = 300
    par_THETA_DEG = 45
    par_THETA = np.deg2rad(par_THETA_DEG)
    ar_z, ar_y = get_global_center_coordinates(ds)
    conf = get_meta_data(ds)
    E, A = get_E_and_A(conf)

    max_tension = mooring_tools.max_mooring_tension(ar_y, ar_z, E, A, par_DEPTH, par_DEPTH_0, par_THETA)
    return max_tension

