from postprocessor import get_max_tension_for_file, get_meta_data
from pathlib import Path
import pandas as pd
import xarray as xr
import hashlib
from tqdm import tqdm

def compute_hash_from_values(values, precision=6):
    row_str = "_".join(f"{x:.{precision}f}" for x in values)
    return hashlib.sha256(row_str.encode()).hexdigest()





def get_hash_value_from_file(file):
    ds = xr.open_dataset(file)
    d = get_meta_data(ds)
    row_values = [
    d["hydro_load_properties"]["H_wave_m"],
    d["hydro_load_properties"]["current_velocity"],
    d["hydro_load_properties"]["T_wave_s"],
    d["mooring_system_properties"]["youngs_modulus_aramid"],
    d["tunnel_properties"]["wall_thickness_tunnel"],
    d["mooring_system_properties"]["anchor_distance_x"],
    d["tunnel_properties"]["density_concrete"],
    ]
    row_hash = compute_hash_from_values(row_values)
    return row_hash

def get_env_valuesZ_from_file(file):
    ds = xr.open_dataset(file)
    d = get_meta_data(ds)
    T_max = get_max_tension_for_file(file)
    row_values = [
    d["hydro_load_properties"]["H_wave_m"],
    d["hydro_load_properties"]["current_velocity"],
    d["hydro_load_properties"]["T_wave_s"],
    d["mooring_system_properties"]["youngs_modulus_aramid"],
    d["tunnel_properties"]["wall_thickness_tunnel"],
    d["mooring_system_properties"]["anchor_distance_x"],
    d["tunnel_properties"]["density_concrete"],
    T_max
    ]

    col_names = [
    "H_wave_m",
    "current_velocity",
    "T_wave_s",
    "youngs_modulus_aramid",
    "all_thickness_tunnel",
    "anchor_distance_x",
    "density_concrete",
    'max_tension'
    ]
    

    df_row = pd.DataFrame([row_values], columns=col_names)

    return df_row



def get_paired_tension_df(files):
    list_rows = []    
    for file in tqdm(files):
        row = get_env_valuesZ_from_file(file)
        list_rows.append(row)
    df_combined = pd.concat(list_rows, ignore_index=True)
    return df_combined
    

def add_max_tension(df, hash_pairs, hash_col="row_hash", out_col="max_tension"):
    """
    Adds a max_tension column to a dataframe by matching row_hash.

    Parameters:
        df (pd.DataFrame): Input dataframe (must contain hash column)
        hash_pairs (list): [[T1, hash1], [T2, hash2], ...]
        hash_col (str): Name of hash column in df
        out_col (str): Name of output column

    Returns:
        pd.DataFrame: Copy with max_tension column added
    """
    df_copy = df.copy()

    # Build lookup dictionary: hash -> tension
    hash_to_tension = {h: T for T, h in hash_pairs}

    # Map efficiently
    df_copy[out_col] = df_copy[hash_col].map(hash_to_tension)

    return df_copy

nc_files_current = list(Path("Simulation_Results").rglob("*.nc"))
nc_files_waves   = list(Path("Simulation_Results_wsh").rglob("*.nc"))

df_current = get_paired_tension_df(nc_files_current)
df_waves   = get_paired_tension_df(nc_files_waves)

df_current.to_csv('paired_current.csv')
df_waves.to_csv('paired_waves.csv')





