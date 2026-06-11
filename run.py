from model import Model
import pathlib
import json
import argparse
from datetime import datetime
import xarray as xr
import numpy as np

def save_ivp_to_netcdf(res, u0, filename, include_velocity=False, metadata=''):
    """
    Save solve_ivp result to NetCDF.

    Parameters
    ----------
    res : OdeResult
        Output from scipy.solve_ivp
    u0 : array-like
        Initial displacement (used to split u/v)
    filename : str
        Output NetCDF file path
    include_velocity : bool
        Whether to also store velocities
    """
    n = u0.size
    
    # Extract time
    t = res.t
    
    # Extract displacement and velocity
    u = res.y[:n, :]        # shape (n, nt)
    v = res.y[n:, :]        # shape (n, nt)

    # Transpose so time is first dimension
    u = u.T                 # shape (nt, n)
    v = v.T

    # Build dataset
    data_vars = {
        "u": (("time", "dof"), u)
    }

    if include_velocity:
        data_vars["v"] = (("time", "dof"), v)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": t,
            "dof": np.arange(n)
        }
    )
    ds.attrs["metadata_json"] = metadata

    ds.to_netcdf(filename)

def main():
    parser = argparse.ArgumentParser(description="Model application with properties configuration.")
    parser.add_argument('--props', type=str, default='properties.json', help='Path to the properties configuration file (default: properties.json)')
    parser.add_argument('--output', type=str, default='output.txt', required=True)
    parser.add_argument('--solver', type=str, default='RK45')
    parser.add_argument('--mode', type=str, default='linear')

    args = parser.parse_args()
    
    with open(args.props, 'r') as f:
        d = json.load(f)
    
    print('building model')
    model = Model(d)
    #s = input('press enter to start timestepping')
    print('begin solving model')
    t0 = datetime.now()

    if args.mode == 'linear':
        res, u0 = model.solve_system_linear(args.solver)
    elif args.mode == 'nonlinear':
        res, u0 = model.solve_system_moored_nonlin(args.solver)
    else : 
        raise ValueError(args.mode)
    
    
    t1 = datetime.now()

    print(f'main processing took {t1 - t0}')

    u0 = u0
    s = json.dumps(d)
    save_ivp_to_netcdf(res, u0, f'./{args.output}', include_velocity=False, metadata=s)





if __name__ == "__main__":
    main()
