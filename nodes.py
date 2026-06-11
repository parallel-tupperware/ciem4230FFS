import geometry
import mooring_tools
import seismic_tools
import numpy as np
#from parameter_loader import Params
from bathymetry_tools import BathymetrySampler
from tqdm import tqdm
from datetime import datetime
from field_tools import HydroDynamicField
from numba import njit

class Node :
    def __init__(self, n, x, y, z, ndofs):
        self.x = x
        self.y = y
        self.z = z
        self.idx = n
        self.dofs = np.arange(ndofs) + n * ndofs

class NodeLedger :
    def __init__ (self, df, lonlat_0, lonlat_1, tiff_file, conf, q, sampler):
        self.df = df
        self.q = q
        self.conf = conf
        self.bathymetry_sampler = sampler
        #self._add_bathymetry_sampler(lonlat_0, lonlat_1, tiff_file)
        self._add_initial_conditions(sampler)
        #print(self.df)

    def get_coords(self, idx):
        row = self.df.iloc[idx]
        xyz = row[['global_x', 'global_y', 'global_z']]
        return xyz
    
    def get_deflections_to_global_converter(self):
        ar_coords_global = np.ascontiguousarray(
            self.df[['global_x', 'global_y', 'global_z']].to_numpy()
        )

        @njit
        def f(ar_deflections, ar_global = ar_coords_global):
            return ar_deflections + ar_global

        return f
    
    def _add_initial_conditions(self, sampler):

        d = self.conf['mooring_system_properties']

        s_anchor = d['anchor_distance_x']
        E = d['youngs_modulus_aramid']
        A_cross = d['diameter_mooring_line'] * d['diameter_mooring_line'] * np.pi  / 4
        A_N = A_cross * d['n_strands_per_anchor']
        min_depth = d['minimum_mooring_depth']

        theta_deg = d['theta_deg_anchor']
        theta = theta_deg * np.pi / 180

        coords = self.df[['global_x', 'global_y', 'global_z']]
        diff = coords.diff()
        self.df['delta_s'] = np.sqrt(np.square(diff).sum(axis=1))
        self.df['total_s'] = self.df['delta_s'].cumsum()
        bins = (self.df["total_s"] // s_anchor).astype(int)
        self.df["has_anchor"] = bins.diff().fillna(0) > 0

        mask = self.df['has_anchor']
        coords = self.df.loc[mask, ['global_x', 'global_y']].values

        depths = sampler.sample_depth(coords)
        self.df.loc[mask, 'depth'] = depths

        mask = self.df['depth'] < -min_depth

        l_z1 = []
        for row in self.df[mask].iloc():
            y = 0
            t1=datetime.now()
            z_1 = mooring_tools.get_z1(
                abs(row['depth']), # depth of location
                abs(row['global_z']), # d0 at location
                theta, # theta at location
                E,
                A_N,
                self.q,
                s_anchor
            )
            l_z1.append(z_1)
        self.df.loc[mask, 'z1'] = l_z1
        self.df['is_underground'] = self.df['global_z'] < self.df['depth']
        underground_mask = self.df['is_underground'] == True
        self.df.loc[underground_mask, 'z1'] = np.nan
        self.df.loc[mask, 'has_hydro_loading'] = self.df['global_z'] >= self.df['depth']
        self.df['has_anchor'] = self.df['has_hydro_loading']
        self.df['u_z_initial'] = self.df['global_z'] - self.df['z1']
        self.df.loc[self.df.index[0], 'u_z_initial'] = 0
        self.df.loc[self.df.index[-1], 'u_z_initial'] = 0

        self.df.loc[self.df.index[0], 'has_hydro_loading'] = False
        self.df.loc[self.df.index[-1], 'has_hydro_loading'] = False

        self.df['u_z_initial'] = self.df['u_z_initial'].interpolate(method='polynomial', order=3)
        self.df['u_y_initial'] = 0
        self.df['u_x_initial'] = 0
        self.df['theta_x_initial'] = 0
        self.df['theta_y_initial'] = 0
        self.df['theta_z_initial'] = 0

        ar = self.df[['u_x_initial', 'u_y_initial', 'u_z_initial', 'theta_x_initial', 'theta_y_initial', 'theta_z_initial']].to_numpy()

        self.u0 = ar.flatten()



        return
        #print(self.df[mask]['depth']<-10)
        self.symbolic_K_matrix = mooring_tools.symbolic_stifness_matrix_total()
        l_k_yy = []
        l_k_zy = []
        l_k_yz = []
        l_k_zz = []

        f_get_K = mooring_tools.global_node_mooring_stiffness_generator(
            self.symbolic_K_matrix)


        for row in tqdm(self.df[mask].iloc()):
            y = 0
            t1=datetime.now()
            z_1 = mooring_tools.get_z1(
                abs(row['depth']), # depth of location
                abs(row['global_z']), # d0 at location
                theta, # theta at location
                E,
                A_N,
                self.q,
                s_anchor
            )
            t2 = datetime.now()

            K_numeric = f_get_K(0, z_1,
                abs(row['depth']),
                abs(row['global_z']),
                theta,
                E,
                A_N
            )

            t3 = datetime.now()

            k_yy = K_numeric[0,0]
            k_yz = K_numeric[0,1]
            k_zy = K_numeric[1,0]
            k_zz = K_numeric[1,1]

            l_k_yy.append(k_yy)
            l_k_yz.append(k_yz)
            l_k_zy.append(k_zy)
            l_k_zz.append(k_zz)

            s = f'''
            calculating z1 = {t2 - t1}
            getting value= {t3 - t2}
            '''
            #tqdm.write(s)

        self.df.loc[mask, 'k_yy'] = l_k_yy
        self.df.loc[mask, 'k_zy'] = l_k_zy
        self.df.loc[mask, 'k_yz'] = l_k_yz
        self.df.loc[mask, 'k_zz'] = l_k_zz



    def _add_bathymetry_sampler(self, lonlat_0, lonlat_1, file):
        row_0 = self.df.iloc[0]
        row_1 = self.df.iloc[-1]
        
        x_0 = row_0['global_x']
        y_0 = row_0['global_y']

        x_1 = row_1['global_x']
        y_1 = row_1['global_y']

        local_points = [(x_0, y_0), (x_1, y_1)]
        wgs84_pts = [lonlat_0, lonlat_1]

        sampler = BathymetrySampler(file, local_points, wgs84_pts, virtual_k_layer=10)
        self.sampler = sampler
        




        


