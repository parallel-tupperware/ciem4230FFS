import geometry
import field_tools
import elements
import argparse
import json
from nodes import NodeLedger, Node
from elements import ElementLedger3DBeamStructure, MooringStiffnesssMatrix
import pandas as pd
from math import pi
from field_tools import HydroDynamicField
from datetime import datetime
from scipy.sparse.linalg import splu
from scipy.integrate import solve_ivp
import numpy as np
from scipy.sparse import eye, csr_matrix, bmat
import boundary_conditions as bc
import helpers

class Model:
    def __init__(self, d):
        self.conf = d
        self._set_main_geometry()
        t0 = datetime.now()
        self._set_hydro_dynamic_field()
        t1 = datetime.now()
        self._build_node_ledger()
        t2 = datetime.now()
        self._build_element_ledger()
        t3 = datetime.now()
        self._build_mooring_constructor()
        t4 = datetime.now()
        self._build_linear_system()
        t5 = datetime.now()
        self._build_jacobian()

        t0 = d['timestepping_properties']['t0']
        t1 = d['timestepping_properties']['t1']
        self.tspan = (t0, t1)

        # print('begin solving model')
        
        # t5 = datetime.now()

        # print(f'''
        # hydro field {t1 - t0}
        # node ledger {t2 - t1}
        # element ledger {t3 - t2}
        # mooring {t4 - t3}
        # ''')

    def _build_linear_system(self):
        self.K0 = self.element_ledger._K_global_linear
        self.M  = self.element_ledger._M_global_linear
        self.u  = self.node_ledger.u0

    def _build_element_ledger(self):
        print('assembling elements')
        self.element_ledger = ElementLedger3DBeamStructure(self.node_ledger, self.export_geometry(), self.hydro_field )

    def get_timedependent_hydro_load(self, t):
        
        return self.element_ledger.get_morisson_force_vector(t=t)

    def _build_mooring_constructor(self):
        d = self.conf['mooring_system_properties']
        val_A_n = d['n_strands_per_anchor'] * 0.25 * np.pi * d['diameter_mooring_line']**2
        theta_deg = d['theta_deg_anchor']
        self._mooring_constructor = MooringStiffnesssMatrix(
            E= d['youngs_modulus_aramid'],
            A_n = val_A_n,
            node_ledger = self.node_ledger,
            theta = np.deg2rad(theta_deg)
        )
        #self.element_ledger._build_mooring_constructor(self.node_ledger)



    def _build_jacobian(self):
        """
        Construct and store the constant Jacobian matrix J for the system:
        
            u' = v
            v' = M^{-1}(q(t) - K u)
        
        Resulting Jacobian:
        
            J = [ 0    I ]
                [ -A   0 ]
        
        where A = M^{-1} K
        """
        u0 = self.node_ledger.u0
        n = u0.size

        M = self.M
        K0 = self.K0
        K_mooring = self._mooring_constructor.get_nonlinear_stiffness(u0)
        K = K0 + K_mooring

        # LU factorization (reuse if already stored)
        if not hasattr(self, "M_lu"):
            from scipy.sparse.linalg import splu
            self.M_lu = splu(bc.apply_clamped_bc(M, M)[0])

        # Compute A = M^{-1} K
        # (simple and reliable version)
        A_dense = self.M_lu.solve(K.toarray())
        A = csr_matrix(A_dense)

        # Build block Jacobian
        Z = csr_matrix((n, n))
        I = eye(n, format="csr")

        self.J = bmat([
            [Z, I],
            [-A, Z]
        ], format="csr")

    def solve_system_linear(self, solver='RK45'):
        u0 = self.node_ledger.u0
        K0 = self.K0
        K_mooring = self._mooring_constructor.get_nonlinear_stiffness(u0)
        K = K0 + K_mooring
        #helpers.save_variable(K, 'K_FF_matrix.pkl')
        #helpers.save_variable(self.M, 'M_FF_matrix.pkl')
        #assert False

        K = bc.apply_clamped_bc(K, K)[0]
        q = self.get_timedependent_hydro_load
        t_span = self.tspan

        q_buoyancy = np.zeros_like(q(0))        
        qz0 = ( 25000 / self.conf['geometric_properties']['ds'] ) * self._properties_specific_buoyant_weight * self._properties_specific_buoyant_weight
        q_buoyancy[2::6] = qz0

        n = u0.size
        M_lu = self.M_lu

        # -----------------------------------
        # ✅ q(t) cache (OUTSIDE rhs!)
        # -----------------------------------
        dt_cache = 0.05  # seconds (adjust if needed)

        last_q = {
            "t": None,
            "val": None
        }

        def q_cached(t):
            if last_q["t"] is None or abs(t - last_q["t"]) > dt_cache:
                last_q["val"] = q(t)
                last_q["t"] = t
            return last_q["val"]

        # -----------------------------------
        # ✅ RHS
        # -----------------------------------
        def rhs(t, y):
            print(f'calculating t={t}')
            # optional: random progress print (cheap)
            #if np.random.random() < 0.01:
                #print(f"t = {t:.2f}")
            
            u = y[:n]
            v = y[n:]
            # print(u[:6])
            # print(u[-6:])
            # print('-----------------')
            #assert False
            f = bc.apply_clamped_bc_forcing(q_cached(t) + q_buoyancy) - K @ u

            # print(f[-6:])
            # xxx=  K @ u
            # print( xxx[-6:])
            #assert False
            a = M_lu.solve(f)

            # faster than concatenate
            out = np.empty_like(y)
            out[:n] = v
            out[n:] = a
            return out

        print(f'shape u0 = {u0.shape}')

        y0 = np.concatenate([u0, np.zeros_like(u0)])

        return solve_ivp(
            rhs,
            t_span,
            y0,
            method=solver,
            jac=self.J), u0

    def solve_system_no_cache(self, solver="RK45"):
        u0 = self.node_ledger.u0
        K0 = self.K0
        K_mooring =  self._mooring_constructor.get_nonlinear_stiffness(u0)
        K = K0 + K_mooring
        M = self.M
        q = self.get_timedependent_hydro_load
        t_span = self.tspan
        
        n = u0.size
        M_lu = self.M_lu
        
        def rhs(t, y):
                
            if np.random.random() < 0.01:  # 1% chance
                print(f"t = {t:.2f}")

            u = y[:n]
            v = y[n:]
            a = M_lu.solve(q(t) - K @ u)
            return np.concatenate([v, a])
        
        print(f'shape u0 = {u0.shape}')
        #print(u0)

        y0 = np.concatenate([u0, np.zeros_like(u0)])
        
        return solve_ivp(rhs, t_span, y0, method="BDF", jac=self.J)


    def solve_system_moored_nonlin(self, solver="RK45"):
        K = self.K0
        M = self.M
        q = self.get_timedependent_hydro_load
        u0 = self.node_ledger.u0
        q_buoyancy = np.zeros_like(q(0))        
        qz0 = ( 25000 / self.conf['geometric_properties']['ds'] ) * self._properties_specific_buoyant_weight * self._properties_specific_buoyant_weight
        q_buoyancy[2::6] = qz0
        t_span = self.tspan
        n = u0.size
        
        # Factorize mass matrix once
        M_lu = splu(M)
        
        def rhs(t, y):
            u = y[:n]
            v = y[n:]
            # Linear force
            f_lin = K @ u
            print(f'currently solving t = {t}')
            
            # Nonlinear stiffness matrix
            K_nonlin = self._mooring_constructor.get_nonlinear_stiffness(u)
            
            # Nonlinear force
            f_nonlin = K_nonlin @ u
            
            # Acceleration
            a = M_lu.solve(q(t) + q_buoyancy - f_lin - f_nonlin)
            
            return np.concatenate([v, a])
        
        y0 = np.concatenate([u0, np.zeros_like(u0)])
        

        return solve_ivp(rhs, t_span, y0, method=solver), u0




    def _set_hydro_dynamic_field(self):       

        d = self.conf['geometric_properties']

        x0 = d['x0']
        x1 = d['x1']

        z0 = d['z0']
        z1 = d['z1']

        r_y = d['Ry']
        r_z = d['Rz']
        d_s = d['ds']

        lat_0 = d['lat_0']
        lon_0 = d['lon_0']
        lat_1 = d['lat_1']
        lon_1 = d['lon_1']

        self._global_x0 = x0
        self._global_x1 = x1        
        self._global_y0 = 0
        self._global_y1 = 0
        self._global_z0 = z0
        self._global_z1 = z1
        self._global_p0 = (x0, 0)
        self._global_p1 = (x1, 0)

        self._wgs84_x0 = lon_0
        self._wgs84_x1 = lon_1
        self._wgs84_y0 = lat_0
        self._wgs84_y1 = lat_1
        self._wgs84_p0 = (lon_0, lat_0)
        self._wgs84_p1 = (lon_1, lat_1)

        self.hydro_field = HydroDynamicField(
            tif_path = self.conf['bathymetry_file'],
            local_pts = (self._global_p0, self._global_p1),
            wgs84_pts = (self._wgs84_p0, self._wgs84_p1),
            T = self.conf['hydro_load_properties']['T_wave_s'],
            a = self.conf['hydro_load_properties']['H_wave_m'] / 2,
            theta_w = np.deg2rad(self.conf['hydro_load_properties']['theta_wave_deg_wrt_yaxis']),
            theta_c = np.deg2rad(self.conf['hydro_load_properties']['theta_current_deg_wrt_yaxis']),
            U = self.conf['hydro_load_properties']['current_velocity'],
            z0 = self.conf['hydro_load_properties']['bottom_roughness']
        )
        

    def _set_main_geometry(self):
        d = self.conf['tunnel_properties']
        self._geometry_parameters_diameter = d['outer_diameter_tunnel']
        self._geometry_parameters_thickness = d['wall_thickness_tunnel']
        self._geometry_inner_diameter = self._geometry_parameters_diameter - 2 * self._geometry_parameters_thickness
        self._geometry_A_crosssection = (pi/4) * (self._geometry_parameters_diameter**2 - self._geometry_inner_diameter**2)
        
        self._Iy = (pi/64) * (self._geometry_parameters_diameter**4 - self._geometry_inner_diameter**4)
        self._Iz = self._Iy
        self._Ix = self._Iy * 2

        self._material_parameters_density = d['density_concrete']
        self._material_parameters_elasticity = d['youngs_modulus_concrete']
        self._water_density = d['density_water']
        self._poissons_ratio_concrete = d['poissons_ratio']

        self._properties_specific_self_weight = self._geometry_A_crosssection * self._material_parameters_density * 9.81
        self._properties_specific_self_buoyancy = self._water_density * 9.81 * pi * self._geometry_parameters_diameter**2 / 4
        self._properties_specific_buoyant_weight = self._properties_specific_self_buoyancy - self._properties_specific_self_weight

    def export_geometry(self):
        d = {
            'E_val' : self._material_parameters_elasticity,
            'A_val' : self._geometry_A_crosssection,
            'G_val' : self._material_parameters_elasticity / ( 2 * (1 + self._poissons_ratio_concrete)),
            'I_y_val' : self._Iy,
            'I_z_val' : self._Iz,
            'J_val' : self._Ix,
            'rho_val': self._material_parameters_density,
            'D' : self._geometry_parameters_diameter
        }
        return d


    def _build_node_ledger(self):
        d = self.conf['geometric_properties']

        x0 = d['x0']
        x1 = d['x1']

        z0 = d['z0']
        z1 = d['z1']

        r_y = d['Ry']
        r_z = d['Rz']
        d_s = d['ds']

        lat_0 = d['lat_0']
        lon_0 = d['lon_0']
        lat_1 = d['lat_1']
        lon_1 = d['lon_1']
        
        tiff_file_path = self.conf['bathymetry_file']

        print('Building Geometry')
        ar = geometry.create_double_arc_geometry(x0=x0,
                                                 x1=x1,
                                                 z0=z0,
                                                 z1=z1,
                                                 R_y=r_y,
                                                 R_z=r_z,
                                                 ds=d_s)
        


        df = pd.DataFrame(ar, columns=['global_x', 'global_y', 'global_z'])
        p0_lonlat = (lon_0, lat_0)
        p1_lonlat = (lon_1, lat_1)

        print('building node ledger')
        self.node_ledger = NodeLedger(df, p0_lonlat, p1_lonlat, tiff_file_path, self.conf, self._properties_specific_buoyant_weight, self.hydro_field.sampler)
        
        #mask = node_ledger.df['has_anchor']==True
        
        #node_ledger.df[mask].to_csv('anchor_specs.csv')

        #ar_coords = node_ledger.df[['global_x', 'global_y', 'global_z']]
        #print('building sampler')
        
        #print('sampler finished')
        #t0 = datetime.now()
        #ar_vel = hydro_field.inference_velocities(ar_coords, 1)
        #t1 = datetime.now()
        #print(f'sampling velocities cost : {t1 - t0}')
        #print(ar_vel)
        
        

        
        
        #element_ledger = ElementLedger()

