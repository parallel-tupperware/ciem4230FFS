import numpy as np
import rasterio
from rasterio.transform import rowcol
from datetime import datetime

class BathymetrySampler:
    def __init__(self, tif_path, local_pts, wgs84_pts, virtual_k_layer=None):
        self.src = rasterio.open(tif_path)
        self.band = self.src.read(1)

        P = np.array(local_pts, dtype=float)
        G = np.array(wgs84_pts, dtype=float)

        self._build_transform(P, G)

        # Optional k-layer
        self.k_band = None
        if virtual_k_layer is not None:
            self._build_k_layer(virtual_k_layer)

    def _build_transform(self, P, G):
        dP = P[1] - P[0]
        dG = G[1] - G[0]

        scale = np.linalg.norm(dG) / np.linalg.norm(dP)

        angle_P = np.arctan2(dP[1], dP[0])
        angle_G = np.arctan2(dG[1], dG[0])
        theta = angle_G - angle_P

        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])

        self.scale = scale
        self.R = R
        self.t = G[0] - scale * (R @ P[0])

    def _local_to_wgs84(self, coords):
        coords = np.asarray(coords)
        return (self.scale * (coords @ self.R.T)) + self.t

    def _solve_dispersion(self, depth, omega, g=9.81, max_iter=20):
        """
        Solve: omega^2 = g k tanh(k d)
        Vectorized Newton-Raphson
        """
        depth = np.maximum(depth, 1e-6)  # avoid zero depth

        # Initial guess (deep water)
        k = omega**2 / g * np.ones_like(depth)

        for _ in range(max_iter):
            kd = k * depth
            tanh_kd = np.tanh(kd)

            f = g * k * tanh_kd - omega**2
            df = g * tanh_kd + g * k * depth * (1 - tanh_kd**2)

            k = k - f / (df + 1e-12)

        return k

    
    def _build_k_layer(self, T):
        """
        Build virtual raster with k values
        """
        omega = 2 * np.pi / T

        depth = np.abs(self.band)  # ensure positive depth
        k = self._solve_dispersion(depth, omega)
        self.k_band = k

    def sample_depth(self, coords):
        coords = np.asarray(coords)
        lonlat = self._local_to_wgs84(coords)

        rows, cols = rowcol(self.src.transform, lonlat[:, 0], lonlat[:, 1])
        return self.band[rows, cols]

    def sample_k(self, coords):
        if self.k_band is None:
            raise ValueError("k-layer not initialized. Pass virtual_k_layer=T in constructor.")

        coords = np.asarray(coords)
        lonlat = self._local_to_wgs84(coords)

        rows, cols = rowcol(self.src.transform, lonlat[:, 0], lonlat[:, 1])
        return self.k_band[rows, cols]

    def close(self):
        self.src.close()
