import numpy as np
from numba import njit, prange
from bathymetry_tools import BathymetrySampler


# ===============================
# FIELD CLASS
# ===============================
class HydroDynamicField:
    def __init__(
        self,
        tif_path,
        local_pts,
        wgs84_pts,
        T,
        a,
        theta_w,
        theta_c,
        U,
        z0,
        ):
        """
        Velocity and acceleration field consisting of:
        - logarithmic current profile,
        - linear wave orbital velocity,
        - second-order Stokes correction.

        Coordinate convention
        ---------------------
        z = -d   -> seabed
        z =  0   -> still water level / free surface

        Direction convention
        --------------------
        theta_w and theta_c are measured from the +y axis.

        theta = 0      -> +y direction
        theta = pi / 2 -> +x direction

        Parameters
        ----------
        tif_path : str
            Path to bathymetry GeoTIFF.

        local_pts : array-like
            Local coordinate reference points.

        wgs84_pts : array-like
            WGS84 coordinate reference points.

        T : float
            Wave period [s].

        a : float
            Wave amplitude [m].

        theta_w : float
            Wave direction angle measured from +y axis [rad].

        theta_c : float
            Current direction angle measured from +y axis [rad].

        U : float
            Current velocity magnitude at z = 0 [m/s].

        z0 : float
            Bed roughness length [m].
        """

        self.a = a
        self.theta_w = theta_w
        self.theta_c = theta_c
        self.U = U
        self.z0 = z0

        self.omega = 2.0 * np.pi / T

        # Direction vector for angles measured from +y:
        # e = [sin(theta), cos(theta)]
        self.sin_tw = np.sin(theta_w)
        self.cos_tw = np.cos(theta_w)

        self.sin_tc = np.sin(theta_c)
        self.cos_tc = np.cos(theta_c)

        # Instantiate sampler with k-layer
        self.sampler = BathymetrySampler(
            tif_path,
            local_pts,
            wgs84_pts,
            virtual_k_layer=T,
        )

    def inference_velocities(self, coords, t, relative_to=None):
        """
        Compute velocity vectors at coordinates.

        Parameters
        ----------
        coords : ndarray
            Shape (N, 3), containing coordinates [x, y, z].

        t : float
            Time [s].

        relative_to : ndarray, optional
            Shape (N, 3), containing velocity vectors [u_x, u_y, u_z].

            If provided, this returns:

                fluid_velocity - relative_to

        Returns
        -------
        out : ndarray
            Shape (N, 3), containing velocity vectors [u_x, u_y, u_z].
        """

        coords = np.asarray(coords, dtype=np.float64)

        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must have shape (N, 3)")

        if relative_to is not None:
            relative_to = np.asarray(relative_to, dtype=np.float64)

            if relative_to.ndim != 2 or relative_to.shape[1] != 3:
                raise ValueError("relative_to must have shape (N, 3)")

            if relative_to.shape[0] != coords.shape[0]:
                raise ValueError(
                    "relative_to must have the same number of rows as coords"
                )

        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        # ---------------------------
        # Sample bathymetry and wave number
        # ---------------------------
        d = np.abs(self.sampler.sample_depth(coords[:, :2]))
        k = self.sampler.sample_k(coords[:, :2])

        d = np.asarray(d, dtype=np.float64)
        k = np.asarray(k, dtype=np.float64)

        # ---------------------------
        # Compute fluid velocity
        # ---------------------------
        ux, uy, uz = _compute_velocity_kernel(
            x,
            y,
            z,
            d,
            k,
            t,
            self.a,
            self.omega,
            self.sin_tw,
            self.cos_tw,
            self.sin_tc,
            self.cos_tc,
            self.U,
            self.z0,
        )

        out = np.empty((coords.shape[0], 3), dtype=np.float64)
        out[:, 0] = ux
        out[:, 1] = uy
        out[:, 2] = uz

        # ---------------------------
        # Optional relative velocity
        # ---------------------------
        if relative_to is not None:
            out -= relative_to

        return out

    def inference_accelerations(self, coords, t, relative_to=None):
        """
        Compute acceleration vectors at coordinates.

        The acceleration field is defined as:

            acceleration = partial velocity / partial time

        or:

            a = ∂u / ∂t

        The current is steady, so only the wave terms contribute.

        Parameters
        ----------
        coords : ndarray
            Shape (N, 3), containing coordinates [x, y, z].

        t : float
            Time [s].

        relative_to : ndarray, optional
            Shape (N, 3), containing acceleration vectors [a_x, a_y, a_z].

            If provided, this returns:

                fluid_acceleration - relative_to

        Returns
        -------
        out : ndarray
            Shape (N, 3), containing acceleration vectors [a_x, a_y, a_z].
        """

        coords = np.asarray(coords, dtype=np.float64)

        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must have shape (N, 3)")

        if relative_to is not None:
            relative_to = np.asarray(relative_to, dtype=np.float64)

            if relative_to.ndim != 2 or relative_to.shape[1] != 3:
                raise ValueError("relative_to must have shape (N, 3)")

            if relative_to.shape[0] != coords.shape[0]:
                raise ValueError(
                    "relative_to must have the same number of rows as coords"
                )

        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        # ---------------------------
        # Sample bathymetry and wave number
        # ---------------------------
        d = np.abs(self.sampler.sample_depth(coords[:, :2]))
        k = self.sampler.sample_k(coords[:, :2])

        d = np.asarray(d, dtype=np.float64)
        k = np.asarray(k, dtype=np.float64)

        # ---------------------------
        # Compute fluid acceleration
        # ---------------------------
        ax, ay, az = _compute_acceleration_kernel(
            x,
            y,
            z,
            d,
            k,
            t,
            self.a,
            self.omega,
            self.sin_tw,
            self.cos_tw,
            self.z0,
        )

        out = np.empty((coords.shape[0], 3), dtype=np.float64)
        out[:, 0] = ax
        out[:, 1] = ay
        out[:, 2] = az

        # ---------------------------
        # Optional relative acceleration
        # ---------------------------
        if relative_to is not None:
            out -= relative_to

        return out

    def close(self):
        self.sampler.close()


# ===============================
# VELOCITY KERNEL
# ===============================
@njit(parallel=True, fastmath=True)
def _compute_velocity_kernel(
    x,
    y,
    z,
    d,
    k,
    t,
    a,
    omega,
    sin_tw,
    cos_tw,
    sin_tc,
    cos_tc,
    U,
    z0,
):
    N = x.shape[0]

    u_x = np.empty(N)
    u_y = np.empty(N)
    u_z = np.empty(N)

    for i in prange(N):
        xi = x[i]
        yi = y[i]
        zi = z[i]
        di = d[i]
        ki = k[i]

        # ---------------------------
        # Vertical coordinate
        # ---------------------------
        # zpd is height above bed.
        #
        # z = -d -> zpd = 0
        # z =  0 -> zpd = d
        zpd = zi + di

        # ---------------------------
        # Stability guards
        # ---------------------------
        if di < 1e-6:
            di = 1e-6

        z0_eff = z0
        if z0_eff < 1e-9:
            z0_eff = 1e-9

        # Avoid invalid log near or below roughness height.
        if zpd <= z0_eff:
            zpd = z0_eff + 1e-6

        # Clamp above still water level.
        if zpd > di:
            zpd = di

        # Effective depth for current normalization.
        di_eff = di
        if di_eff <= z0_eff:
            di_eff = z0_eff + 1e-6

        log_surface = np.log(di_eff / z0_eff)
        if log_surface < 1e-6:
            log_surface = 1e-6

        # Wave depth parameter.
        kd = ki * di
        if kd < 1e-6:
            kd = 1e-6

        # ---------------------------
        # Phase
        # ---------------------------
        # Direction convention:
        #
        # wave propagation direction = [sin(theta_w), cos(theta_w)]
        #
        phase = ki * (xi * sin_tw + yi * cos_tw) - omega * t

        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)

        cos_2phase = np.cos(2.0 * phase)
        sin_2phase = np.sin(2.0 * phase)

        # ---------------------------
        # Hyperbolic terms
        # ---------------------------
        kz = ki * zpd

        cosh_kz = np.cosh(kz)
        sinh_kz = np.sinh(kz)

        cosh_2kz = np.cosh(2.0 * kz)
        sinh_2kz = np.sinh(2.0 * kz)

        sinh_kd = np.sinh(kd)
        if sinh_kd < 1e-6:
            sinh_kd = 1e-6

        sinh_kd_sq = sinh_kd * sinh_kd

        # ---------------------------
        # Current velocity
        # ---------------------------
        # U is specified at z = 0.
        #
        # At z = 0:
        #     zpd = di
        #
        # Therefore this normalized log profile gives:
        #     Uc = U
        #
        Uc = U * np.log(zpd / z0_eff) / log_surface

        # ---------------------------
        # Linear wave velocity
        # ---------------------------
        U1 = a * omega * (cosh_kz / sinh_kd) * cos_phase
        W1 = a * omega * (sinh_kz / sinh_kd) * sin_phase

        # ---------------------------
        # Second-order Stokes velocity correction
        # ---------------------------
        U2 = (
            a
            * a
            * omega
            * ki
            * (cosh_2kz / (2.0 * sinh_kd_sq))
            * cos_2phase
        )

        W2 = (
            a
            * a
            * omega
            * ki
            * (sinh_2kz / (2.0 * sinh_kd_sq))
            * sin_2phase
        )

        Uw = U1 + U2
        W = W1 + W2

        # ---------------------------
        # Projection to x/y components
        # ---------------------------
        u_x[i] = Uc * sin_tc + Uw * sin_tw
        u_y[i] = Uc * cos_tc + Uw * cos_tw
        u_z[i] = W

    return u_x, u_y, u_z


# ===============================
# ACCELERATION KERNEL
# ===============================
@njit(parallel=True, fastmath=True)
def _compute_acceleration_kernel(
    x,
    y,
    z,
    d,
    k,
    t,
    a,
    omega,
    sin_tw,
    cos_tw,
    z0,
):
    N = x.shape[0]

    a_x = np.empty(N)
    a_y = np.empty(N)
    a_z = np.empty(N)

    for i in prange(N):
        xi = x[i]
        yi = y[i]
        zi = z[i]
        di = d[i]
        ki = k[i]

        # ---------------------------
        # Vertical coordinate
        # ---------------------------
        # zpd is height above bed.
        #
        # z = -d -> zpd = 0
        # z =  0 -> zpd = d
        zpd = zi + di

        # ---------------------------
        # Stability guards
        # ---------------------------
        if di < 1e-6:
            di = 1e-6

        z0_eff = z0
        if z0_eff < 1e-9:
            z0_eff = 1e-9

        if zpd <= z0_eff:
            zpd = z0_eff + 1e-6

        if zpd > di:
            zpd = di

        kd = ki * di
        if kd < 1e-6:
            kd = 1e-6

        # ---------------------------
        # Phase
        # ---------------------------
        phase = ki * (xi * sin_tw + yi * cos_tw) - omega * t

        cos_phase = np.cos(phase)
        sin_phase = np.sin(phase)

        cos_2phase = np.cos(2.0 * phase)
        sin_2phase = np.sin(2.0 * phase)

        # ---------------------------
        # Hyperbolic terms
        # ---------------------------
        kz = ki * zpd

        cosh_kz = np.cosh(kz)
        sinh_kz = np.sinh(kz)

        cosh_2kz = np.cosh(2.0 * kz)
        sinh_2kz = np.sinh(2.0 * kz)

        sinh_kd = np.sinh(kd)
        if sinh_kd < 1e-6:
            sinh_kd = 1e-6

        sinh_kd_sq = sinh_kd * sinh_kd

        # =====================================================
        # Acceleration = partial velocity / partial time
        # =====================================================
        #
        # phase = k(x sin(theta_w) + y cos(theta_w)) - omega t
        #
        # d phase / dt = -omega
        #
        # d cos(phase) / dt =  omega sin(phase)
        # d sin(phase) / dt = -omega cos(phase)
        #
        # d cos(2 phase) / dt =  2 omega sin(2 phase)
        # d sin(2 phase) / dt = -2 omega cos(2 phase)
        #
        # The current is steady, so:
        #
        # d current / dt = 0
        #

        # ---------------------------
        # Linear horizontal wave acceleration
        # ---------------------------
        # U1 = A1 cos(phase)
        A1 = a * omega * (cosh_kz / sinh_kd)

        # dU1/dt = A1 * omega * sin(phase)
        dU1_dt = A1 * omega * sin_phase

        # ---------------------------
        # Linear vertical wave acceleration
        # ---------------------------
        # W1 = B1 sin(phase)
        B1 = a * omega * (sinh_kz / sinh_kd)

        # dW1/dt = -B1 * omega * cos(phase)
        dW1_dt = -B1 * omega * cos_phase

        # ---------------------------
        # Second-order horizontal wave acceleration
        # ---------------------------
        # U2 = A2 cos(2 phase)
        A2 = (
            a
            * a
            * omega
            * ki
            * (cosh_2kz / (2.0 * sinh_kd_sq))
        )

        # dU2/dt = 2 omega A2 sin(2 phase)
        dU2_dt = 2.0 * omega * A2 * sin_2phase

        # ---------------------------
        # Second-order vertical wave acceleration
        # ---------------------------
        # W2 = B2 sin(2 phase)
        B2 = (
            a
            * a
            * omega
            * ki
            * (sinh_2kz / (2.0 * sinh_kd_sq))
        )

        # dW2/dt = -2 omega B2 cos(2 phase)
        dW2_dt = -2.0 * omega * B2 * cos_2phase

        # Total horizontal acceleration along wave direction.
        Aw = dU1_dt + dU2_dt

        # Total vertical acceleration.
        Az = dW1_dt + dW2_dt

        # Project horizontal acceleration to x/y.
        # Current acceleration is zero because current is steady.
        a_x[i] = Aw * sin_tw
        a_y[i] = Aw * cos_tw
        a_z[i] = Az

    return a_x, a_y, a_z