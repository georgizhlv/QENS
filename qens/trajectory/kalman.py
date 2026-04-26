"""
Error-State Kalman Filter for inertial navigation.

State vector: δx = [δr (3), δv (3)] — position and velocity *errors* relative
to the INS solution. The KF estimates the error, not the absolute state.

Process model (INS error propagation):
    δr(t+dt) = δr(t) + δv(t)*dt
    δv(t+dt) = δv(t) + w_a           (w_a ~ N(0, Q_a))

Process noise Q_a comes from the accelerometer noise — classical or quantum.

Measurement model (position fix, e.g. pulsar timing):
    z = δr + v_meas                  (v_meas ~ N(0, R))

After each update the estimated error is fed back into the INS solution,
keeping the navigator close to truth.

Why error-state KF?
    The INS handles large-signal nonlinear dynamics (RK4). The KF handles
    small, nearly-linear errors. This split is standard in aerospace INS/GNSS
    integration (loosely-coupled architecture).
"""

import numpy as np


class ErrorStateKalmanFilter:
    """
    6-state linear KF operating on INS position/velocity errors.

    Parameters
    ----------
    accel_noise : accelerometer noise [m/s²] — sets process noise Q
    meas_noise  : position measurement noise [m] — sets measurement noise R
    """

    def __init__(self, accel_noise: float, meas_noise: float):
        self.accel_noise = accel_noise
        self.meas_noise = meas_noise

        # State estimate: [δr(3), δv(3)]
        self.x = np.zeros(6)

        # Covariance — start with zero initial error (perfect init)
        self.P = np.zeros((6, 6))

        # Measurement matrix: we observe position error only
        self.H = np.zeros((3, 6))
        self.H[:, :3] = np.eye(3)

        # Measurement noise covariance
        self.R = (meas_noise ** 2) * np.eye(3)

    def _build_F(self, dt: float) -> np.ndarray:
        """State transition matrix for error propagation."""
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        return F

    def _build_Q(self, dt: float) -> np.ndarray:
        """
        Process noise covariance from accelerometer noise.
        Discretised for timestep dt.
        """
        q = self.accel_noise ** 2
        Q = np.zeros((6, 6))
        # Position noise: (dt²/2)² * q  — double integration of accel noise
        Q[:3, :3] = q * (dt ** 4 / 4) * np.eye(3)
        # Velocity noise: dt² * q
        Q[3:, 3:] = q * (dt ** 2) * np.eye(3)
        # Cross terms
        Q[:3, 3:] = q * (dt ** 3 / 2) * np.eye(3)
        Q[3:, :3] = q * (dt ** 3 / 2) * np.eye(3)
        return Q

    def predict(self, dt: float):
        """Propagate error state and covariance by one timestep."""
        F = self._build_F(dt)
        Q = self._build_Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, position_fix: np.ndarray, ins_position: np.ndarray):
        """
        Incorporate a position measurement (e.g. from pulsar timing).

        Parameters
        ----------
        position_fix  : measured position [m] in inertial frame
        ins_position  : current INS position estimate [m]

        Returns the position correction to apply to the INS.
        """
        # Innovation: measured position minus current INS position
        z = position_fix - ins_position   # this is the *observed* position error

        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state estimate and covariance
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(6) - K @ self.H) @ self.P

        # Return the position correction to feed back into the INS
        return self.x[:3].copy()

    @property
    def position_uncertainty(self) -> float:
        """1-sigma position uncertainty [m] (isotropic approximation)."""
        return float(np.sqrt(np.trace(self.P[:3, :3]) / 3))

    @property
    def velocity_uncertainty(self) -> float:
        """1-sigma velocity uncertainty [m/s]."""
        return float(np.sqrt(np.trace(self.P[3:, 3:]) / 3))
