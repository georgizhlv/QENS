"""
Classical Inertial Navigation System (INS).

Models a MEMS-grade IMU (Inertial Measurement Unit) as used in spacecraft.
Key property: errors accumulate over time (random walk + bias drift), making
long-duration deep-space navigation increasingly inaccurate without external
corrections — the fundamental motivation for quantum-enhanced sensors.

Error model (Allan deviation-based):
  Position error  ∝ σ_a · t²   (from accelerometer noise)
  Attitude error  ∝ σ_ω · t    (from gyroscope noise)
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class INSParams:
    # Gyroscope noise (angle random walk) [rad/s/√Hz]
    gyro_noise: float = 1e-4
    # Gyroscope bias stability [rad/s] — slow drift
    gyro_bias: float = 1e-5
    # Accelerometer noise [m/s²/√Hz]
    accel_noise: float = 1e-4
    # Accelerometer bias [m/s²]
    accel_bias: float = 1e-5


@dataclass
class INSState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))    # [m]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))    # [m/s]
    attitude: np.ndarray = field(default_factory=lambda: np.zeros(3))    # [rad] Euler angles
    time: float = 0.0


class InertialNavigationSystem:
    """
    Simulates a classical IMU integrating noisy measurements over time.

    The navigator propagates position/velocity/attitude by numerically
    integrating accelerometer and gyroscope readings at each timestep.
    Noise accumulates — position error grows as ~t².
    """

    def __init__(self, params: INSParams | None = None, rng: np.random.Generator | None = None):
        self.p = params or INSParams()
        self.rng = rng or np.random.default_rng()
        self.state = INSState()
        self._gyro_bias_current = self.rng.normal(0, self.p.gyro_bias, 3)
        self._accel_bias_current = self.rng.normal(0, self.p.accel_bias, 3)

    def reset(self, position=None, velocity=None, attitude=None):
        self.state = INSState(
            position=np.array(position if position is not None else [0., 0., 0.]),
            velocity=np.array(velocity if velocity is not None else [0., 0., 0.]),
            attitude=np.array(attitude if attitude is not None else [0., 0., 0.]),
        )

    def step(self, true_accel: np.ndarray, true_omega: np.ndarray, dt: float) -> INSState:
        """
        Propagate navigation state by one timestep dt [s].

        Parameters
        ----------
        true_accel : true acceleration vector [m/s²] in body frame
        true_omega : true angular velocity vector [rad/s] in body frame
        dt         : timestep [s]
        """
        # Measured values = truth + white noise + bias
        meas_accel = (true_accel
                      + self.rng.normal(0, self.p.accel_noise / np.sqrt(dt), 3)
                      + self._accel_bias_current)
        meas_omega = (true_omega
                      + self.rng.normal(0, self.p.gyro_noise / np.sqrt(dt), 3)
                      + self._gyro_bias_current)

        # Simple Euler integration (no attitude DCM for clarity)
        self.state.attitude += meas_omega * dt
        self.state.velocity += meas_accel * dt
        self.state.position += self.state.velocity * dt
        self.state.time += dt

        return INSState(
            position=self.state.position.copy(),
            velocity=self.state.velocity.copy(),
            attitude=self.state.attitude.copy(),
            time=self.state.time,
        )

    def position_error_bound(self, t: float) -> float:
        """
        Theoretical 1-sigma position error at time t [s].
        Dominated by accelerometer noise integrating twice: σ_pos ≈ σ_a·t²/√3
        """
        return self.p.accel_noise * t**2 / np.sqrt(3)

    def __repr__(self) -> str:
        return (f"INS(gyro_noise={self.p.gyro_noise:.1e} rad/s/√Hz, "
                f"accel_noise={self.p.accel_noise:.1e} m/s²/√Hz)")
