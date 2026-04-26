"""
Navigator: compares classical INS vs quantum-enhanced INS on a real trajectory.

Runs a spacecraft simulation, feeds true accelerations/rotations into both
navigators, and tracks their position error over time. This is the core
scientific result of QENS.
"""

import numpy as np
from ..classical.ins import InertialNavigationSystem, INSParams
from ..quantum.gyroscope import QuantumGyroscope, GyroscopeParams
from ..quantum.accelerometer import QuantumAccelerometer, AccelerometerParams
from .spacecraft import Spacecraft, OrbitalParams
from dataclasses import dataclass


@dataclass
class NavigationResult:
    times: np.ndarray
    true_positions: np.ndarray          # [m] ground truth
    classical_positions: np.ndarray     # [m] estimated by classical INS
    quantum_positions: np.ndarray       # [m] estimated by quantum INS
    classical_errors: np.ndarray        # [m] |est - true|
    quantum_errors: np.ndarray          # [m] |est - true|


class Navigator:
    """
    Runs a navigation comparison experiment.

    Both navigators start with perfect initial conditions.
    They integrate sensor measurements over time and accumulate error.
    The quantum navigator uses lower-noise sensors (Heisenberg limit).
    """

    def __init__(
        self,
        orbital_params: OrbitalParams | None = None,
        classical_ins_params: INSParams | None = None,
        quantum_gyro_params: GyroscopeParams | None = None,
        quantum_accel_params: AccelerometerParams | None = None,
        seed: int = 42,
    ):
        self.spacecraft = Spacecraft(orbital_params)
        rng_c = np.random.default_rng(seed)
        rng_q = np.random.default_rng(seed + 1)

        self.classical_ins = InertialNavigationSystem(classical_ins_params, rng_c)

        # Quantum INS: replace gyro/accel noise with quantum sensor sensitivity
        gyro = QuantumGyroscope(quantum_gyro_params, rng_q)
        accel = QuantumAccelerometer(quantum_accel_params, rng_q)

        # Build an INS-equivalent params using quantum sensor sensitivities
        q_ins_params = INSParams(
            gyro_noise=gyro.sensitivity,
            gyro_bias=1e-7,              # quantum sensors have much lower bias
            accel_noise=accel.sensitivity,
            accel_bias=1e-7,
        )
        self.quantum_ins = InertialNavigationSystem(q_ins_params, rng_q)

    def run(self, duration: float = 600.0, dt: float = 1.0) -> NavigationResult:
        """
        Run the simulation for `duration` seconds with `dt` second steps.

        Parameters
        ----------
        duration : total simulation time [s]
        dt       : integration timestep [s]
        """
        sc = Spacecraft(self.spacecraft.p)
        self.classical_ins.reset(sc.p.r0, sc.p.v0)
        self.quantum_ins.reset(sc.p.r0, sc.p.v0)

        # Zero-noise reference INS: shares Euler integration scheme with both
        # navigators, so systematic integration errors cancel. What remains is
        # purely the sensor noise contribution — the quantity we want to compare.
        ref_ins = InertialNavigationSystem(
            INSParams(gyro_noise=0.0, gyro_bias=0.0, accel_noise=0.0, accel_bias=0.0),
            np.random.default_rng(0),
        )
        ref_ins.reset(sc.p.r0, sc.p.v0)

        steps = int(duration / dt)
        times = np.zeros(steps)
        true_pos = np.zeros((steps, 3))
        cls_pos = np.zeros((steps, 3))
        qnt_pos = np.zeros((steps, 3))

        for i in range(steps):
            _, _, a_true = sc.step(dt)
            omega_true = np.zeros(3)

            ref_state = ref_ins.step(a_true, omega_true, dt)
            cls_state = self.classical_ins.step(a_true, omega_true, dt)
            qnt_state = self.quantum_ins.step(a_true, omega_true, dt)

            times[i] = sc.t
            true_pos[i] = ref_state.position
            cls_pos[i] = cls_state.position
            qnt_pos[i] = qnt_state.position

        cls_errors = np.linalg.norm(cls_pos - true_pos, axis=1)
        qnt_errors = np.linalg.norm(qnt_pos - true_pos, axis=1)

        return NavigationResult(times, true_pos, cls_pos, qnt_pos, cls_errors, qnt_errors)
