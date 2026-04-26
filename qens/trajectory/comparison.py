"""
Three-way navigation comparison:
    1. Pure classical INS (no corrections)
    2. Classical INS + Kalman filter + pulsar fixes
    3. Quantum INS  + Kalman filter + pulsar fixes

This is the core scientific result of QENS Layer 2.
"""

import numpy as np
from dataclasses import dataclass, field
from .spacecraft import Spacecraft, OrbitalParams
from .kalman import ErrorStateKalmanFilter
from ..classical.ins import InertialNavigationSystem, INSParams
from ..quantum.gyroscope import QuantumGyroscope, GyroscopeParams
from ..quantum.accelerometer import QuantumAccelerometer, AccelerometerParams
from ..sources.pulsar import PulsarNavSystem


@dataclass
class ComparisonResult:
    times: np.ndarray
    # Position errors vs zero-noise reference [m]
    pure_classical_errors: np.ndarray
    kf_classical_errors: np.ndarray
    kf_quantum_errors: np.ndarray
    # 1-sigma covariance bounds from KF [m]
    kf_classical_sigma: np.ndarray
    kf_quantum_sigma: np.ndarray
    # Number of pulsar fixes received
    n_fixes: int
    pulsar_accuracy: float


def build_quantum_ins_params(N: int = 1000) -> INSParams:
    gyro  = QuantumGyroscope(GyroscopeParams(N=N, entangled=True))
    accel = QuantumAccelerometer(AccelerometerParams(N=N, entangled=True))
    return INSParams(
        gyro_noise=gyro.sensitivity,
        gyro_bias=1e-9,
        accel_noise=accel.sensitivity,
        accel_bias=1e-9,
    )


class NavigationComparison:
    """
    Runs all three navigators on the same trajectory simultaneously.

    All three share the same true acceleration inputs. The difference is:
    - Pure classical: integrates noisy MEMS measurements, no corrections
    - KF classical:  same MEMS + Kalman filter corrected by pulsar fixes
    - KF quantum:    quantum sensors + Kalman filter corrected by pulsar fixes
    """

    # Tactical-grade MEMS IMU — realistic for low-cost CubeSats and small probes.
    # Navigation-grade units (1e-4 m/s²/√Hz) are too accurate relative to current
    # pulsar fix precision; tactical-grade demonstrates the XNAV benefit clearly.
    CLASSICAL_PARAMS = INSParams(
        gyro_noise=1e-1,
        gyro_bias=1e-3,
        accel_noise=1e-1,
        accel_bias=1e-3,
    )

    def __init__(
        self,
        orbital_params: OrbitalParams | None = None,
        N_atoms: int = 1000,
        seed: int = 42,
    ):
        self.orbital_params = orbital_params or OrbitalParams()
        self.N_atoms = N_atoms
        rng = np.random.default_rng(seed)

        quantum_params = build_quantum_ins_params(N_atoms)

        # pure_classical and kf_classical share the same seed so they produce
        # identical noise — the only difference in their outputs is KF corrections.
        self.pure_classical = InertialNavigationSystem(self.CLASSICAL_PARAMS, np.random.default_rng(seed))
        self.kf_classical   = InertialNavigationSystem(self.CLASSICAL_PARAMS, np.random.default_rng(seed))
        self.kf_quantum     = InertialNavigationSystem(quantum_params,         np.random.default_rng(seed + 1))
        self.ref_ins        = InertialNavigationSystem(
            INSParams(gyro_noise=0, gyro_bias=0, accel_noise=0, accel_bias=0),
            np.random.default_rng(0),
        )

        self.pulsar = PulsarNavSystem(rng=np.random.default_rng(seed + 3))

        # Separate KFs for classical and quantum (different process noise)
        self._kf_cls_accel_noise = self.CLASSICAL_PARAMS.accel_noise
        self._kf_qnt_accel_noise = quantum_params.accel_noise

    def run(self, duration: float = 3600.0, dt: float = 1.0) -> ComparisonResult:
        """
        Parameters
        ----------
        duration : simulation time [s] — default 1 hour for deep-space scenario
        dt       : integration timestep [s]
        """
        sc = Spacecraft(self.orbital_params)

        r0, v0 = sc.p.r0, sc.p.v0
        for ins in (self.pure_classical, self.kf_classical, self.kf_quantum, self.ref_ins):
            ins.reset(r0, v0)

        kf_cls = ErrorStateKalmanFilter(
            accel_noise=self._kf_cls_accel_noise,
            meas_noise=self.pulsar.combined_accuracy,
        )
        kf_qnt = ErrorStateKalmanFilter(
            accel_noise=self._kf_qnt_accel_noise,
            meas_noise=self.pulsar.combined_accuracy,
        )

        steps = int(duration / dt)
        times             = np.zeros(steps)
        pure_cls_errors   = np.zeros(steps)
        kf_cls_errors     = np.zeros(steps)
        kf_qnt_errors     = np.zeros(steps)
        kf_cls_sigma      = np.zeros(steps)
        kf_qnt_sigma      = np.zeros(steps)

        # Positions that the KF navigators report (INS + KF correction)
        kf_cls_pos = r0.astype(float).copy()
        kf_qnt_pos = r0.astype(float).copy()
        n_fixes = 0

        for i in range(steps):
            _, _, a_true = sc.step(dt)
            omega_true = np.zeros(3)
            t = sc.t

            ref_state  = self.ref_ins.step(a_true, omega_true, dt)
            pure_state = self.pure_classical.step(a_true, omega_true, dt)
            cls_state  = self.kf_classical.step(a_true, omega_true, dt)
            qnt_state  = self.kf_quantum.step(a_true, omega_true, dt)

            # Propagate both KFs
            kf_cls.predict(dt)
            kf_qnt.predict(dt)

            # Apply pulsar fix if available at this timestep
            if self.pulsar.should_fix(t, dt):
                true_pos = ref_state.position
                fix = self.pulsar.get_fix(true_pos)

                corr_cls = kf_cls.update(fix, cls_state.position)
                corr_qnt = kf_qnt.update(fix, qnt_state.position)

                # Feed corrections back into INS position state
                cls_state.position += corr_cls
                self.kf_classical.state.position += corr_cls
                qnt_state.position += corr_qnt
                self.kf_quantum.state.position += corr_qnt

                if i > 0:
                    n_fixes += 1

            ref_pos = ref_state.position
            times[i]           = t
            pure_cls_errors[i] = np.linalg.norm(pure_state.position - ref_pos)
            kf_cls_errors[i]   = np.linalg.norm(cls_state.position  - ref_pos)
            kf_qnt_errors[i]   = np.linalg.norm(qnt_state.position  - ref_pos)
            kf_cls_sigma[i]    = kf_cls.position_uncertainty
            kf_qnt_sigma[i]    = kf_qnt.position_uncertainty

        return ComparisonResult(
            times=times,
            pure_classical_errors=pure_cls_errors,
            kf_classical_errors=kf_cls_errors,
            kf_quantum_errors=kf_qnt_errors,
            kf_classical_sigma=kf_cls_sigma,
            kf_quantum_sigma=kf_qnt_sigma,
            n_fixes=n_fixes,
            pulsar_accuracy=self.pulsar.combined_accuracy,
        )
