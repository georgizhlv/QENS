"""
Quantum Accelerometer — acceleration sensing via atom interferometry.

Physics basis: gravity gradiometry / inertial sensing with cold atom clouds.
The atom's wave function splits, travels two paths, and recombines.
Accumulated phase: Δφ = k_eff · a · T²

where k_eff is the effective laser wavenumber and T is the free-evolution time.

Entangled atom pairs (NOON states) push sensitivity to the Heisenberg limit:
σ_a = 1 / (N · k_eff · T²)  vs SQL: σ_a = 1 / (√N · k_eff · T²)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class AccelerometerParams:
    k_eff: float = 1.6e7       # effective laser wavenumber [rad/m] (780 nm Rb transition)
    T: float = 0.1             # free-evolution time [s]
    N: int = 1000              # atoms per shot
    entangled: bool = True
    readout_noise: float = 0.0  # [m/s²]


class QuantumAccelerometer:
    """
    Simulates a quantum accelerometer measuring acceleration a [m/s²].
    """

    def __init__(self, params: AccelerometerParams | None = None, rng: np.random.Generator | None = None):
        self.p = params or AccelerometerParams()
        self.rng = rng or np.random.default_rng()

    @property
    def sensitivity(self) -> float:
        """Minimum detectable acceleration [m/s²] per shot."""
        phase_per_accel = self.p.k_eff * self.p.T ** 2
        if self.p.entangled:
            return 1.0 / (self.p.N * phase_per_accel)
        else:
            return 1.0 / (np.sqrt(self.p.N) * phase_per_accel)

    def measure(self, true_accel: float) -> float:
        """Return a single noisy acceleration measurement [m/s²]."""
        noise = self.rng.normal(0.0, self.sensitivity + self.p.readout_noise)
        return true_accel + noise

    def measure_batch(self, true_accel: float, n_measurements: int) -> np.ndarray:
        noise = self.rng.normal(0.0, self.sensitivity + self.p.readout_noise, n_measurements)
        return true_accel + noise

    def sensitivity_ratio_vs_classical(self) -> float:
        if not self.p.entangled:
            return 1.0
        return np.sqrt(self.p.N)

    def __repr__(self) -> str:
        mode = "Heisenberg" if self.p.entangled else "SQL"
        return (f"QuantumAccelerometer(N={self.p.N}, T={self.p.T}s, mode={mode}, "
                f"sensitivity={self.sensitivity:.3e} m/s²)")
