"""
Quantum Gyroscope — rotation sensing via atom interferometry.

Physics basis: Sagnac effect with matter waves.
Classical Sagnac:  Δφ_classical = 4πAΩ / λc
Quantum (N atoms): Δφ_quantum  = 4πAΩ / λ_dB  (λ_dB = de Broglie wavelength)

For entangled N-atom states (NOON states), sensitivity scales as 1/N (Heisenberg
limit) vs 1/√N (Standard Quantum Limit) for unentangled atoms.

This module simulates the measurement statistics of both regimes so the
navigator can compare accuracy as a function of atom count N.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class GyroscopeParams:
    area: float = 1e-4          # interferometer loop area [m²]
    atom_mass: float = 1.44e-25 # Rb-87 mass [kg]
    velocity: float = 0.01      # mean atom velocity [m/s]
    N: int = 1000               # number of atoms per shot
    entangled: bool = True      # True = Heisenberg limit, False = SQL
    readout_noise: float = 0.0  # additional readout noise [rad/s]


class QuantumGyroscope:
    """
    Simulates a quantum gyroscope measuring rotation rate Ω [rad/s].

    The output is a noisy estimate of Ω with standard deviation set by
    the chosen quantum limit (SQL or Heisenberg).
    """

    HBAR = 1.0545718e-34   # J·s
    H    = 6.62607015e-34  # J·s

    def __init__(self, params: GyroscopeParams | None = None, rng: np.random.Generator | None = None):
        self.p = params or GyroscopeParams()
        self.rng = rng or np.random.default_rng()

    @property
    def de_broglie_wavelength(self) -> float:
        """λ_dB = h / (m·v)"""
        return self.H / (self.p.atom_mass * self.p.velocity)

    @property
    def phase_per_rad_s(self) -> float:
        """
        Sagnac phase per unit rotation rate for matter waves [rad / (rad/s)].

        Matter-wave Sagnac: Δφ = 4π·m·A·Ω / h
        (Unlike optical Sagnac, there is no c — the de Broglie phase is
        proportional to mass, not inversely proportional to wavelength·c.)
        """
        return 4 * np.pi * self.p.atom_mass * self.p.area / self.H

    @property
    def sensitivity(self) -> float:
        """
        Minimum detectable rotation rate [rad/s] for one measurement.

        SQL:       σ_Ω = 1 / (√N · Δφ/Ω)
        Heisenberg: σ_Ω = 1 / (N  · Δφ/Ω)
        """
        scale = self.phase_per_rad_s
        if self.p.entangled:
            return 1.0 / (self.p.N * scale)
        else:
            return 1.0 / (np.sqrt(self.p.N) * scale)

    def measure(self, true_omega: float) -> float:
        """
        Return a single noisy measurement of rotation rate Ω.

        Parameters
        ----------
        true_omega : true rotation rate [rad/s]
        """
        noise = self.rng.normal(0.0, self.sensitivity + self.p.readout_noise)
        return true_omega + noise

    def measure_batch(self, true_omega: float, n_measurements: int) -> np.ndarray:
        """Return n_measurements independent rotation rate estimates."""
        noise = self.rng.normal(0.0, self.sensitivity + self.p.readout_noise, n_measurements)
        return true_omega + noise

    def sensitivity_ratio_vs_classical(self) -> float:
        """
        How many times more sensitive is this gyroscope vs a classical (SQL) one
        with the same number of atoms?

        Returns > 1 when entangled (Heisenberg limit gives √N improvement).
        Returns 1.0 when not entangled (same as classical).
        """
        if not self.p.entangled:
            return 1.0
        return np.sqrt(self.p.N)   # Heisenberg / SQL = N / √N = √N

    def __repr__(self) -> str:
        mode = "Heisenberg" if self.p.entangled else "SQL"
        return (f"QuantumGyroscope(N={self.p.N}, mode={mode}, "
                f"sensitivity={self.sensitivity:.3e} rad/s)")
