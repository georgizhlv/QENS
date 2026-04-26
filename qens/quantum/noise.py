"""
Decoherence and noise models for entangled quantum sensors.

Real quantum sensors lose entanglement over time due to environment interaction.
This module models that degradation so we can compare ideal vs realistic quantum
navigation performance.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, phase_damping_error


class DephasedEntangledPair:
    """
    Entangled pair with exponential dephasing: ρ_off ∝ exp(−t/T2).

    Used by EntangledPair.decohere() — not meant to be instantiated directly.
    """

    def __init__(self, bell_state: str, shots: int, t: float, T2: float):
        self.bell_state = bell_state
        self.shots = shots
        self.t = t
        self.T2 = T2
        self._decay = np.exp(-t / T2) if T2 > 0 else 0.0

        # Build noise model: phase damping on both qubits
        error_rate = 1 - self._decay
        error_rate = float(np.clip(error_rate, 0.0, 1.0))
        # Use depolarizing noise: uniformly mixes all Pauli channels, reducing
        # ALL two-qubit correlations (including ⟨Z⊗Z⟩) by (1-p)^2.
        # Phase damping only destroys X/Y coherences, not ⟨Z⊗Z⟩ — so it would
        # not reduce the navigation-relevant correlation signal.
        noise_model = NoiseModel()
        if error_rate > 0:
            dep_err_1q = depolarizing_error(float(np.clip(error_rate * 0.75, 0, 1)), 1)
            dep_err_2q = depolarizing_error(float(np.clip(error_rate * 0.75, 0, 1)), 2)
            noise_model.add_all_qubit_quantum_error(dep_err_1q, ["h", "x", "z"])
            noise_model.add_all_qubit_quantum_error(dep_err_2q, ["cx"])
        self._noise_model = noise_model
        self._simulator = AerSimulator(noise_model=noise_model)

    def _build_bare_circuit(self) -> QuantumCircuit:
        from .entanglement import BellState
        qc = QuantumCircuit(2, 2)
        if self.bell_state == BellState.PHI_PLUS:
            qc.h(0); qc.cx(0, 1)
        elif self.bell_state == BellState.PHI_MINUS:
            qc.h(0); qc.cx(0, 1); qc.z(0)
        elif self.bell_state == BellState.PSI_PLUS:
            qc.h(0); qc.cx(0, 1); qc.x(1)
        elif self.bell_state == BellState.PSI_MINUS:
            qc.h(0); qc.cx(0, 1); qc.x(1); qc.z(0)
        qc.measure([0, 1], [0, 1])
        return qc

    def measure(self) -> dict[str, int]:
        qc = self._build_bare_circuit()
        job = self._simulator.run(qc, shots=self.shots)
        return job.result().get_counts()

    def correlation(self) -> float:
        counts = self.measure()
        total = sum(counts.values())
        corr = 0.0
        for bitstring, count in counts.items():
            q0 = int(bitstring[-1])
            q1 = int(bitstring[-2])
            corr += ((1 - 2*q0) * (1 - 2*q1)) * count / total
        return corr

    def theoretical_correlation(self) -> float:
        """Analytical prediction: C(t) = exp(−t/T2) for Φ+ state."""
        from .entanglement import BellState
        sign = 1 if self.bell_state in (BellState.PHI_PLUS, BellState.PHI_MINUS) else -1
        return sign * self._decay

    def __repr__(self) -> str:
        return (f"DephasedEntangledPair(state={self.bell_state}, "
                f"t={self.t:.3g}s, T2={self.T2:.3g}s, decay={self._decay:.3f})")
