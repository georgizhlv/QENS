"""
Quantum entangled pair generation and Bell state measurement.

Models a source of entangled photon pairs used in quantum navigation sensors.
The key physics: measuring one qubit of an entangled pair instantly determines
the state of its partner, enabling noise-correlated differential measurements
that cancel common-mode errors — impossible with classical sensors.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace


class BellState:
    """The four maximally entangled two-qubit Bell states."""
    PHI_PLUS  = "Phi+"   # (|00> + |11>) / sqrt(2)  — standard entangled pair
    PHI_MINUS = "Phi-"   # (|00> - |11>) / sqrt(2)
    PSI_PLUS  = "Psi+"   # (|01> + |10>) / sqrt(2)
    PSI_MINUS = "Psi-"   # (|01> - |10>) / sqrt(2)


class EntangledPair:
    """
    Generates and measures entangled qubit pairs simulating a quantum sensor source.

    In a real quantum gyroscope/accelerometer, matter-wave interferometry with
    entangled atom pairs suppresses shot noise below the standard quantum limit
    (SQL), approaching the Heisenberg limit: σ ∝ 1/N instead of 1/√N.
    """

    def __init__(self, bell_state: str = BellState.PHI_PLUS, shots: int = 1024):
        self.bell_state = bell_state
        self.shots = shots
        self.simulator = AerSimulator()

    def build_circuit(self) -> QuantumCircuit:
        """Construct Bell state preparation circuit."""
        qc = QuantumCircuit(2, 2)

        if self.bell_state == BellState.PHI_PLUS:
            qc.h(0)
            qc.cx(0, 1)

        elif self.bell_state == BellState.PHI_MINUS:
            qc.h(0)
            qc.cx(0, 1)
            qc.z(0)

        elif self.bell_state == BellState.PSI_PLUS:
            qc.h(0)
            qc.cx(0, 1)
            qc.x(1)

        elif self.bell_state == BellState.PSI_MINUS:
            qc.h(0)
            qc.cx(0, 1)
            qc.x(1)
            qc.z(0)

        qc.measure([0, 1], [0, 1])
        return qc

    def measure(self) -> dict[str, int]:
        """Run the circuit and return measurement counts."""
        qc = self.build_circuit()
        job = self.simulator.run(qc, shots=self.shots)
        return job.result().get_counts()

    def correlation(self) -> float:
        """
        Compute the two-qubit correlation ⟨Z⊗Z⟩ from measurement statistics.

        Returns +1 for perfect correlation (Φ±), −1 for anti-correlation (Ψ±).
        Classical noise sources appear as a reduction from the ideal ±1 value.
        """
        counts = self.measure()
        total = sum(counts.values())
        corr = 0.0
        for bitstring, count in counts.items():
            # Qiskit returns bits as "q1 q0" (rightmost = qubit 0)
            q0 = int(bitstring[-1])
            q1 = int(bitstring[-2])
            z0 = 1 - 2 * q0   # maps |0⟩→+1, |1⟩→−1
            z1 = 1 - 2 * q1
            corr += (z0 * z1) * count / total
        return corr

    def entanglement_fidelity(self) -> float:
        """
        Estimate fidelity to the ideal Bell state using statevector simulation.

        Fidelity = |⟨ψ_ideal | ψ_sim⟩|² — 1.0 is perfect, below 0.9 means
        decoherence is significant.
        """
        qc_no_meas = self.build_circuit().remove_final_measurements(inplace=False)
        sv = Statevector.from_instruction(qc_no_meas)

        ideal = {
            BellState.PHI_PLUS:  np.array([1, 0, 0, 1]) / np.sqrt(2),
            BellState.PHI_MINUS: np.array([1, 0, 0, -1]) / np.sqrt(2),
            BellState.PSI_PLUS:  np.array([0, 1, 1, 0]) / np.sqrt(2),
            BellState.PSI_MINUS: np.array([0, 1, -1, 0]) / np.sqrt(2),
        }
        ideal_sv = Statevector(ideal[self.bell_state])
        return float(abs(sv.inner(ideal_sv)) ** 2)

    def decohere(self, t: float, T2: float) -> "EntangledPair":
        """
        Return a new EntangledPair that models dephasing decoherence.

        Dephasing reduces off-diagonal density matrix elements by exp(−t/T2).
        At t=0 the pair is ideal; at t≫T2 it collapses to a classical mixture.

        Parameters
        ----------
        t  : elapsed time since entanglement generation (seconds)
        T2 : dephasing time of the physical qubit (seconds)
              Trapped ions: ~1 s, Superconducting: ~100 µs, Photonic: ~ns
        """
        from .noise import DephasedEntangledPair
        return DephasedEntangledPair(self.bell_state, self.shots, t, T2)

    def __repr__(self) -> str:
        return f"EntangledPair(state={self.bell_state}, shots={self.shots})"
