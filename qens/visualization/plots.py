"""
Visualization for QENS simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ..trajectory.navigator import NavigationResult
from ..trajectory.comparison import ComparisonResult


def plot_navigation_comparison(result: NavigationResult, save_path: str | None = None):
    """
    Side-by-side comparison of classical vs quantum navigation error over time.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("QENS — Quantum vs Classical Navigation Error", fontsize=14, fontweight='bold')

    t_min = result.times / 60  # convert to minutes

    # Left: position error over time
    ax = axes[0]
    ax.semilogy(t_min, result.classical_errors, label="Classical INS", color="#e74c3c", linewidth=1.5)
    ax.semilogy(t_min, result.quantum_errors,   label="Quantum INS (Heisenberg)", color="#2ecc71", linewidth=1.5)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Position Error [m]")
    ax.set_title("Navigation Error Growth")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: improvement ratio
    ax2 = axes[1]
    ratio = result.classical_errors / np.maximum(result.quantum_errors, 1e-10)
    ax2.plot(t_min, ratio, color="#3498db", linewidth=1.5)
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label="No improvement")
    ax2.set_xlabel("Time [min]")
    ax2.set_ylabel("Error Ratio (Classical / Quantum)")
    ax2.set_title("Quantum Advantage Factor")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_entanglement_decoherence(T2: float = 1.0, save_path: str | None = None):
    """
    Shows how entanglement correlation decays with time for different T2 values.
    """
    from ..quantum.entanglement import EntangledPair, BellState

    fig, ax = plt.subplots(figsize=(9, 5))
    times = np.linspace(0, 3 * T2, 50)

    for t in times:
        pair = EntangledPair(BellState.PHI_PLUS).decohere(t, T2)

    # Theoretical curve
    t_arr = np.linspace(0, 3 * T2, 300)
    corr_theory = np.exp(-t_arr / T2)
    ax.plot(t_arr / T2, corr_theory, 'b-', linewidth=2, label=r"Theory: $e^{-t/T_2}$")

    # Simulated points
    simulated = []
    t_sim = np.linspace(0, 3 * T2, 20)
    for t in t_sim:
        pair = EntangledPair(BellState.PHI_PLUS, shots=2048).decohere(t, T2)
        simulated.append(pair.correlation())
    ax.scatter(t_sim / T2, simulated, color='red', zorder=5, label="Simulated (Qiskit)")

    ax.set_xlabel(r"$t / T_2$")
    ax.set_ylabel(r"Correlation $\langle Z \otimes Z \rangle$")
    ax.set_title(f"Entanglement Decoherence (T₂ = {T2} s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle=':')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_sensor_sensitivity(N_range=None, save_path: str | None = None):
    """
    Sensitivity vs atom number N — SQL vs Heisenberg scaling.
    """
    from ..quantum.gyroscope import QuantumGyroscope, GyroscopeParams

    if N_range is None:
        N_range = np.logspace(1, 5, 100).astype(int)

    sql_sens = []
    heisenberg_sens = []

    for N in N_range:
        sql_g = QuantumGyroscope(GyroscopeParams(N=N, entangled=False))
        heis_g = QuantumGyroscope(GyroscopeParams(N=N, entangled=True))
        sql_sens.append(sql_g.sensitivity)
        heisenberg_sens.append(heis_g.sensitivity)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.loglog(N_range, sql_sens,       label="Standard Quantum Limit (1/√N)", color="#e74c3c", linewidth=2)
    ax.loglog(N_range, heisenberg_sens, label="Heisenberg Limit (1/N)",         color="#2ecc71", linewidth=2)
    ax.set_xlabel("Number of Atoms N")
    ax.set_ylabel("Gyroscope Sensitivity [rad/s]")
    ax.set_title("Quantum Sensing: SQL vs Heisenberg Limit")
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_kalman_comparison(result: ComparisonResult, save_path: str | None = None):
    """
    Three-way comparison: pure INS / KF+classical / KF+quantum.
    Top panel: position errors. Bottom panel: KF covariance bounds.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(
        "QENS Layer 2 - Kalman Filter + Pulsar Navigation\n"
        f"Pulsar fix accuracy: {result.pulsar_accuracy:.0f} m  |  "
        f"Total fixes received: {result.n_fixes}",
        fontsize=12, fontweight='bold'
    )

    t_min = result.times / 60

    ax1.semilogy(t_min, result.pure_classical_errors,
                 color="#e74c3c", linewidth=1.5, label="Pure classical INS (no correction)")
    ax1.semilogy(t_min, result.kf_classical_errors,
                 color="#e67e22", linewidth=1.8, label="KF + classical INS + pulsar fixes")
    ax1.semilogy(t_min, result.kf_quantum_errors,
                 color="#2ecc71", linewidth=2.0, label="KF + quantum INS + pulsar fixes")
    ax1.axhline(result.pulsar_accuracy, color='gray', linestyle='--', alpha=0.6,
                label=f"Pulsar accuracy ({result.pulsar_accuracy:.0f} m)")
    ax1.set_ylabel("Position Error [m]")
    ax1.set_title("Navigation Error")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(t_min, result.kf_classical_sigma,
                 color="#e67e22", linewidth=1.8, linestyle='--',
                 label="KF classical 1-sigma")
    ax2.semilogy(t_min, result.kf_quantum_sigma,
                 color="#2ecc71", linewidth=2.0, linestyle='--',
                 label="KF quantum 1-sigma")
    ax2.set_xlabel("Time [min]")
    ax2.set_ylabel("Position Uncertainty [m]")
    ax2.set_title("Kalman Filter Predicted Uncertainty")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
