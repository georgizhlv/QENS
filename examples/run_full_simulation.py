"""
QENS Full Simulation Example
============================

Demonstrations:
  1. Entanglement generation and decoherence (Qiskit)
  2. Sensor sensitivity: SQL vs Heisenberg limit
  3. Basic navigation: classical INS vs quantum INS
  4. [Layer 2] Kalman filter + X-ray pulsar navigation — 3-way comparison

Run with:  python examples/run_full_simulation.py
"""

import sys
import os
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qens.quantum.entanglement import EntangledPair, BellState
from qens.trajectory.navigator import Navigator
from qens.trajectory.comparison import NavigationComparison
from qens.sources.pulsar import PulsarNavSystem
from qens.visualization import (
    plot_navigation_comparison,
    plot_entanglement_decoherence,
    plot_sensor_sensitivity,
    plot_kalman_comparison,
)


def demo_entanglement():
    print("\n=== 1. Entanglement Generation ===")
    pair = EntangledPair(BellState.PHI_PLUS, shots=4096)
    counts = pair.measure()
    corr = pair.correlation()
    fidelity = pair.entanglement_fidelity()
    print(f"  Bell state:  {pair.bell_state}")
    print(f"  Counts:      {counts}")
    print(f"  <Z@Z> corr:  {corr:.4f}  (ideal: +1.000)")
    print(f"  Fidelity:    {fidelity:.4f}  (1.000 = perfect)")
    print("\n  Plotting decoherence curve...")
    plot_entanglement_decoherence(T2=1.0)


def demo_sensors():
    print("\n=== 2. Sensor Sensitivity Scaling ===")
    from qens.quantum.gyroscope import QuantumGyroscope, GyroscopeParams

    for N in [100, 1000, 10000]:
        g_sql  = QuantumGyroscope(GyroscopeParams(N=N, entangled=False))
        g_heis = QuantumGyroscope(GyroscopeParams(N=N, entangled=True))
        print(f"  N={N:6d}: SQL={g_sql.sensitivity:.2e} rad/s | "
              f"Heisenberg={g_heis.sensitivity:.2e} rad/s | "
              f"Improvement={g_heis.sensitivity_ratio_vs_classical():.1f}x")

    plot_sensor_sensitivity()


def demo_navigation():
    print("\n=== 3. Basic Navigation Comparison (LEO, 10 min) ===")
    nav = Navigator(seed=42)
    result = nav.run(duration=600.0, dt=1.0)

    final_cls = result.classical_errors[-1]
    final_qnt = result.quantum_errors[-1]
    print(f"  Classical INS error: {final_cls:.2f} m")
    print(f"  Quantum INS error:   {final_qnt:.4f} m")
    print(f"  Quantum advantage:   {final_cls/final_qnt:.1f}x")

    plot_navigation_comparison(result)


def demo_kalman():
    print("\n=== 4. [Layer 2] Kalman Filter + Pulsar Navigation (1 hour) ===")

    pulsar = PulsarNavSystem()
    print(f"  Pulsar system:  {pulsar}")

    comp = NavigationComparison(N_atoms=1000, seed=42)
    result = comp.run(duration=3600.0, dt=1.0)

    print(f"  Fixes received: {result.n_fixes}")
    print()
    print(f"  Pure classical INS final error:  {result.pure_classical_errors[-1]:>10.1f} m")
    print(f"  KF + classical INS final error:  {result.kf_classical_errors[-1]:>10.1f} m")
    print(f"  KF + quantum INS final error:    {result.kf_quantum_errors[-1]:>10.4f} m")
    print()
    cls_adv = result.kf_classical_errors[-1] / max(result.kf_quantum_errors[-1], 1e-12)
    print(f"  KF quantum vs KF classical:      {cls_adv:.1f}x more accurate")

    plot_kalman_comparison(result)


if __name__ == "__main__":
    print("=" * 56)
    print("  QENS - Quantum Entanglement Navigation Simulator")
    print("=" * 56)

    demo_entanglement()
    demo_sensors()
    demo_navigation()
    demo_kalman()

    print("\nDone.")
