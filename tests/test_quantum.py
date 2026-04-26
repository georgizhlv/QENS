"""Basic tests for quantum sensor modules."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qens.quantum.entanglement import EntangledPair, BellState
from qens.quantum.gyroscope import QuantumGyroscope, GyroscopeParams
from qens.quantum.accelerometer import QuantumAccelerometer, AccelerometerParams


class TestEntangledPair:
    def test_phi_plus_correlation_near_one(self):
        pair = EntangledPair(BellState.PHI_PLUS, shots=8192)
        assert abs(pair.correlation() - 1.0) < 0.05

    def test_psi_plus_correlation_near_minus_one(self):
        pair = EntangledPair(BellState.PSI_PLUS, shots=8192)
        assert abs(pair.correlation() + 1.0) < 0.05

    def test_fidelity_near_one(self):
        pair = EntangledPair(BellState.PHI_PLUS)
        assert pair.entanglement_fidelity() > 0.99

    def test_decoherence_reduces_correlation(self):
        pair = EntangledPair(BellState.PHI_PLUS, shots=4096)
        decohered = pair.decohere(t=2.0, T2=1.0)
        assert abs(decohered.correlation()) < abs(pair.correlation())


class TestQuantumGyroscope:
    def test_heisenberg_better_than_sql(self):
        g_sql  = QuantumGyroscope(GyroscopeParams(N=1000, entangled=False))
        g_heis = QuantumGyroscope(GyroscopeParams(N=1000, entangled=True))
        assert g_heis.sensitivity < g_sql.sensitivity

    def test_sensitivity_ratio_equals_sqrt_N(self):
        N = 500
        g = QuantumGyroscope(GyroscopeParams(N=N, entangled=True))
        assert abs(g.sensitivity_ratio_vs_classical() - np.sqrt(N)) < 1e-6

    def test_measure_is_near_truth(self):
        rng = np.random.default_rng(0)
        g = QuantumGyroscope(GyroscopeParams(N=10000, entangled=True), rng=rng)
        true_omega = 7.292e-5  # Earth's rotation rate [rad/s]
        measurements = g.measure_batch(true_omega, 1000)
        assert abs(np.mean(measurements) - true_omega) < 3 * g.sensitivity


class TestQuantumAccelerometer:
    def test_heisenberg_better_than_sql(self):
        a_sql  = QuantumAccelerometer(AccelerometerParams(N=1000, entangled=False))
        a_heis = QuantumAccelerometer(AccelerometerParams(N=1000, entangled=True))
        assert a_heis.sensitivity < a_sql.sensitivity

    def test_measure_unbiased(self):
        rng = np.random.default_rng(1)
        a = QuantumAccelerometer(AccelerometerParams(N=10000, entangled=True), rng=rng)
        true_a = 9.81
        meas = a.measure_batch(true_a, 1000)
        assert abs(np.mean(meas) - true_a) < 3 * a.sensitivity
