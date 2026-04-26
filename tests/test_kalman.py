"""Tests for Kalman filter and pulsar navigation."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qens.trajectory.kalman import ErrorStateKalmanFilter
from qens.sources.pulsar import PulsarNavSystem, XRayPulsar
from qens.trajectory.comparison import NavigationComparison


class TestKalmanFilter:
    def test_covariance_grows_without_updates(self):
        kf = ErrorStateKalmanFilter(accel_noise=1e-4, meas_noise=5000.0)
        sigma_0 = kf.position_uncertainty
        for _ in range(60):
            kf.predict(dt=1.0)
        assert kf.position_uncertainty > sigma_0

    def test_update_reduces_uncertainty(self):
        kf = ErrorStateKalmanFilter(accel_noise=1e-4, meas_noise=5000.0)
        for _ in range(120):
            kf.predict(dt=1.0)
        sigma_before = kf.position_uncertainty
        kf.update(np.zeros(3), np.zeros(3))
        assert kf.position_uncertainty < sigma_before

    def test_correction_moves_toward_measurement(self):
        kf = ErrorStateKalmanFilter(accel_noise=1e-4, meas_noise=100.0)
        for _ in range(60):
            kf.predict(dt=1.0)
        # True position is 500 m away from INS estimate
        true_pos = np.array([500.0, 0.0, 0.0])
        ins_pos  = np.zeros(3)
        corr = kf.update(true_pos, ins_pos)
        # Correction should be positive (towards true_pos)
        assert corr[0] > 0

    def test_quantum_kf_has_lower_uncertainty_than_classical(self):
        kf_cls = ErrorStateKalmanFilter(accel_noise=1e-4, meas_noise=5000.0)
        kf_qnt = ErrorStateKalmanFilter(accel_noise=1e-9, meas_noise=5000.0)
        for _ in range(120):
            kf_cls.predict(dt=1.0)
            kf_qnt.predict(dt=1.0)
        assert kf_qnt.position_uncertainty < kf_cls.position_uncertainty


class TestPulsarNav:
    def test_fix_is_near_true_position(self):
        rng = np.random.default_rng(0)
        pulsar = PulsarNavSystem(rng=rng)
        true_pos = np.array([1e9, 2e9, 3e9])
        fixes = [pulsar.get_fix(true_pos) for _ in range(1000)]
        mean_fix = np.mean(fixes, axis=0)
        assert np.linalg.norm(mean_fix - true_pos) < pulsar.combined_accuracy

    def test_should_fix_triggers_at_interval(self):
        pulsar = PulsarNavSystem()
        dt = 1.0
        fix_times = [t for t in range(1, 601)
                     if pulsar.should_fix(float(t), dt)]
        expected = int(600 / pulsar.fix_interval)
        assert abs(len(fix_times) - expected) <= 1


class TestNavigationComparison:
    def test_quantum_kf_beats_classical_kf(self):
        comp = NavigationComparison(N_atoms=1000, seed=7)
        result = comp.run(duration=600.0, dt=1.0)
        assert result.kf_quantum_errors[-1] < result.kf_classical_errors[-1]

    def test_run_returns_correct_structure(self):
        comp = NavigationComparison(N_atoms=1000, seed=7)
        result = comp.run(duration=300.0, dt=1.0)
        assert len(result.times) == 300
        assert result.kf_quantum_errors.shape == (300,)
        assert result.n_fixes >= 0

    def test_kf_uncertainty_bounded_after_fixes(self):
        # KF sigma should converge when INS drift is much larger than pulsar accuracy.
        # Verify by testing the KF directly with large process noise.
        kf = ErrorStateKalmanFilter(accel_noise=1.0, meas_noise=5000.0)
        pulsar_R = 5000.0
        for step in range(3600):
            kf.predict(dt=1.0)
            if step % 120 == 119:  # fix every 120 steps
                kf.update(np.zeros(3), np.zeros(3))
        # After fixes, KF uncertainty should be near pulsar accuracy, not unbounded
        assert kf.position_uncertainty < pulsar_R * 5
