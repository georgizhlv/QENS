"""Tests for spacecraft dynamics and navigator comparison."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from qens.trajectory.spacecraft import Spacecraft, OrbitalParams
from qens.trajectory.navigator import Navigator
from qens.classical.ins import InertialNavigationSystem, INSParams


class TestSpacecraft:
    def test_circular_orbit_conserves_energy(self):
        sc = Spacecraft()
        r0 = sc.r.copy()
        times, pos, vel, _ = sc.run(duration=5400.0, dt=10.0)  # ~one LEO orbit
        # Orbital radius should stay approximately constant (< 1% drift)
        radii = np.linalg.norm(pos, axis=1)
        r_ref = np.linalg.norm(r0)
        assert np.max(np.abs(radii - r_ref) / r_ref) < 0.01

    def test_step_advances_time(self):
        sc = Spacecraft()
        sc.step(dt=10.0)
        assert abs(sc.t - 10.0) < 1e-10


class TestINS:
    def test_error_grows_over_time(self):
        rng = np.random.default_rng(42)
        ins = InertialNavigationSystem(rng=rng)
        ins.reset()
        sc = Spacecraft()

        for _ in range(100):
            _, _, a = sc.step(1.0)
            ins.step(a, np.zeros(3), 1.0)

        pos_error_100s = np.linalg.norm(ins.state.position - sc.r)
        ins.reset()
        sc2 = Spacecraft()
        for _ in range(10):
            _, _, a = sc2.step(1.0)
            ins.step(a, np.zeros(3), 1.0)

        pos_error_10s = np.linalg.norm(ins.state.position - sc2.r)
        assert pos_error_100s > pos_error_10s


class TestNavigator:
    def test_quantum_beats_classical(self):
        nav = Navigator(seed=0)
        result = nav.run(duration=300.0, dt=1.0)
        # Quantum should be significantly better after 5 minutes
        assert result.quantum_errors[-1] < result.classical_errors[-1]

    def test_both_start_near_zero(self):
        nav = Navigator(seed=0)
        result = nav.run(duration=60.0, dt=1.0)
        # First step has small Euler integration error vs RK4 ground truth (~few m)
        assert result.classical_errors[0] < 20.0
        assert result.quantum_errors[0] < 20.0
