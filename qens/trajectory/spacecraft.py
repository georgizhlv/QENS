"""
Two-body orbital mechanics for a spacecraft around a central body.

Uses Keplerian dynamics: a = −μ·r/|r|³
Simple but sufficient to generate a realistic reference trajectory
against which navigation errors are compared.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class OrbitalParams:
    mu: float = 3.986e14      # gravitational parameter [m³/s²] — Earth default
    r0: np.ndarray = None     # initial position [m]
    v0: np.ndarray = None     # initial velocity [m/s]

    def __post_init__(self):
        if self.r0 is None:
            # Low Earth orbit: 400 km altitude
            self.r0 = np.array([6.771e6, 0.0, 0.0])
        if self.v0 is None:
            # Circular orbit velocity at r0
            v_circ = np.sqrt(self.mu / np.linalg.norm(self.r0))
            self.v0 = np.array([0.0, v_circ, 0.0])


class Spacecraft:
    """
    Propagates a spacecraft trajectory under two-body gravity using RK4.
    Yields ground-truth position, velocity, and acceleration at each step.
    """

    def __init__(self, params: OrbitalParams | None = None):
        self.p = params or OrbitalParams()
        self.r = self.p.r0.astype(float).copy()
        self.v = self.p.v0.astype(float).copy()
        self.t = 0.0

    def _accel(self, r: np.ndarray) -> np.ndarray:
        return -self.p.mu * r / np.linalg.norm(r)**3

    def step(self, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """RK4 integration step. Returns (position, velocity, acceleration)."""
        r, v = self.r, self.v

        def drdt(r, v): return v
        def dvdt(r, v): return self._accel(r)

        k1r = drdt(r, v)
        k1v = dvdt(r, v)
        k2r = drdt(r + 0.5*dt*k1r, v + 0.5*dt*k1v)
        k2v = dvdt(r + 0.5*dt*k1r, v + 0.5*dt*k1v)
        k3r = drdt(r + 0.5*dt*k2r, v + 0.5*dt*k2v)
        k3v = dvdt(r + 0.5*dt*k2r, v + 0.5*dt*k2v)
        k4r = drdt(r + dt*k3r, v + dt*k3v)
        k4v = dvdt(r + dt*k3r, v + dt*k3v)

        self.r = r + (dt/6) * (k1r + 2*k2r + 2*k3r + k4r)
        self.v = v + (dt/6) * (k1v + 2*k2v + 2*k3v + k4v)
        self.t += dt

        return self.r.copy(), self.v.copy(), self._accel(self.r)

    def run(self, duration: float, dt: float):
        """
        Propagate for `duration` seconds with timestep `dt`.
        Returns arrays: times, positions, velocities, accelerations.
        """
        steps = int(duration / dt)
        times = np.zeros(steps)
        positions = np.zeros((steps, 3))
        velocities = np.zeros((steps, 3))
        accels = np.zeros((steps, 3))

        for i in range(steps):
            r, v, a = self.step(dt)
            times[i] = self.t
            positions[i] = r
            velocities[i] = v
            accels[i] = a

        return times, positions, velocities, accels
