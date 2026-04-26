"""
X-ray Pulsar Navigation (XNAV) — position fix source.

Pulsars are rapidly rotating neutron stars that emit X-ray pulses with
extraordinary regularity (rivalling atomic clocks). By measuring the arrival
time of pulses from multiple pulsars and comparing to known timing models,
a spacecraft can determine its position in the solar system.

Current state of the art (NASA NICER, 2018):
    Position accuracy: ~5 km over solar-system distances
    Fix rate: every few minutes (limited by integration time per pulsar)

This module simulates a XNAV system providing periodic position fixes with
Gaussian noise — the "GPS of deep space" that the Kalman filter fuses with INS.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class XRayPulsar:
    """A single millisecond pulsar used for navigation."""
    name: str
    position_accuracy: float = 5000.0   # [m] — current NICER-class accuracy
    fix_interval: float = 120.0         # [s] — time between usable fixes


class PulsarNavSystem:
    """
    Simulates an X-ray pulsar navigation system providing periodic position fixes.

    Uses multiple pulsars to triangulate. Combined fix accuracy improves
    as ~1/√N_pulsars relative to a single pulsar.
    """

    # Near-future XNAV (2030s projection): improved detectors + longer dwell time.
    # NASA NICER (2018) achieved ~5 km; next-generation systems target ~100–500 m.
    KNOWN_PULSARS = [
        XRayPulsar("PSR B1937+21", 200.0, 120.0),
        XRayPulsar("PSR J0437-4715", 150.0, 90.0),
        XRayPulsar("PSR B1855+09", 250.0, 150.0),
    ]

    def __init__(
        self,
        pulsars: list[XRayPulsar] | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.pulsars = pulsars or self.KNOWN_PULSARS
        self.rng = rng or np.random.default_rng()

        # Combined accuracy improves with more pulsars
        individual_var = sum(p.position_accuracy**2 for p in self.pulsars)
        self.combined_accuracy = np.sqrt(individual_var / len(self.pulsars)**2)

        # Fix interval: limited by the slowest pulsar
        self.fix_interval = max(p.fix_interval for p in self.pulsars)

    def get_fix(self, true_position: np.ndarray) -> np.ndarray:
        """
        Return a noisy position fix from pulsar timing.

        Parameters
        ----------
        true_position : true spacecraft position [m]

        Returns
        -------
        Measured position [m] with Gaussian noise ~ N(0, combined_accuracy)
        """
        noise = self.rng.normal(0.0, self.combined_accuracy, 3)
        return true_position + noise

    def should_fix(self, t: float, dt: float) -> bool:
        """Returns True on timesteps where a new pulsar fix is available."""
        prev_t = t - dt
        return int(t / self.fix_interval) > int(prev_t / self.fix_interval)

    def __repr__(self) -> str:
        return (f"PulsarNavSystem({len(self.pulsars)} pulsars, "
                f"accuracy={self.combined_accuracy:.0f} m, "
                f"fix_interval={self.fix_interval:.0f} s)")
