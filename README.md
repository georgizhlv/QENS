# QENS — Quantum Entanglement Navigation Simulator

An open-source educational simulator for quantum-enhanced spacecraft navigation.
Models how entangled quantum sensors outperform classical inertial navigation
in deep space, where GPS is unavailable and sensor drift is the limiting factor.

This is an active research area (NASA, ESA, DLR) with no existing public simulation
at an accessible level. QENS fills that gap.

---

## The Problem

Classical inertial navigation systems (INS) accumulate position error over time:

- Accelerometer noise integrates twice → position error grows as **t²**
- In deep space (no GPS), a 10 cm/s² sensor bias produces **~65 km error after 1 hour**
- X-ray pulsar fixes (XNAV) can correct this, but require accurate INS between fixes

Quantum sensors using entangled atom pairs suppress noise below the Standard Quantum
Limit (SQL), approaching the **Heisenberg limit**: sensitivity scales as **1/N** instead
of **1/√N** for N atoms. This is a 31x improvement for N=1000 atoms.

---

## What QENS Simulates

### Layer 1 — Quantum Physics (Qiskit)

| Component | Physics | File |
|-----------|---------|------|
| Bell state generation | Φ+, Φ−, Ψ+, Ψ− via Hadamard + CNOT | `quantum/entanglement.py` |
| Decoherence model | Depolarizing noise, exp(−t/T₂) decay | `quantum/noise.py` |
| Quantum gyroscope | Matter-wave Sagnac effect: Δφ = 4π·m·A·Ω/h | `quantum/gyroscope.py` |
| Quantum accelerometer | Atom interferometry: Δφ = k_eff·a·T² | `quantum/accelerometer.py` |

### Layer 2 — Navigation

| Component | Description | File |
|-----------|-------------|------|
| Classical INS | MEMS IMU with white noise + bias drift | `classical/ins.py` |
| Spacecraft | Two-body Keplerian dynamics, RK4 integrator | `trajectory/spacecraft.py` |
| Navigator | Classical vs quantum INS comparison | `trajectory/navigator.py` |
| Kalman filter | 6-state error-state KF for sensor fusion | `trajectory/kalman.py` |
| Pulsar nav (XNAV) | X-ray pulsar timing position fixes | `sources/pulsar.py` |
| 3-way comparison | Pure INS / KF+classical / KF+quantum | `trajectory/comparison.py` |

---

## Results

### Sensor Sensitivity (N = 1000 atoms, Rb-87)

| Regime | Sensitivity | Formula |
|--------|-------------|---------|
| Standard Quantum Limit (SQL) | 1.16 × 10⁻⁷ rad/s | 1/√N |
| Heisenberg Limit (entangled) | 3.66 × 10⁻⁹ rad/s | 1/N |
| **Improvement** | **31.6×** | **√N** |

### Navigation (LEO orbit, 10 minutes)

| Navigator | Final position error |
|-----------|---------------------|
| Classical INS | 5.38 m |
| Quantum INS | 0.043 m |
| **Quantum advantage** | **124×** |

### Navigation + Kalman Filter + XNAV (1 hour, tactical-grade classical sensor)

| Navigator | Final position error |
|-----------|---------------------|
| Pure classical INS | ~41 km |
| KF + classical INS + pulsar fixes | ~123 m |
| KF + quantum INS + pulsar fixes | ~0.015 m |
| **KF quantum vs KF classical** | **~8000×** |

The Kalman filter fuses INS predictions with periodic X-ray pulsar position fixes.
Quantum INS has such low process noise that it barely needs external corrections.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/QENS.git
cd QENS
pip install -r requirements.txt
```

**Requirements:** Python 3.11+, qiskit ≥ 2.4, qiskit-aer ≥ 0.17, numpy, scipy, matplotlib

---

## Usage

```bash
# Run all four demonstrations with plots
python examples/run_full_simulation.py

# Run tests
python -m pytest tests/ -v
```

### Quick start in Python

```python
from qens.quantum.entanglement import EntangledPair, BellState
from qens.trajectory.comparison import NavigationComparison

# Generate a Bell state and measure entanglement correlation
pair = EntangledPair(BellState.PHI_PLUS, shots=4096)
print(f"<Z⊗Z> correlation: {pair.correlation():.4f}")  # → ~1.0000
print(f"Fidelity: {pair.entanglement_fidelity():.4f}")  # → ~1.0000

# Run 3-way navigation comparison (1 hour)
comp = NavigationComparison(N_atoms=1000, seed=42)
result = comp.run(duration=3600.0, dt=1.0)
print(f"Quantum advantage: {result.kf_classical_errors[-1] / result.kf_quantum_errors[-1]:.0f}x")
```

---

## Physics Background

### Why quantum sensors are better

Classical accelerometers measure forces mechanically. Noise floor is set by thermal
fluctuations and electronics — the **Standard Quantum Limit** σ ∝ 1/√N.

Quantum atom interferometers split a cold atom cloud into two paths using laser pulses,
accumulate phase proportional to acceleration, then recombine. With **entangled NOON states**,
the phase sensitivity scales as 1/N — the **Heisenberg limit**.

For N = 1000 atoms: 31.6× improvement. For N = 10,000: 100× improvement.

### Why deep space navigation is hard

GPS satellites broadcast at ~20,200 km altitude. Beyond ~100,000 km GPS is unusable.
The next best option: **X-ray pulsar navigation (XNAV)** — millisecond pulsars emit
X-ray pulses with atomic clock regularity. NASA NICER (2018) demonstrated ~5 km accuracy.
Next-generation systems (2030s) are projected to reach ~100–500 m.

Between pulsar fixes, the spacecraft relies on its INS. Quantum INS reduces the
between-fix drift from kilometers (classical MEMS) to centimeters.

### Kalman filter fusion

The error-state Kalman filter tracks [δposition, δvelocity] — the difference between
the INS solution and the truth. Process noise Q is set by the accelerometer noise.
Measurement noise R is set by the pulsar fix accuracy. When INS drift (P) exceeds
pulsar accuracy (R), the KF trusts the fix; otherwise it ignores it.

This is why quantum INS is transformative: its process noise Q is ~10⁶× smaller,
so the KF only needs to make tiny corrections.

---

## Project Structure

```
QENS/
├── qens/
│   ├── quantum/          # Qiskit quantum sensor simulation
│   │   ├── entanglement.py
│   │   ├── gyroscope.py
│   │   ├── accelerometer.py
│   │   └── noise.py
│   ├── classical/        # Classical MEMS INS
│   │   └── ins.py
│   ├── trajectory/       # Orbital mechanics + navigation
│   │   ├── spacecraft.py
│   │   ├── navigator.py
│   │   ├── kalman.py
│   │   └── comparison.py
│   ├── sources/          # External measurement sources
│   │   └── pulsar.py
│   └── visualization/    # matplotlib plots
│       └── plots.py
├── tests/                # 23 unit tests
├── examples/
│   └── run_full_simulation.py
└── requirements.txt
```

---

## Roadmap

- [x] Layer 1: Quantum entanglement simulation (Qiskit)
- [x] Layer 1: Quantum gyroscope and accelerometer models
- [x] Layer 2: Classical INS with realistic noise model
- [x] Layer 2: Two-body orbital mechanics (RK4)
- [x] Layer 2: Error-state Kalman filter
- [x] Layer 2: X-ray pulsar navigation (XNAV)
- [x] Layer 2: Three-way navigation comparison
- [ ] Layer 3: Raspberry Pi hardware INS module
- [ ] Layer 3: Optical polarization analogue experiment

---

## References

- Degen, C. L., Reinhard, F., & Cappellaro, P. (2017). *Quantum sensing*. Reviews of Modern Physics.
- Gustavson, T. L., et al. (1997). *Precision rotation measurements with an atom interferometry gyroscope*. Physical Review Letters.
- Winternitz, L. M. B., et al. (2016). *XNAV beyond the Moon*. ION GNSS+.
- NASA NICER mission: [https://www.nasa.gov/nicer](https://www.nasa.gov/nicer)

---

## License

MIT License — see [LICENSE](LICENSE) file.
