# Emergent Coulomb Potentials and Orbital Quantization via Classical Dynamics on $S^3$-Hopf Fibrations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.18136027)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)

**Author:** Denys Arthur Richter Alves  
**Date:** January 2026  
**Status:** Proof of Concept / Academic Preprint

![Project Status](https://img.shields.io/badge/Physics-Unified-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🌌 Abstract

The reconciliation of General Relativity (geometric determinism) and Quantum Mechanics (probabilistic indeterminism) remains the central challenge of modern physics. This project presents a computational framework where fundamental quantum phenomena emerge from strictly deterministic dynamics in a 4-dimensional hyperspherical manifold ($S^3$).

We demonstrate that a classical harmonic oscillator confined to $S^3$ geometry and projected onto $\mathbb{R}^3$ via the Hopf Fibration naturally reproduces:
1.  **The Coulomb Potential ($1/r$):** Derived purely from 4D geometric constraints.
2.  **Atomic Stability:** Numerical validation reveals a Virial Ratio of $\approx 0.500$, consistent with hydrogen-like systems.
3.  **Orbital Quantization:** The emergence of discrete topological structures indistinguishable from Schrödinger’s probability orbitals.

---

## 📂 Repository Structure

This repository contains the source code used to validate the theory presented in the paper.

| File | Description |
| :--- | :--- |
| `main_simulation.py` | **The Core Simulation.** Generates the 4D trajectory and visualizes the emergent "electron cloud" and topological twists. (Generates Figure 1) |
| `validation_metrics.py` | **The Mathematical Proof.** Runs the physics engine in "Metrology Mode" to calculate the Virial Ratio ($\langle T \rangle / \langle V \rangle$). Validates the discovery parameters ($K=12.98$, $Spin=5.0$). |
| `visual_comparison.py` | **The "Turing Test" for Orbitals.** Runs the simulation side-by-side with an analytical Schrödinger wavefunction solver to visually compare the geometric shell vs. quantum probability density. |

*(Note: These files correspond to the finalized scripts `4d_earth_simulation_nobel_candidate.py`, `4d_earth_simulation_quantic_proof_2.py`, etc., renamed for clarity)*

---

## 🚀 Key Results

### 1. The "Virial 0.5" Discovery
By strictly following a 4D Harmonic Oscillator model ($F = -kx$), the projected 3D system exhibits the statistical behavior of a Coulomb system ($F \propto 1/r^2$).

```text
--- PRECISION MEASUREMENTS ---
Parameters: K=12.98, Spin=5.0
1. Average Total Energy: 9.739968
2. Virial Ratio (<T>/<V>): 0.500765  <-- MATCHES HYDROGEN (0.5)

```

### 2. Emergent Orbitals

The simulation (Cyan) generates topological shells that match the isosurfaces of the Schrödinger equation (Green).

| **Geometric Simulation (Ours)** | **Quantum Reality (Schrödinger)** |
| --- | --- |
|  |  |
| *(Replace these placeholders with your actual screenshots in the /results folder)* |  |

---

## 🛠️ Installation & Usage

### Prerequisites

You need Python 3.8+ and standard scientific libraries.

```bash
pip install numpy matplotlib scipy

```

### Running the Proofs

**1. To verify the Virial Ratio (Mathematical Proof):**

```bash
python validation_metrics.py

```

**2. To visualize the Quantum Comparison:**

```bash
python visual_comparison.py

```

**3. To run the full 4D simulation:**

```bash
python main_simulation.py

```

---

## 🧠 Theoretical Background

### The Hamiltonian

The particle moves in  under a Lagrangian:
$$ \mathcal{L} = \frac{1}{2}m (\dot{q} \cdot \dot{q}) - \frac{1}{2} K (q \cdot q) $$
Subject to the constraint  (it lives on the 3-sphere).

### The Projection

We observe the shadow of this particle via the Hopf Map :
$$ \vec{r} = (2(q_1 q_3 + q_2 q_4), 2(q_2 q_3 - q_1 q_4), q_1^2 + q_2^2 - q_3^2 - q_4^2) $$

This non-linear projection transforms the harmonic potential into a gravitational/electrostatic one, suggesting that **Gravity is the shadow of 4D Elasticity.**

---

## 📄 Citation

If you use this code or concepts in your research, please cite:

> Richter, D. (2026). *Emergent Coulomb Potentials and Orbital Quantization via Classical Dynamics on -Hopf Fibrations*. GitHub Repository.

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
