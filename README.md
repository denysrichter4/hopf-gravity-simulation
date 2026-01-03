# Hopf Fibration & Emergent Gravity Simulator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXX.svg)](https://doi.org/10.5281/zenodo.18136027)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)


**An interactive N-Body simulation demonstrating Emergent Gravity and Relativistic Time Dilation arising from the geometry of Hopf Fibrations.**

This software accompanies the paper: *"Emergent Dynamics in Hopf Bundles: Unification of Gravity and Electromagnetism via Geometric Constraints in $S^3$"* by Denys A. R. Alves.

## 🔬 Overview

This project challenges the axiomatic status of Fundamental Forces. Instead of programming Gravity or Time Dilation as fundamental laws, this engine simulates **pure inertial motion on a 4D Hypersphere ($S^3$)**.

By projecting this 4D motion into 3D space using the **Hopf Map**, we observe that:
1.  **Gravity** emerges as a topological interference pattern between fibers ($1/r^2$ law).
2.  **Relativity (Time Dilation)** emerges as a "Computational Lag" or information propagation limit within the manifold.

## ✨ Key Features

* **Real-Time Physics Engine:** Solves 4D geodesic equations with topological coupling on the fly.
* **Hopf Visualization:** Renders the stereographic projection of the 3-sphere fibers.
* **Dual Graph System:**
    * *Left:* 3D Spatial Trajectories.
    * *Right:* Proper Time ($\tau$) vs. System Time ($t$).
* **Interactive Laboratory:** Modify constants of the universe in real-time to falsify or validate the theory.

## 🚀 Installation & Usage

### Prerequisites
* Python 3.8 or higher
* pip

### Setup
```bash
# 1. Clone the repository
git clone [https://github.com/YOUR_USERNAME/hopf-gravity-simulation.git](https://github.com/YOUR_USERNAME/hopf-gravity-simulation.git)
cd hopf-gravity-simulation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the simulation
python simulation.py

```

## 🎛️ Interactive Controls

The simulation provides a control panel to manipulate the fundamental constants of the toy universe:

| Parameter | Type | Description | Effect on Simulation |
| --- | --- | --- | --- |
| **SPEED** | Input Energy | Scales the 4D velocity vectors. | Expands/Shrinks the orbital radii (Universe Expansion). |
| **GRAVITY** | Coupling () | Intensity of fiber interference. | **0.0:** Pure Hopf Circles (Inertia).<br>

<br>**>0.0:** Newtonian Orbits & Chaos. |
| **LIMIT C** | Hardware Limit | Max information propagation speed. | **Lowering C:** Causes fast particles to "freeze" in time (Horizontal line on the Time Graph). |

## 📐 Mathematical Basis

The engine does not use . It integrates the following unified dynamic equation:

$$ \mathbf{a}_{i} = - (\mathbf{v}_i \cdot \mathbf{v}_i) \mathbf{q}*i + \lambda \sum*{j \neq i} \frac{\mathbf{q}_j - \mathbf{q}_i}{||\Pi(\mathbf{q}_j) - \Pi(\mathbf{q}_i)||^2 + \epsilon} $$

Where  is the Hopf Map. Time dilation is calculated as a norm conservation:

$$ \frac{d\tau}{dt} = \sqrt{1 - \left( \frac{||\mathbf{v}*{4D}||}{C*{LIM}} \right)^2} $$

## 🤝 Contributing

This is an open-source research tool. Pull requests for optimization (e.g., Symplectic Integrators, GPU acceleration via CUDA) are welcome.

## 📄 Citation

If you use this code in your research, please cite the associated paper:

```bibtex
@article{alves2026hopf,
  title={Emergent Dynamics in Hopf Bundles},
  author={Alves, Denys A. R.},
  journal={Zenodo},
  year={2026},
  doi={10.5281/zenodo.XXXXXX}
}

```

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.
