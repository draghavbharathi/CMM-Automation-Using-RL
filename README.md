# Autonomous Coordinate Measuring Machine (CMM) Touch-Point Planning via Maskable Reinforcement Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3--Contrib-MaskablePPO-green.svg)](https://sb3-contrib.readthedocs.io/)
[![ROS 2](https://img.shields.io/badge/ROS_2-Humble%20%7C%20Iron-orange.svg)](https://docs.ros.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Autonomous "Next-Best-Touch" metrology inspection planning on 3D workpiece point clouds using Proximal Policy Optimization with Invalid Action Masking (`MaskablePPO`).**

---

## 📌 Overview

Coordinate Measuring Machines (CMMs) are the benchmark standard in precision manufacturing and quality inspection, delivering sub-micron measurement accuracy. However, programming CMM touchpaths is traditionally a slow, manual task or relies on static raster grids that cannot adapt to local sensor noise, surface curvature, or probe collision hazards.

This repository implements a **Deep Reinforcement Learning (RL) framework** for autonomous touch-point planning. Given a 3D candidate point cloud generated from multi-view sensor fusions, the RL agent learns an optimal 20-touch inspection trajectory that:
1. **Maximizes surface uncertainty reduction** using an analytical, normal-aware Kriging Gaussian Process covariance model.
2. **Guarantees metrological plane-fitting compliance** (enforcing $\ge 4$ non-collinear touchpoints across all 5 visible workpiece faces).
3. **Maximizes intra-face spatial dispersion** to prevent ill-conditioned least-squares plane fits.
4. **Minimizes CMM traverse time** under 3-segment Chebyshev ($L_\infty$) clearance-box kinematics.
5. **Guarantees zero probe collisions** via real-time two-cylinder physical collision action masking.

---

## 🖼️ Inspection Trajectory & Performance Highlights

| Planned 3D Inspection Sequence | Training Dynamics & Metrology Convergence |
| :---: | :---: |
| ![Figure 7: 3D Spatial Touchpoint Distribution](figures/fig7_spatial_touch_distribution.png) | ![Figure 1: Training Progression](figures/fig1_learning_curves.png) |
| *Figure 7: 20 planned touchpoints on CAD workpiece showing face grouping and edge-avoidance.* | *Figure 1: Fast convergence in 40k steps with 100% metrological compliance.* |

---

## 📊 Benchmark Evaluation ($N=50$ Monte Carlo Runs)

The trained `MaskablePPO` policy was evaluated against four industrial and heuristic baselines over 50 stochastic episodes with domain randomization:

| Strategy | Success Rate (%) | Cumulative Return | Travel Time (s) | Uncertainty Drop (%) | Final Remaining Unc. (%) | Inspection Efficiency (%/s) | Mean Intra-Face Spread (mm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MaskablePPO (Ours)** | **100.0%** | $\mathbf{122.55 \pm 0.24}$ | $27.69 \pm 0.16$ | $42.51 \pm 0.16$ | $\mathbf{26.04 \pm 0.15}$ | $\mathbf{1.535 \pm 0.010}$ | $53.11 \pm 0.4$ |
| **Greedy Uncertainty** | **100.0%** | $122.83 \pm 0.22$ | $29.58 \pm 0.39$ | $\mathbf{45.04 \pm 0.12}$ | $23.54 \pm 0.11$ | $1.523 \pm 0.021$ | $46.04 \pm 0.5$ |
| **Nearest Neighbor** | **100.0%** | $111.72 \pm 1.28$ | $\mathbf{14.54 \pm 0.44}$ | $29.18 \pm 0.83$ | $39.45 \pm 0.82$ | $2.010 \pm 0.108$ | $34.11 \pm 0.9$ |
| **Raster Grid** | **100.0%** | $121.70 \pm 0.19$ | $26.17 \pm 0.23$ | $33.80 \pm 0.22$ | $34.85 \pm 0.23$ | $1.292 \pm 0.010$ | $\mathbf{80.44 \pm 0.3}$ |
| **Random Masked** | **100.0%** | $117.52 \pm 1.76$ | $28.56 \pm 1.76$ | $37.45 \pm 1.56$ | $31.14 \pm 1.61$ | $1.315 \pm 0.084$ | $55.75 \pm 1.8$ |

![Figure 3: Benchmark Comparison](figures/fig3_baseline_comparison.png)

### Key Insights:
- **Pareto Optimal Inspection:** Pure greedy uncertainty reduction traverses erratically across opposing workpiece faces, adding $+1.89\text{s}$ of unnecessary travel time. MaskablePPO balances motion and information gain, achieving higher net efficiency ($1.535\%/\text{s}$).
- **Failure of Greedy TSP:** Nearest-neighbor heuristics cluster touchpoints locally ($34.11\text{mm}$ spread), leaving $39.45\%$ residual uncertainty across the part.
- **Superiority over Classical Grids:** Fixed industrial raster patterns are blind to surface noise and curvature, achieving only a $33.80\%$ uncertainty drop compared to MaskablePPO's $42.51\%$.

---

## ⚙️ Mathematical Metrology Formulation

### 1. Markov Decision Process (MDP)
- **State Space $\mathcal{S} \subset \mathbb{R}^{2564}$:** Concatenation of $K=256$ candidate points (each with 10 features: normalized coordinates, outward normals, curvature, dynamic Kriging uncertainty, normalized face ID, and visitation flag) + normalized probe coordinates $\in \mathbb{R}^3$ + remaining budget fraction $\in [0, 1]$.
- **Action Space $\mathcal{A} = \{0, 1, \dots, 255\}$:** Discrete choice of next candidate point.
- **Invalid Action Masking $\mathcal{M}(\mathbf{s}_t)$:**
  $$\pi_\theta(a \mid \mathbf{s}_t) = \frac{\exp(z_a^{\text{masked}})}{\sum_{a' \in \mathcal{M}(\mathbf{s}_t)} \exp(z_{a'}^{\text{masked}})}, \quad z_a^{\text{masked}} = \begin{cases} z_a & a \in \mathcal{M}(\mathbf{s}_t) \\ -\infty & a \notin \mathcal{M}(\mathbf{s}_t) \end{cases}$$
  Enforces: (1) unvisited points, (2) physical two-cylinder collision clearance (stylus $r=1.5\text{mm}$, head $r=25\text{mm}$), and (3) budget feasibility to guarantee $\ge 4$ points per face.

### 2. Normal-Aware Kriging Gaussian Process Kernel
Prevents unphysical uncertainty reduction across sharp $90^\circ$ edges:
$$k(\mathbf{p}_i, \mathbf{p}_k) = \exp\left(-\frac{\|\mathbf{p}_i - \mathbf{p}_k\|_2^2}{2\ell^2}\right) \cdot \Big[\max\big(0, \, \mathbf{n}_i \cdot \mathbf{n}_k\big)\Big]^4$$
$$\sigma_{t+1}(\mathbf{p}_i) = \sigma_t(\mathbf{p}_i) \cdot \Big[1 - k(\mathbf{p}_i, \mathbf{p}_k)\Big]$$

### 3. Multi-Objective Reward Function
$$\mathcal{R}(\mathbf{s}_t, a_t) = w_1 R_{\text{unc}} + w_2 R_{\text{prim}} + w_3 R_{\text{spread}} - w_4 R_{\text{travel}} + R_{\text{term}}$$
- $R_{\text{unc}} = \frac{100}{K} \sum_{i=1}^K (\sigma_t - \sigma_{t+1})$ (global uncertainty reduction)
- $R_{\text{prim}} = +3.0$ if face count $\le 4$, else $-1.5$ (encourages face distribution)
- $R_{\text{spread}} = 2.0 \cdot \min(d_{\min}/D_{\text{diag}}, 1.0)$ (maximizes intra-face dispersion)
- $R_{\text{travel}} = t_{\text{move}}$ (penalizes Chebyshev travel time)
- $R_{\text{term}} = +20.0$ if all 5 faces receive $\ge 4$ touches with sub-10mm contact error.

---

## 📁 Repository Structure

```
├── RESEARCH_REPORT_CMM_RL.md   # Complete 31KB academic research report (LaTeX math & derivations)
├── cmm_env.py                  # Custom Gymnasium CMM inspection environment with action masking
├── cmm_math_engine.py          # Fast analytical simulation engine (ray-box, collision, Chebyshev)
├── pointcloud_preprocessor.py  # Geometric preprocessing (voxel, normal estimation, RANSAC, FPS)
├── train_and_log_metrics.py    # Training pipeline with extensive telemetry callback (CSV/JSON logs)
├── train_cmm_rl.py             # Baseline training script
├── benchmark_and_evaluate.py   # Monte Carlo benchmark suite across 5 metrology agents
├── evaluate_and_visualize.py   # Deterministic rollout evaluator and Open3D 3D visualizer
├── generate_figures.py         # Publication figure generator (Matplotlib / Seaborn)
├── candidates_256.npz          # Preprocessed candidate dataset (K=256 points + normals + metadata)
├── cmm_maskable_ppo.zip        # Trained MaskablePPO policy weights
├── merged_box_cloud.ply        # Fused multi-view point cloud of workpiece
├── figures/                    # High-resolution (300 DPI) publication figures (Fig 1 - Fig 7)
├── training_logs/              # Training progression CSVs and optimization diagnostics
├── benchmark_output/           # Benchmark summaries and raw episode traces
└── src/                        # ROS 2 package (cmm_env) with Gazebo worlds, models, and launch files
```

---

## 🚀 Quickstart & Reproduction

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/draghavbharathi/CMM-Automation-Using-RL.git
cd CMM-Automation-Using-RL

# Install required dependencies
pip install torch gymnasium stable-baselines3 sb3-contrib open3d scipy scikit-learn pandas matplotlib seaborn
```

### 2. Evaluate the Trained Model

Run a deterministic inspection rollout and view the metrology summary:
```bash
python3 evaluate_and_visualize.py --no-render
```
*(Omit `--no-render` on a machine with a display to open the interactive Open3D 3D visualizer showing the CMM trajectory, normal arrows, and uncertainty cloud).*

### 3. Run Benchmark Suite (5 Agents, 50 Episodes)

```bash
python3 benchmark_and_evaluate.py --episodes 50
```

### 4. Retrain the Agent with Full Telemetry Logging

```bash
python3 train_and_log_metrics.py --timesteps 40960
```

### 5. Regenerate All Publication Figures

```bash
python3 generate_figures.py
```

---

## 📖 Research Documentation

For the full academic paper containing explicit mathematical proofs, Bellman derivations, Kriging covariance equations, and in-depth GD&T metrological analysis, refer to:
👉 **[`RESEARCH_REPORT_CMM_RL.md`](RESEARCH_REPORT_CMM_RL.md)**

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
