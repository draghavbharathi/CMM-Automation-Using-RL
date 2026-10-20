#!/usr/bin/env python3
"""
Custom Gymnasium Environment for Reinforcement Learning CMM Touch-Point Planning.
Compatible with stable-baselines3 and sb3-contrib MaskablePPO.

Implements:
- 10-feature candidate observation space + probe position + budget fraction
- Discrete(K) action space with Invalid Action Masking (action_masks())
- Kriging / Gaussian Process uncertainty update with normal-aware radial kernel
- Multi-objective reward (uncertainty reduction, primitive coverage, spatial spread, travel time)
- Terminal completion bonus for 5-face CMM coverage.
"""

import math
from typing import Tuple, Dict, Any, Optional, List
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from cmm_math_engine import CMMMathEngine
from pointcloud_preprocessor import PreprocessedCandidateData, PointCloudPreprocessor


class CMMTouchPointEnv(gym.Env):
    """
    Adaptive Next-Best-Touch CMM Planning Environment.
    """
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self,
                 candidate_data: Optional[PreprocessedCandidateData] = None,
                 npz_path: Optional[str] = None,
                 ply_path: Optional[str] = None,
                 K: int = 256,
                 max_touches: int = 20,
                 length_scale: float = 0.025,
                 box_dimensions: Tuple[float, float, float] = (0.10, 0.10, 0.10),
                 box_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 probe_mode: str = 'mode_a',
                 axis_speeds: Tuple[float, float, float] = (0.15, 0.15, 0.15),
                 reward_weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 0.5),
                 domain_randomize: bool = False):
        super().__init__()

        self.K = int(K)
        self.max_touches = int(max_touches)
        self.length_scale = float(length_scale)
        self.box_dimensions = box_dimensions
        self.box_center = box_center
        self.probe_mode = probe_mode
        self.axis_speeds = axis_speeds
        self.reward_weights = reward_weights
        self.domain_randomize = domain_randomize

        # 1. Load or Generate Candidate Dataset
        if candidate_data is not None:
            self.candidate_data = candidate_data
        elif npz_path is not None and npz_path != '':
            self.candidate_data = PreprocessedCandidateData.load_npz(npz_path)
        elif ply_path is not None and ply_path != '':
            self.candidate_data = PointCloudPreprocessor.process(ply_path, K=self.K)
        else:
            # Synthetic / analytical demo box generator (fallback if no PLY provided)
            self.candidate_data = self._generate_synthetic_box_candidates(self.K)

        self.coords_base = self.candidate_data.coords.copy()
        self.normals_base = self.candidate_data.normals.copy()
        self.curvatures_base = self.candidate_data.curvatures.copy()
        self.uncertainties_base = self.candidate_data.initial_uncertainties.copy()
        self.primitive_ids = self.candidate_data.primitive_ids.copy()
        self.dense_cloud = self.candidate_data.dense_cloud.copy()
        self.scale = float(self.candidate_data.scale)

        # 2. Mathematical CMM Simulation Engine
        self.math_engine = CMMMathEngine(
            box_dimensions=self.box_dimensions,
            box_center=self.box_center,
            probe_mode=self.probe_mode,
            axis_speeds=self.axis_speeds
        )

        # Precompute static validity collision mask for all candidate points
        self.static_valid_mask = self.math_engine.compute_action_validity_mask(
            self.coords_base,
            self.normals_base,
            self.dense_cloud
        )

        # 3. Action Space: Discrete selection of next candidate point [0, K-1]
        self.action_space = spaces.Discrete(self.K)

        # 4. Observation Space:
        # For each candidate point: [x, y, z, nx, ny, nz, kappa, sigma_t, p_id_norm, visited_flag] -> 10 features
        # Plus current probe position (x, y, z) -> 3 features
        # Plus remaining budget fraction -> 1 feature
        # Total dimension = 10 * K + 4
        self.obs_dim = 10 * self.K + 4
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # Internal episode states
        self.step_count = 0
        self.visited = np.zeros(self.K, dtype=np.float32)
        self.uncertainties = np.zeros(self.K, dtype=np.float32)
        self.coords = np.zeros((self.K, 3), dtype=np.float32)
        self.normals = np.zeros((self.K, 3), dtype=np.float32)
        self.probe_pos = np.zeros(3, dtype=np.float32)
        self.touch_history: List[int] = []
        self.touch_errors: List[float] = []
        self.total_travel_time = 0.0

        # Face touch counters (5 faces: 0 to 4)
        self.face_touch_counts = np.zeros(5, dtype=np.int32)
        # Face touch positions for spread calculation
        self.face_touch_pts: Dict[int, List[np.ndarray]] = {f: [] for f in range(5)}

        self.reset()

    def _generate_synthetic_box_candidates(self, K: int) -> PreprocessedCandidateData:
        """
        Synthesize clean surface points across the 5 visible faces of the demo box
        with realistic Gaussian measurement noise for testing without Gazebo.
        """
        pts_per_face = int(math.ceil(K / 5.0))
        pts_list = []
        normals_list = []
        prim_list = []

        half = 0.05
        # Face 0: Top (+Z)
        x = np.random.uniform(-half, half, pts_per_face)
        y = np.random.uniform(-half, half, pts_per_face)
        z = np.full(pts_per_face, half)
        pts_list.append(np.column_stack([x, y, z]))
        normals_list.append(np.tile([0.0, 0.0, 1.0], (pts_per_face, 1)))
        prim_list.append(np.full(pts_per_face, 0, dtype=np.int64))

        # Face 1: North (+X)
        x = np.full(pts_per_face, half)
        y = np.random.uniform(-half, half, pts_per_face)
        z = np.random.uniform(-half, half, pts_per_face)
        pts_list.append(np.column_stack([x, y, z]))
        normals_list.append(np.tile([1.0, 0.0, 0.0], (pts_per_face, 1)))
        prim_list.append(np.full(pts_per_face, 1, dtype=np.int64))

        # Face 2: South (-X)
        x = np.full(pts_per_face, -half)
        y = np.random.uniform(-half, half, pts_per_face)
        z = np.random.uniform(-half, half, pts_per_face)
        pts_list.append(np.column_stack([x, y, z]))
        normals_list.append(np.tile([-1.0, 0.0, 0.0], (pts_per_face, 1)))
        prim_list.append(np.full(pts_per_face, 2, dtype=np.int64))

        # Face 3: East (+Y)
        x = np.random.uniform(-half, half, pts_per_face)
        y = np.full(pts_per_face, half)
        z = np.random.uniform(-half, half, pts_per_face)
        pts_list.append(np.column_stack([x, y, z]))
        normals_list.append(np.tile([0.0, 1.0, 0.0], (pts_per_face, 1)))
        prim_list.append(np.full(pts_per_face, 3, dtype=np.int64))

        # Face 4: West (-Y)
        x = np.random.uniform(-half, half, pts_per_face)
        y = np.full(pts_per_face, -half)
        z = np.random.uniform(-half, half, pts_per_face)
        pts_list.append(np.column_stack([x, y, z]))
        normals_list.append(np.tile([0.0, -1.0, 0.0], (pts_per_face, 1)))
        prim_list.append(np.full(pts_per_face, 4, dtype=np.int64))

        all_pts = np.vstack(pts_list)[:K]
        all_normals = np.vstack(normals_list)[:K]
        all_prim = np.concatenate(prim_list)[:K]

        # Add simulated sensor noise (sigma = 0.0005m / 0.5mm)
        noise = np.random.normal(0.0, 0.0005, all_pts.shape)
        noisy_pts = all_pts + noise

        # Synthetic curvature: higher at corners / edges
        d_center = np.linalg.norm(noisy_pts, axis=1)
        curv = (d_center - np.min(d_center)) / (np.ptp(d_center) + 1e-8)
        uncertainties = 0.5 + 0.5 * curv

        return PreprocessedCandidateData(
            coords=noisy_pts.astype(np.float32),
            normals=all_normals.astype(np.float32),
            curvatures=curv.astype(np.float32),
            initial_uncertainties=uncertainties.astype(np.float32),
            primitive_ids=all_prim.astype(np.int64),
            centroid=np.zeros(3, dtype=np.float32),
            scale=0.10,
            dense_cloud=noisy_pts.astype(np.float32),
            dense_normals=all_normals.astype(np.float32)
        )

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.step_count = 0
        self.visited = np.zeros(self.K, dtype=np.float32)
        self.uncertainties = self.uncertainties_base.copy()
        self.coords = self.coords_base.copy()
        self.normals = self.normals_base.copy()
        self.touch_history.clear()
        self.touch_errors.clear()
        self.total_travel_time = 0.0

        self.face_touch_counts = np.zeros(5, dtype=np.int32)
        self.face_touch_pts = {f: [] for f in range(5)}

        # Domain randomization during training for policy robustness
        if self.domain_randomize:
            # Jitter coordinates slightly (+- 0.5mm)
            coord_jitter = np.random.normal(0.0, 0.0005, self.coords.shape).astype(np.float32)
            self.coords += coord_jitter
            # Jitter initial uncertainties (+- 0.05)
            unc_jitter = np.random.uniform(-0.05, 0.05, self.uncertainties.shape).astype(np.float32)
            self.uncertainties = np.clip(self.uncertainties + unc_jitter, 0.4, 1.0)

        # Initial probe resting position: clearance height above box top
        z_safe = float(self.coords[:, 2].max() + 0.040)
        self.probe_pos = np.array([0.0, 0.0, z_safe], dtype=np.float32)

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def action_masks(self) -> np.ndarray:
        """
        Invalid Action Masking for sb3-contrib MaskablePPO:
        Action i is valid (True) if and only if:
        1. Point i has NOT been visited yet (m_i == 0), AND
        2. Point i passes two-cylinder collision check and floor clearance, AND
        3. Metrology plane-fitting budget feasibility constraint: if remaining touches
           equal touches still needed to reach 4 per face, restrict to faces needing touches.
        """
        valid = (self.visited == 0.0) & self.static_valid_mask

        # Metrology Constraint Masking for 4-point plane fitting
        remaining_budget = self.max_touches - self.step_count
        needed_per_face = np.maximum(0, 4 - self.face_touch_counts)
        total_needed = int(np.sum(needed_per_face))

        if 0 < total_needed and remaining_budget <= total_needed:
            faces_needing = np.where(needed_per_face > 0)[0]
            face_mask = np.isin(self.primitive_ids, faces_needing)
            restricted = valid & face_mask
            if np.any(restricted):
                valid = restricted

        # If no valid actions remain, avoid all-False crash in MaskablePPO
        if not np.any(valid):
            valid = (self.visited == 0.0)
            if not np.any(valid):
                valid[0] = True
        return valid.astype(bool)

    def _get_observation(self) -> np.ndarray:
        """
        Construct flattened state observation vector:
        1. For each point: [x_norm, y_norm, z_norm, nx, ny, nz, kappa, sigma_t, p_id_norm, visited]
        2. Probe tip position: [xp_norm, yp_norm, zp_norm]
        3. Fraction of touch budget remaining: (T_max - t) / T_max
        """
        coords_norm = self.coords / self.scale
        normals = self.normals
        curvatures = self.curvatures_base[:, None]
        uncertainties = self.uncertainties[:, None]
        prim_ids_norm = (self.primitive_ids[:, None].astype(np.float32)) / 4.0
        visited = self.visited[:, None]

        # Stack candidate features: (K, 10)
        point_features = np.hstack([
            coords_norm,
            normals,
            curvatures,
            uncertainties,
            prim_ids_norm,
            visited
        ]).astype(np.float32)

        probe_pos_norm = (self.probe_pos / self.scale).astype(np.float32)
        budget_fraction = np.array([(self.max_touches - self.step_count) / float(self.max_touches)], dtype=np.float32)

        obs = np.concatenate([point_features.flatten(), probe_pos_norm, budget_fraction])
        return obs

    def _get_info(self) -> Dict[str, Any]:
        return {
            'step_count': self.step_count,
            'mean_uncertainty': float(np.mean(self.uncertainties)),
            'total_travel_time': self.total_travel_time,
            'face_touch_counts': self.face_touch_counts.tolist(),
            'touches': list(self.touch_history),
            'mean_touch_error': float(np.mean(self.touch_errors)) if self.touch_errors else 0.0
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = int(action)

        # Check action validity
        if action < 0 or action >= self.K:
            raise ValueError(f"Invalid action {action} outside range [0, {self.K-1}]")

        if self.visited[action] == 1.0:
            # Harsh penalty for repeating a touch
            obs = self._get_observation()
            return obs, -10.0, False, False, self._get_info()

        # 1. Mark point as visited
        self.visited[action] = 1.0
        self.touch_history.append(action)
        self.step_count += 1

        p_chosen = self.coords[action]
        n_chosen = self.normals[action]
        face_id = int(self.primitive_ids[action])

        # 2. Micron-accurate CAD Ground-Truth Touch
        p_true, deviation_error = self.math_engine.ground_truth_touch(p_chosen, n_chosen)
        self.touch_errors.append(deviation_error)

        # 3. Calculate Chebyshev Travel Time
        t_move = self.math_engine.compute_travel_time(
            p_prev=self.probe_pos,
            p_curr=p_chosen,
            n_curr=n_chosen,
            cloud_max_z=float(self.coords[:, 2].max())
        )
        self.total_travel_time += t_move
        self.probe_pos = p_chosen.copy()

        # 4. Kriging / Gaussian Process Uncertainty Update
        # Radial basis function with normal-awareness:
        # k(i, j) = exp(-||p_i - p_j||^2 / (2*ell^2)) * max(0, n_i . n_j)^4
        # Points around orthogonal corners do not experience uncertainty reduction
        delta_p = self.coords - p_chosen  # (K, 3)
        dist_sq = np.sum(delta_p * delta_p, axis=1)  # (K,)
        normal_dots = np.maximum(0.0, np.sum(self.normals * n_chosen, axis=1))  # (K,)

        spatial_kernel = np.exp(-dist_sq / (2.0 * (self.length_scale ** 2)))
        normal_kernel = normal_dots ** 4
        kernel_weights = spatial_kernel * normal_kernel

        old_uncertainties = self.uncertainties.copy()
        self.uncertainties = self.uncertainties * (1.0 - kernel_weights)
        self.uncertainties = np.clip(self.uncertainties, 0.0, 1.0)

        # 5. Multi-Objective Reward Computation
        w1, w2, w3, w4 = self.reward_weights

        # 5.1 Uncertainty Reduction Reward
        # Total normalized drop across all K candidate points
        r_uncertainty = float(np.mean(old_uncertainties - self.uncertainties) * 100.0)

        # 5.2 Geometric Primitive Coverage Bonus
        self.face_touch_counts[face_id] += 1
        touches_on_this_face = self.face_touch_counts[face_id]
        if touches_on_this_face <= 4:
            r_primitive = 3.0
        else:
            r_primitive = -1.5  # Strongly encourage spreading across all other faces

        # 5.3 Spatial Spread Bonus on the Same Face
        # Reward larger distance from previously touched points on the SAME face
        prev_face_pts = self.face_touch_pts[face_id]
        face_diag = float(math.sqrt(self.box_dimensions[0]**2 + self.box_dimensions[1]**2))
        if len(prev_face_pts) == 0:
            r_spread = 1.0  # First touch on this face
        else:
            dists = [np.linalg.norm(p_chosen - p_prev) for p_prev in prev_face_pts]
            d_min = min(dists)
            r_spread = min(d_min / max(face_diag, 1e-4), 1.0) * 2.0

        self.face_touch_pts[face_id].append(p_chosen.copy())

        # 5.4 Travel Time Penalty
        # Subtract travel time to encourage shortest-path TSP loops
        r_travel = float(t_move)

        # Step Reward
        reward = (w1 * r_uncertainty) + (w2 * r_primitive) + (w3 * r_spread) - (w4 * r_travel)

        # 6. Episode Termination Check
        mean_unc = float(np.mean(self.uncertainties))
        budget_exhausted = (self.step_count >= self.max_touches)
        uncertainty_satisfied = (mean_unc < 0.05)

        valid_actions_remain = np.any(self.action_masks())
        terminated = budget_exhausted or uncertainty_satisfied or (not valid_actions_remain)
        truncated = False

        # 7. Terminal Completion Bonus
        if terminated:
            # Award +20.0 bonus if all 5 visible faces have at least 4 touchpoints
            all_faces_covered = bool(np.all(self.face_touch_counts >= 4))
            low_error = (np.mean(self.touch_errors) < 0.010) if self.touch_errors else False
            if all_faces_covered and low_error:
                reward += 20.0

        obs = self._get_observation()
        info = self._get_info()
        return obs, float(reward), terminated, truncated, info
