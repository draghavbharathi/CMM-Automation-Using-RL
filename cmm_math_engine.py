#!/usr/bin/env python3
"""
Pure Mathematical CMM Simulation Engine.
Replaces slow physical robot simulation with three ultra-fast analytical operations:
1. Ground-Truth Touch via Ray-Mesh Intersection with CAD model.
2. Two-Cylinder Mathematical Collision Detection for Stylus & Probe Head (Action Masking).
3. Chebyshev Clearance-Box Travel Time (t_move) via 3-segment retract-traverse-plunge path.
"""

import math
from typing import Tuple, Optional, Union
import numpy as np


class CMMMathEngine:
    """
    Fast analytical CMM simulation engine capable of 10,000+ evaluations per second.
    """

    def __init__(self,
                 box_dimensions: Tuple[float, float, float] = (0.10, 0.10, 0.10),
                 box_center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 probe_mode: str = 'mode_a',
                 stylus_length: float = 0.050,    # 50mm stylus
                 stylus_radius: float = 0.0015,   # 1.5mm radius shaft (3mm diameter)
                 head_length: float = 0.080,      # 80mm probe head (Renishaw PH20/PH10)
                 head_radius: float = 0.025,      # 25mm radius head (50mm diameter)
                 safety_margin: float = 0.002,    # 2mm safety margin
                 axis_speeds: Tuple[float, float, float] = (0.15, 0.15, 0.15),  # 150mm/s
                 d_standoff: float = 0.020,       # 20mm standoff
                 z_min_clearance: float = -0.048):  # Table floor clearance (2mm above bottom)
        self.box_dimensions = np.array(box_dimensions, dtype=np.float64)
        self.box_center = np.array(box_center, dtype=np.float64)
        self.probe_mode = probe_mode.lower()
        self.stylus_length = float(stylus_length)
        self.stylus_radius = float(stylus_radius)
        self.head_length = float(head_length)
        self.head_radius = float(head_radius)
        self.safety_margin = float(safety_margin)
        self.axis_speeds = np.array(axis_speeds, dtype=np.float64)
        self.d_standoff = float(d_standoff)
        self.z_min_clearance = float(z_min_clearance)

        # Precompute CAD box AABB bounds
        self.box_min = self.box_center - self.box_dimensions / 2.0
        self.box_max = self.box_center + self.box_dimensions / 2.0

        # Optional Trimesh CAD representation
        self.mesh = None
        self._init_mesh()

    def _init_mesh(self):
        try:
            import trimesh
            self.mesh = trimesh.creation.box(extents=self.box_dimensions)
            self.mesh.apply_translation(self.box_center)
        except Exception:
            self.mesh = None

    def ray_box_intersection(self, ray_origin: np.ndarray, ray_dir: np.ndarray) -> Optional[np.ndarray]:
        """
        Fast analytical slab ray-box intersection against CAD box.
        Returns true intersection point on box surface or None if ray misses.
        """
        # Direction unit vector
        norm_dir = np.linalg.norm(ray_dir)
        if norm_dir < 1e-8:
            return None
        d = ray_dir / norm_dir

        tmin = -1e9
        tmax = 1e9

        for i in range(3):
            if abs(d[i]) < 1e-8:
                if ray_origin[i] < self.box_min[i] or ray_origin[i] > self.box_max[i]:
                    return None
            else:
                t1 = (self.box_min[i] - ray_origin[i]) / d[i]
                t2 = (self.box_max[i] - ray_origin[i]) / d[i]
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmin > tmax or tmax < 0:
                    return None

        t_hit = tmin if tmin >= 0 else tmax
        if t_hit < 0:
            return None

        hit_point = ray_origin + t_hit * d
        # Clamp to exact face boundary to eliminate float roundoff
        hit_point = np.clip(hit_point, self.box_min, self.box_max)
        return hit_point

    def ground_truth_touch(self, p_i: np.ndarray, n_i: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Simulate micron-accurate ground-truth touch via ray casting from standoff point:
            p_standoff = p_i + d_standoff * n_i
            ray cast along direction -n_i against Ground-Truth CAD box.
        Returns:
            p_true: True micron-accurate surface coordinate (3,)
            deviation_error: True Euclidean deviation ||p_true - p_i||_2 in meters.
        """
        norm_n = np.linalg.norm(n_i)
        unit_n = n_i / norm_n if norm_n > 1e-8 else np.array([0.0, 0.0, 1.0])

        p_standoff = p_i + self.d_standoff * unit_n
        ray_dir = -unit_n

        hit_pt = self.ray_box_intersection(p_standoff, ray_dir)

        if hit_pt is None:
            # Fallback to nearest surface point on CAD box
            hit_pt = np.clip(p_i, self.box_min, self.box_max)

        deviation_error = float(np.linalg.norm(hit_pt - p_i))
        return hit_pt, deviation_error

    def get_approach_direction(self, normal: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Compute probe approach vector a based on probe mode:
          Mode A (5-Axis Indexing Head, Renishaw PH20): a = normal
          Mode B (Fixed Vertical Z-Probe): a = [0, 0, 1]
        Returns:
          approach_vector: Unit vector (3,)
          valid_direction: False if probe points down into table (nz < -0.1) in Mode A.
        """
        norm_n = np.linalg.norm(normal)
        unit_n = normal / norm_n if norm_n > 1e-8 else np.array([0.0, 0.0, 1.0])

        if self.probe_mode == 'mode_b':
            # Fixed vertical probe: always enters from +Z
            return np.array([0.0, 0.0, 1.0], dtype=np.float64), True
        else:
            # Mode A: 5-axis indexing head aligns with outward normal
            if unit_n[2] < -0.1:
                # Approach vector points down toward floor (collides with machine table)
                return unit_n, False
            return unit_n, True

    def check_two_cylinder_collision(self, p_i: np.ndarray, n_i: np.ndarray,
                                     cloud_points: np.ndarray) -> bool:
        """
        Check if physical CMM stylus shaft or probe head collides with workpiece.
        Returns:
            True if candidate point p_i is in COLLISION (invalid action), False if safe.
        """
        # Table Floor Clearance Check
        if p_i[2] < self.z_min_clearance:
            return True

        # Approach direction
        approach_dir, valid_dir = self.get_approach_direction(n_i)
        if not valid_dir:
            return True

        if len(cloud_points) == 0:
            return False

        # Vector from target point to all cloud points: v_j = q_j - p_i
        V = cloud_points - p_i  # (N, 3)

        # Axial projection: h_j = v_j . a
        h = np.dot(V, approach_dir)  # (N,)

        # Radial distance: r_j = ||v_j - h_j * a||_2
        V_perp = V - np.outer(h, approach_dir)  # (N, 3)
        r = np.linalg.norm(V_perp, axis=1)      # (N,)

        # 1. Stylus Cylinder Collision Check
        # Ignore contact ruby tip zone (h <= 0.003m)
        stylus_mask = (h > 0.003) & (h <= self.stylus_length) & \
                      (r < (self.stylus_radius + self.safety_margin))
        if np.any(stylus_mask):
            return True

        # 2. Probe Head Cylinder Collision Check
        head_min_h = self.stylus_length
        head_max_h = self.stylus_length + self.head_length
        head_mask = (h > head_min_h) & (h <= head_max_h) & \
                    (r < (self.head_radius + self.safety_margin))
        if np.any(head_mask):
            return True

        return False

    def compute_action_validity_mask(self, candidate_coords: np.ndarray,
                                     candidate_normals: np.ndarray,
                                     dense_cloud: np.ndarray) -> np.ndarray:
        """
        Compute boolean validity mask of length K across all candidate points.
        Returns:
            valid_mask: np.ndarray of bool (K,), where True means SAFE (action valid).
        """
        K = len(candidate_coords)
        valid_mask = np.zeros(K, dtype=bool)

        for i in range(K):
            in_collision = self.check_two_cylinder_collision(
                candidate_coords[i],
                candidate_normals[i],
                dense_cloud
            )
            valid_mask[i] = not in_collision

        return valid_mask

    def compute_travel_time(self,
                            p_prev: np.ndarray,
                            p_curr: np.ndarray,
                            n_curr: np.ndarray,
                            z_safe: Optional[float] = None,
                            cloud_max_z: Optional[float] = None) -> float:
        """
        Compute Chebyshev Clearance-Box Travel Time for 3-segment safe CMM trajectory:
        1. Retract: p_prev -> P2 = [p_prev_x, p_prev_y, Z_safe]
        2. Traverse: P2 -> P3 = [p_standoff_x, p_standoff_y, Z_safe]
        3. Plunge: P3 -> p_curr

        Using Chebyshev (L_infinity) norm across independent Cartesian CMM axes:
            t_segment = max(|dx|/vx, |dy|/vy, |dz|/vz)
        """
        if z_safe is None:
            ref_z = cloud_max_z if cloud_max_z is not None else self.box_max[2]
            z_safe = ref_z + 0.030  # 30mm clearance above highest point

        # Standoff coordinate before plunge
        norm_n = np.linalg.norm(n_curr)
        unit_n = n_curr / norm_n if norm_n > 1e-8 else np.array([0.0, 0.0, 1.0])
        p_standoff = p_curr + self.d_standoff * unit_n

        P1 = p_prev
        P2 = np.array([p_prev[0], p_prev[1], z_safe], dtype=np.float64)
        P3 = np.array([p_standoff[0], p_standoff[1], z_safe], dtype=np.float64)
        P4 = p_curr

        vx, vy, vz = self.axis_speeds

        # Segment 1: Retract
        t1 = max(abs(P2[0] - P1[0]) / vx, abs(P2[1] - P1[1]) / vy, abs(P2[2] - P1[2]) / vz)
        # Segment 2: Traverse
        t2 = max(abs(P3[0] - P2[0]) / vx, abs(P3[1] - P2[1]) / vy, abs(P3[2] - P2[2]) / vz)
        # Segment 3: Plunge
        t3 = max(abs(P4[0] - P3[0]) / vx, abs(P4[1] - P3[1]) / vy, abs(P4[2] - P3[2]) / vz)

        return float(t1 + t2 + t3)
