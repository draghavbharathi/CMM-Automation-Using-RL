#!/usr/bin/env python3
"""
Geometric Pre-Processing Pipeline (Open3D & NumPy) for CMM Touch-Point Planning.
Loads the merged RealSense point cloud, cleans and downsamples it, estimates outward-oriented
surface normals, computes local curvature and initial sensor uncertainty, segments geometric
primitive faces via iterative RANSAC, and selects K candidate touchpoints via Farthest Point Sampling (FPS).
"""

import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np


@dataclass
class PreprocessedCandidateData:
    """Preprocessed candidate touchpoint data structure for RL observation space."""
    # Fixed K candidate points (K, 3)
    coords: np.ndarray
    # Outward unit normals (K, 3)
    normals: np.ndarray
    # Local surface curvature in [0, 1] (K,)
    curvatures: np.ndarray
    # Initial sensor uncertainty in [0.5, 1.0] (K,)
    initial_uncertainties: np.ndarray
    # Discrete geometric face / primitive ID in {0, ..., M-1} (K,)
    primitive_ids: np.ndarray
    # Centroid that was subtracted (3,)
    centroid: np.ndarray
    # Scaling factor applied for normalization
    scale: float
    # Dense reference point cloud (N, 3) for collision checking
    dense_cloud: np.ndarray
    # Dense reference normals (N, 3)
    dense_normals: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            'coords': self.coords,
            'normals': self.normals,
            'curvatures': self.curvatures,
            'initial_uncertainties': self.initial_uncertainties,
            'primitive_ids': self.primitive_ids,
            'centroid': self.centroid,
            'scale': self.scale,
            'dense_cloud': self.dense_cloud,
            'dense_normals': self.dense_normals,
        }

    def save_npz(self, filepath: str):
        np.savez_compressed(
            filepath,
            coords=self.coords,
            normals=self.normals,
            curvatures=self.curvatures,
            initial_uncertainties=self.initial_uncertainties,
            primitive_ids=self.primitive_ids,
            centroid=self.centroid,
            scale=self.scale,
            dense_cloud=self.dense_cloud,
            dense_normals=self.dense_normals
        )

    @classmethod
    def load_npz(cls, filepath: str) -> 'PreprocessedCandidateData':
        data = np.load(filepath)
        return cls(
            coords=data['coords'],
            normals=data['normals'],
            curvatures=data['curvatures'],
            initial_uncertainties=data['initial_uncertainties'],
            primitive_ids=data['primitive_ids'],
            centroid=data['centroid'],
            scale=float(data['scale']),
            dense_cloud=data['dense_cloud'],
            dense_normals=data['dense_normals']
        )


class PointCloudPreprocessor:
    """
    Open3D-powered geometric pipeline for adaptive metrology sampling.
    """

    # Canonical face normal directions for the 5 visible faces of a box:
    # 0: Top (+Z), 1: North (+X), 2: South (-X), 3: East (+Y), 4: West (-Y)
    CANONICAL_FACE_NORMALS = np.array([
        [0.0, 0.0, 1.0],   # Face 0: Top (+Z)
        [1.0, 0.0, 0.0],   # Face 1: North (+X)
        [-1.0, 0.0, 0.0],  # Face 2: South (-X)
        [0.0, 1.0, 0.0],   # Face 3: East (+Y)
        [0.0, -1.0, 0.0],  # Face 4: West (-Y)
    ], dtype=np.float64)

    FACE_NAMES = ['Top (+Z)', 'North (+X)', 'South (-X)', 'East (+Y)', 'West (-Y)']

    @staticmethod
    def farthest_point_sampling(points: np.ndarray, K: int) -> np.ndarray:
        """
        Pure NumPy Farthest Point Sampling (FPS) selecting K indices uniformly covering points.
        Runs in O(K * N).
        """
        N = len(points)
        if N <= K:
            return np.arange(N)

        selected_indices = np.zeros(K, dtype=np.int64)
        # Start with the point farthest from the centroid
        centroid = np.mean(points, axis=0)
        dists = np.linalg.norm(points - centroid, axis=1)
        selected_indices[0] = np.argmax(dists)

        min_distances = np.linalg.norm(points - points[selected_indices[0]], axis=1)

        for i in range(1, K):
            farthest_idx = np.argmax(min_distances)
            selected_indices[i] = farthest_idx
            new_dists = np.linalg.norm(points - points[farthest_idx], axis=1)
            min_distances = np.minimum(min_distances, new_dists)

        return selected_indices

    @classmethod
    def compute_local_curvatures(cls, points: np.ndarray, k_neighbors: int = 30) -> np.ndarray:
        """
        Compute surface variation (curvature) kappa_i for each point using
        the eigenvalues of the local 3x3 covariance matrix:
            kappa_i = lambda_0 / (lambda_0 + lambda_1 + lambda_2)
        Normalized to [0, 1].
        """
        N = len(points)
        curvatures = np.zeros(N, dtype=np.float64)

        try:
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, N), algorithm='auto').fit(points)
            _, indices = nbrs.kneighbors(points)

            for i in range(N):
                nn_pts = points[indices[i]]
                cov = np.cov(nn_pts, rowvar=False)
                evals = np.linalg.eigvalsh(cov)
                # Ensure sorted non-negative
                evals = np.sort(np.maximum(evals, 0.0))
                trace = np.sum(evals)
                if trace > 1e-12:
                    curvatures[i] = evals[0] / trace
                else:
                    curvatures[i] = 0.0

        except ImportError:
            # Fallback naive KD-tree or spatial grid if sklearn not yet loaded
            for i in range(N):
                dists = np.linalg.norm(points - points[i], axis=1)
                idx = np.argsort(dists)[:min(k_neighbors, N)]
                nn_pts = points[idx]
                cov = np.cov(nn_pts, rowvar=False)
                evals = np.linalg.eigvalsh(cov)
                evals = np.sort(np.maximum(evals, 0.0))
                trace = np.sum(evals)
                curvatures[i] = (evals[0] / trace) if trace > 1e-12 else 0.0

        # Normalize to [0, 1]
        c_min = np.min(curvatures)
        c_max = np.max(curvatures)
        if c_max - c_min > 1e-8:
            curvatures_norm = (curvatures - c_min) / (c_max - c_min)
        else:
            curvatures_norm = np.zeros_like(curvatures)

        return curvatures_norm

    @classmethod
    def segment_box_faces(cls, points: np.ndarray, normals: np.ndarray,
                          dist_thresh: float = 0.004) -> np.ndarray:
        """
        RANSAC / normal-guided geometric primitive segmentation identifying the
        5 visible faces of the demo box:
          0: Top (+Z), 1: North (+X), 2: South (-X), 3: East (+Y), 4: West (-Y).
        Assigns each point an integer face ID in {0, 1, 2, 3, 4}.
        """
        import open3d as o3d
        N = len(points)
        primitive_ids = np.full(N, -1, dtype=np.int64)

        # Segment each of the 5 canonical faces using normal alignment + RANSAC refinement
        for f_idx in range(5):
            f_canon = cls.CANONICAL_FACE_NORMALS[f_idx]
            align = np.dot(normals, f_canon)
            candidate_indices = np.where((align > 0.5) & (primitive_ids == -1))[0]
            if len(candidate_indices) >= 10:
                sub_pcd = o3d.geometry.PointCloud()
                sub_pcd.points = o3d.utility.Vector3dVector(points[candidate_indices])
                try:
                    plane_model, inliers = sub_pcd.segment_plane(
                        distance_threshold=dist_thresh,
                        ransac_n=3,
                        num_iterations=500
                    )
                    inlier_global = candidate_indices[inliers]
                    primitive_ids[inlier_global] = f_idx
                except Exception:
                    pass

        # For unassigned points (or edges/corners), classify by outward normal alignment
        unassigned = primitive_ids < 0
        if np.any(unassigned):
            dots = np.dot(normals[unassigned], cls.CANONICAL_FACE_NORMALS.T)  # (M, 5)
            primitive_ids[unassigned] = np.argmax(dots, axis=1)

        return primitive_ids

    @classmethod
    def process(cls,
                ply_path_or_points,
                K: int = 256,
                voxel_size: float = 0.003,
                normal_search_radius: float = 0.012,
                normal_max_nn: int = 30,
                center_cloud: bool = True) -> PreprocessedCandidateData:
        """
        Execute full Phase 2 pre-processing pipeline on input point cloud:
        1. Voxel downsampling
        2. Outward normal estimation
        3. Local curvature estimation
        4. RANSAC face primitive segmentation
        5. Initial sensor uncertainty score in [0.5, 1.0]
        6. Farthest Point Sampling to K candidate touchpoints.
        """
        import open3d as o3d

        # 1. Load Point Cloud
        if isinstance(ply_path_or_points, str):
            if not os.path.isfile(ply_path_or_points):
                raise FileNotFoundError(f"Point cloud file not found: {ply_path_or_points}")
            pcd = o3d.io.read_point_cloud(ply_path_or_points)
        elif isinstance(ply_path_or_points, o3d.geometry.PointCloud):
            pcd = ply_path_or_points
        elif isinstance(ply_path_or_points, np.ndarray):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(ply_path_or_points.astype(np.float64))
        else:
            raise TypeError("Expected filepath, open3d PointCloud, or numpy array")

        # Strip turntable/table support surface (box is 10x10x10cm resting on base at z=0.04)
        pts_raw = np.asarray(pcd.points)
        if len(pts_raw) > 0 and (np.max(np.abs(pts_raw[:, :2])) > 0.07 or np.min(pts_raw[:, 2]) < 0.042):
            box_filter = (np.abs(pts_raw[:, 0]) <= 0.058) & (np.abs(pts_raw[:, 1]) <= 0.058) & (pts_raw[:, 2] >= 0.042)
            if np.sum(box_filter) > 1000:
                pcd = pcd.select_by_index(np.where(box_filter)[0])

        raw_n = len(pcd.points)

        # 2. Voxel Downsampling to uniform density
        if voxel_size > 0:
            pcd_voxel = pcd.voxel_down_sample(voxel_size=voxel_size)
        else:
            pcd_voxel = pcd

        # Statistical outlier removal to ensure crisp edges
        cl, ind = pcd_voxel.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd_clean = pcd_voxel.select_by_index(ind)

        dense_pts = np.asarray(pcd_clean.points, dtype=np.float64)

        # Compute centroid and center at (0, 0, 0)
        centroid = np.mean(dense_pts, axis=0)
        if center_cloud:
            dense_pts_centered = dense_pts - centroid
        else:
            dense_pts_centered = dense_pts.copy()

        pcd_clean.points = o3d.utility.Vector3dVector(dense_pts_centered)

        # 3. Surface Normal Estimation
        pcd_clean.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_search_radius,
                max_nn=normal_max_nn
            )
        )
        normals = np.asarray(pcd_clean.normals, dtype=np.float64)

        # Orient all normals consistently OUTWARD away from centroid (0, 0, 0)
        # For outward normal, dot product with position vector from center must be > 0
        dots = np.sum(normals * dense_pts_centered, axis=1)
        flip_mask = dots < 0
        normals[flip_mask] = -normals[flip_mask]

        # Re-normalize to unit vectors
        norm_lens = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.where(norm_lens > 1e-8, normals / norm_lens, np.array([0.0, 0.0, 1.0]))
        pcd_clean.normals = o3d.utility.Vector3dVector(normals)

        # 4. Local Surface Curvature
        curvatures = cls.compute_local_curvatures(dense_pts_centered, k_neighbors=30)

        # 5. RANSAC Geometric Primitive Segmentation
        primitive_ids = cls.segment_box_faces(dense_pts_centered, normals, dist_thresh=0.004)

        # 6. Initial Sensor Uncertainty Score: sigma_i^(0) in [0.5, 1.0]
        # Edges and corners (high curvature / high variance) have sigma -> 1.0; flat faces have sigma -> 0.5
        initial_uncertainties = 0.5 + 0.5 * curvatures

        # 7. Farthest Point Sampling to K candidate touchpoints
        N_clean = len(dense_pts_centered)
        if N_clean > K:
            # Open3D FPS
            try:
                pcd_fps = pcd_clean.farthest_point_down_sample(num_samples=K)
                fps_pts = np.asarray(pcd_fps.points)
                # Find matching indices in dense_pts_centered
                from scipy.spatial import cKDTree
                tree = cKDTree(dense_pts_centered)
                _, selected_indices = tree.query(fps_pts, k=1)
            except Exception:
                selected_indices = cls.farthest_point_sampling(dense_pts_centered, K)
        else:
            selected_indices = np.arange(N_clean)

        candidate_coords = dense_pts_centered[selected_indices]
        candidate_normals = normals[selected_indices]
        candidate_curvatures = curvatures[selected_indices]
        candidate_uncertainties = initial_uncertainties[selected_indices]
        candidate_primitive_ids = primitive_ids[selected_indices]

        # Calculate bounding box scale for normalized observations
        bbox_diag = float(np.linalg.norm(np.ptp(dense_pts_centered, axis=0)))
        scale = max(bbox_diag, 0.10)

        data = PreprocessedCandidateData(
            coords=candidate_coords.astype(np.float32),
            normals=candidate_normals.astype(np.float32),
            curvatures=candidate_curvatures.astype(np.float32),
            initial_uncertainties=candidate_uncertainties.astype(np.float32),
            primitive_ids=candidate_primitive_ids.astype(np.int64),
            centroid=centroid.astype(np.float32),
            scale=scale,
            dense_cloud=dense_pts_centered.astype(np.float32),
            dense_normals=normals.astype(np.float32)
        )

        return data


def main():
    parser = argparse.ArgumentParser(description="Preprocess merged RealSense point cloud for CMM RL")
    parser.add_argument("--input", type=str, default="merged_box_cloud.ply", help="Path to input PLY file")
    parser.add_argument("--K", type=int, default=256, help="Number of candidate touchpoints")
    parser.add_argument("--voxel-size", type=float, default=0.003, help="Voxel size in meters")
    parser.add_argument("--output", type=str, default="candidates_256.npz", help="Output NPZ file")
    args = parser.parse_args()

    print(f"[INFO] Loading and preprocessing '{args.input}' with K={args.K}...")
    data = PointCloudPreprocessor.process(args.input, K=args.K, voxel_size=args.voxel_size)

    data.save_npz(args.output)
    print(f"[SUCCESS] Saved preprocessed candidate dataset to '{args.output}'!")
    print(f"  - Total Candidates K     : {len(data.coords)}")
    print(f"  - Coordinate Bounds X    : [{data.coords[:, 0].min():.4f}, {data.coords[:, 0].max():.4f}] m")
    print(f"  - Coordinate Bounds Y    : [{data.coords[:, 1].min():.4f}, {data.coords[:, 1].max():.4f}] m")
    print(f"  - Coordinate Bounds Z    : [{data.coords[:, 2].min():.4f}, {data.coords[:, 2].max():.4f}] m")
    print(f"  - Mean Initial Uncertainty: {np.mean(data.initial_uncertainties):.4f}")
    print(f"  - Primitive Distribution :")
    for face_id, name in enumerate(PointCloudPreprocessor.FACE_NAMES):
        count = int(np.sum(data.primitive_ids == face_id))
        print(f"    * Face {face_id} ({name:12s}): {count:3d} candidate points ({count/args.K*100:.1f}%)")


if __name__ == "__main__":
    main()
