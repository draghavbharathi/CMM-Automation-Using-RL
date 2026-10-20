#!/usr/bin/env python3
"""
Evaluation and 3D Visualizer for Autonomous CMM RL Planner.
Runs deterministic rollout with trained MaskablePPO model on the fused point cloud,
prints an inspection summary table, and opens an interactive 3D visualizer showing:
1. The RealSense point cloud colored by surface uncertainty (Blue=Low, Red=High),
2. The chosen CMM touchpoints as bright green spheres,
3. Red arrows/lines indicating outward surface normal vectors,
4. Blue lines tracing the 3-segment clearance-box CMM trajectory from touch #1 to #T_max.
"""

import os
import sys
import math
import argparse
from typing import List, Tuple
import numpy as np

from cmm_env import CMMTouchPointEnv
from pointcloud_preprocessor import PointCloudPreprocessor, PreprocessedCandidateData


def visualize_cmm_inspection_3d(env: CMMTouchPointEnv, rollout_records: List[dict], show_gui: bool = False):
    """
    Construct Open3D visualization with:
    - Merged point cloud colored by remaining uncertainty (Blue=0% -> Red=100%)
    - Chosen touchpoints as bright green spheres
    - Red lines/arrows indicating outward surface normal vectors
    - Blue lines tracing the 3-segment CMM clearance trajectory
    """
    try:
        import open3d as o3d
    except ImportError:
        print("[WARN] Open3D not installed. Skipping graphical visualizer.")
        return

    geometries = []

    # 1. Point Cloud Colored by Remaining Uncertainty
    pts = env.coords
    unc = env.uncertainties
    # Colormap: Blue (0.0 unc) -> Cyan (0.3) -> Yellow (0.7) -> Red (1.0 unc)
    colors = np.zeros_like(pts)
    for i, u in enumerate(unc):
        u_clamped = float(np.clip(u, 0.0, 1.0))
        # Blue to Red gradient
        colors[i] = [u_clamped, 0.2, 1.0 - u_clamped]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    geometries.append(pcd)

    # 2. Chosen CMM Touchpoints as Bright Green Spheres
    for r in rollout_records:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.0025)
        sphere.translate(r['point'].astype(np.float64))
        sphere.paint_uniform_color([0.0, 1.0, 0.0])  # Bright green
        sphere.compute_vertex_normals()
        geometries.append(sphere)

    # 3. Red Lines / Arrows for Normal Vectors
    normal_lines_pts = []
    normal_lines_idx = []
    arrow_len = 0.015  # 15mm arrow length

    for i, r in enumerate(rollout_records):
        p0 = r['point'].astype(np.float64)
        p1 = p0 + arrow_len * r['normal'].astype(np.float64)
        idx_start = len(normal_lines_pts)
        normal_lines_pts.append(p0)
        normal_lines_pts.append(p1)
        normal_lines_idx.append([idx_start, idx_start + 1])

    if normal_lines_pts:
        normal_lineset = o3d.geometry.LineSet()
        normal_lineset.points = o3d.utility.Vector3dVector(np.array(normal_lines_pts))
        normal_lineset.lines = o3d.utility.Vector2iVector(np.array(normal_lines_idx))
        normal_lineset.paint_uniform_color([1.0, 0.1, 0.1])  # Bright Red
        geometries.append(normal_lineset)

    # 4. Blue Lines Tracing 3-Segment CMM Clearance-Box Trajectory
    traj_pts = []
    traj_lines = []
    z_safe = float(env.coords[:, 2].max() + 0.030)

    # Start from initial resting position
    curr_pos = np.array([0.0, 0.0, z_safe], dtype=np.float64)
    traj_pts.append(curr_pos)

    for r in rollout_records:
        p_target = r['point'].astype(np.float64)
        n_target = r['normal'].astype(np.float64)
        p_standoff = p_target + 0.020 * n_target

        # 3 Segments:
        # Segment 1: Retract to Z_safe
        P2 = np.array([curr_pos[0], curr_pos[1], z_safe], dtype=np.float64)
        # Segment 2: Traverse above object
        P3 = np.array([p_standoff[0], p_standoff[1], z_safe], dtype=np.float64)
        # Segment 3: Plunge to touchpoint
        P4 = p_target

        for P_next in [P2, P3, P4]:
            idx_curr = len(traj_pts) - 1
            traj_pts.append(P_next)
            idx_next = len(traj_pts) - 1
            traj_lines.append([idx_curr, idx_next])
            curr_pos = P_next

    if traj_lines:
        traj_lineset = o3d.geometry.LineSet()
        traj_lineset.points = o3d.utility.Vector3dVector(np.array(traj_pts))
        traj_lineset.lines = o3d.utility.Vector2iVector(np.array(traj_lines))
        traj_lineset.paint_uniform_color([0.1, 0.5, 1.0])  # Blue
        geometries.append(traj_lineset)

    # 5. Coordinate Frame for Reference (Origin)
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03, origin=[0, 0, 0])
    geometries.append(coord_frame)

    # Always export 3D inspection artifacts to disk
    o3d.io.write_point_cloud("cmm_uncertainty_cloud.ply", pcd)
    print("  [SUCCESS] Exported uncertainty-colored cloud to 'cmm_uncertainty_cloud.ply'")

    combined_mesh = o3d.geometry.TriangleMesh()
    for g in geometries:
        if isinstance(g, o3d.geometry.TriangleMesh) and g != coord_frame:
            combined_mesh += g
    if len(combined_mesh.vertices) > 0:
        o3d.io.write_triangle_mesh("cmm_touchpoint_spheres.ply", combined_mesh)
        print("  [SUCCESS] Exported 3D touchpoint spheres to 'cmm_touchpoint_spheres.ply'")

    if normal_lines_pts:
        o3d.io.write_line_set("cmm_normal_arrows.ply", normal_lineset)
        print("  [SUCCESS] Exported approach normal vectors to 'cmm_normal_arrows.ply'")

    if traj_lines:
        o3d.io.write_line_set("cmm_trajectory_path.ply", traj_lineset)
        print("  [SUCCESS] Exported 3-segment clearance trajectory to 'cmm_trajectory_path.ply'")

    # Interactive GUI window if requested
    if show_gui:
        has_display = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
        if has_display:
            print("[INFO] Launching interactive Open3D Visualizer Window...")
            print("  - Green Spheres : CMM Touchpoints")
            print("  - Red Vectors   : Outward Approach Normals")
            print("  - Blue Lines    : 3-Segment Clearance Trajectory")
            print("  - Colormap Cloud: Surface Uncertainty (Blue=0% -> Red=100%)")
            try:
                o3d.visualization.draw_geometries(
                    geometries,
                    window_name="CMM Next-Best-Touch RL Inspection Trajectory",
                    width=1280,
                    height=800,
                    left=50,
                    top=50
                )
            except Exception as e:
                print(f"[WARN] Open3D graphical window could not open ({e}).")
        else:
            print("[INFO] Headless environment detected (no display). 3D assets exported to PLY.")


def evaluate_policy(model_path: str = 'cmm_maskable_ppo.zip',
                    npz_path: str = 'candidates_256.npz',
                    render_3d: bool = True,
                    show_gui: bool = False):
    from sb3_contrib import MaskablePPO

    print(f"[INFO] Loading environment with candidate dataset '{npz_path}'...")
    env = CMMTouchPointEnv(
        npz_path=npz_path,
        K=256,
        max_touches=20,
        domain_randomize=False
    )

    print(f"[INFO] Loading trained MaskablePPO model from '{model_path}'...")
    model = MaskablePPO.load(model_path, env=env)

    obs, info = env.reset()
    terminated = False
    truncated = False

    rollout_records = []
    print("\n" + "="*95)
    print("           AUTONOMOUS CMM NEXT-BEST-TOUCH INSPECTION ROLLOUT (MaskablePPO)")
    print("="*95)

    initial_mean_unc = float(np.mean(env.uncertainties))

    step_idx = 0
    while not (terminated or truncated):
        step_idx += 1
        masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        action = int(action)

        chosen_pt = env.coords[action].copy()
        chosen_normal = env.normals[action].copy()
        face_id = int(env.primitive_ids[action])
        face_name = PointCloudPreprocessor.FACE_NAMES[face_id]

        prev_probe_pos = env.probe_pos.copy()
        t_move = env.math_engine.compute_travel_time(
            p_prev=prev_probe_pos,
            p_curr=chosen_pt,
            n_curr=chosen_normal,
            cloud_max_z=float(env.coords[:, 2].max())
        )

        obs, reward, terminated, truncated, info = env.step(action)
        remaining_unc_pct = float(np.mean(env.uncertainties) * 100.0)

        record = {
            'step': step_idx,
            'point': chosen_pt,
            'normal': chosen_normal,
            'face_id': face_id,
            'face_name': face_name,
            'travel_time': t_move,
            'remaining_unc_pct': remaining_unc_pct,
            'reward': reward
        }
        rollout_records.append(record)

    # 1. Print formatted inspection summary table
    print(f"{'Step #':^7} | {'Point XYZ (m)':^26} | {'Normal Vector':^26} | {'Face ID & Name':^16} | {'Move (s)':^8} | {'Remaining Unc':^13}")
    print("-" * 105)
    for r in rollout_records:
        pt_str = f"({r['point'][0]:+.3f}, {r['point'][1]:+.3f}, {r['point'][2]:+.3f})"
        norm_str = f"[{r['normal'][0]:+.2f}, {r['normal'][1]:+.2f}, {r['normal'][2]:+.2f}]"
        face_str = f"{r['face_id']}: {r['face_name']}"
        print(f" {r['step']:^5d} | {pt_str:^26} | {norm_str:^26} | {face_str:^16} | {r['travel_time']:^8.3f} | {r['remaining_unc_pct']:^11.1f}%")

    print("-" * 105)
    final_mean_unc = float(np.mean(env.uncertainties))
    total_travel_time = env.total_travel_time
    face_counts = env.face_touch_counts
    all_5_faces_covered = bool(np.all(face_counts >= 4))

    print(f"\n[SUMMARY RESULTS]")
    print(f"  * Total Touches Executed   : {step_idx} / {env.max_touches}")
    print(f"  * Total CMM Travel Time    : {total_travel_time:.2f} seconds")
    print(f"  * Uncertainty Reduction    : {initial_mean_unc*100:.1f}% -> {final_mean_unc*100:.1f}% (Drop: {(initial_mean_unc - final_mean_unc)*100:.1f}%)")
    print(f"  * Face Coverage Breakdown  :")
    for fid, name in enumerate(PointCloudPreprocessor.FACE_NAMES):
        print(f"    - Face {fid} ({name:12s}): {face_counts[fid]} touches {'(PASS >= 4)' if face_counts[fid] >= 4 else '(WARN < 4)'}")
    print(f"  * Metrology Requirement (>= 4 pts/face): {'MET! Fully Plane-Fittable' if all_5_faces_covered else 'PARTIAL'}")

    # 2. 3D Open3D Visualization & PLY Export
    if render_3d:
        visualize_cmm_inspection_3d(env, rollout_records, show_gui=show_gui)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CMM Next-Best-Touch RL Policy")
    parser.add_argument("--model", type=str, default="cmm_maskable_ppo.zip", help="Path to trained model zip")
    parser.add_argument("--npz", type=str, default="candidates_256.npz", help="Path to preprocessed candidates NPZ")
    parser.add_argument("--no-render", action="store_true", help="Skip 3D visualizer export")
    parser.add_argument("--gui", action="store_true", help="Open interactive GUI window")
    args = parser.parse_args()

    evaluate_policy(
        model_path=args.model,
        npz_path=args.npz,
        render_3d=not args.no_render,
        show_gui=args.gui
    )


if __name__ == "__main__":
    main()
