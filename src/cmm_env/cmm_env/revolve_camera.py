#!/usr/bin/env python3
"""
Revolve Camera Node for CMM Simulation in Gazebo Jetty.
Revolves the RealSense camera 360 degrees around the demo box in 20-degree increments
(18 stops: 0°, 20°, ..., 340°), captures high-fidelity PointCloud2 data at each stop,
transforms local clouds into the global world frame, merges them, and saves
a unified point cloud (merged_box_cloud.ply).

Fixes applied for corner null/empty bug:
1. Increased orbit radius R = 0.85m (safely >= 0.8m) to avoid near-clipping plane.
2. Non-blocking ROS spin loop with dwell settling time and validation check (>100 non-NaN points).
3. Clean look-at quaternion pointing camera link +X toward (0, 0, target_z).
4. Floor and base filtering (z > z_floor_threshold).
5. Open3D statistical outlier removal and PLY export.
"""

import sys
import os
import math
import time
import argparse
import numpy as np

# Ensure Gazebo Python libraries are accessible
gz_py_path = '/home/raghav/workspace/install/lib/python'
if gz_py_path not in sys.path:
    sys.path.insert(0, gz_py_path)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField

try:
    import gz.transport
    from gz.msgs.pose_pb2 import Pose
    from gz.msgs.boolean_pb2 import Boolean
    GZ_TRANSPORT_AVAILABLE = True
except ImportError:
    GZ_TRANSPORT_AVAILABLE = False


def euler_to_quaternion(roll: float, pitch: float, yaw: float):
    """Convert Euler angles (roll, pitch, yaw in radians) to normalized quaternion (x, y, z, w)."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm > 1e-9:
        qx /= norm
        qy /= norm
        qz /= norm
        qw /= norm
    return qx, qy, qz, qw


def quat_to_rot_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Compute 3x3 rotation matrix from quaternion (qx, qy, qz, qw)."""
    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw),       2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw),       1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw),       2.0 * (qy * qz + qx * qw),       1.0 - 2.0 * (qx * qx + qy * qy)]
    ], dtype=np.float64)


class CameraRevolver(Node):
    def __init__(self, step_deg: float = 20.0, settle_sec: float = 0.5,
                 radius: float = 0.85, height: float = 0.45,
                 target_x: float = 0.0, target_y: float = 0.0, target_z: float = 0.09,
                 z_floor_threshold: float = 0.041, output_ply: str = 'merged_box_cloud.ply',
                 world_name: str = 'cmm_world', model_name: str = 'realsense_camera',
                 model_id: int = 0, loop: bool = False):
        super().__init__('camera_revolver')

        self.step_deg = float(step_deg)
        self.settle_sec = float(settle_sec)
        self.radius = float(radius)
        self.height = float(height)
        self.target_x = float(target_x)
        self.target_y = float(target_y)
        self.target_z = float(target_z)
        self.z_floor_threshold = float(z_floor_threshold)
        self.output_ply = output_ply
        self.world_name = world_name
        self.model_name = model_name
        self.model_id = model_id
        self.loop = loop

        self.service_name = f'/world/{self.world_name}/set_pose'
        self.gz_node = gz.transport.Node() if GZ_TRANSPORT_AVAILABLE else None

        # PointCloud2 subscription with reliable QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        self.sub_pc = self.create_subscription(
            PointCloud2,
            '/camera/points',
            self._on_point_cloud,
            qos
        )
        self.sub_pc_alias = self.create_subscription(
            PointCloud2,
            '/camera/depth/color/points',
            self._on_point_cloud,
            qos
        )

        self.latest_pc_msg = None
        self.has_new_pointcloud = False

        self.get_logger().info('Camera Revolver & Point Cloud Fusion Node initialized:')
        self.get_logger().info(f'  - Step increment  : {self.step_deg}° ({int(round(360.0 / self.step_deg))} stops)')
        self.get_logger().info(f'  - Orbit radius    : {self.radius:.3f}m (safe from corner clipping)')
        self.get_logger().info(f'  - Camera height   : {self.height:.3f}m')
        self.get_logger().info(f'  - Target center   : ({self.target_x:.3f}, {self.target_y:.3f}, {self.target_z:.3f})m')
        self.get_logger().info(f'  - Floor threshold : z > {self.z_floor_threshold:.3f}m')
        self.get_logger().info(f'  - Output PLY file : {self.output_ply}')

    def _on_point_cloud(self, msg: PointCloud2):
        self.latest_pc_msg = msg
        self.has_new_pointcloud = True

    def compute_camera_pose(self, angle_deg: float):
        """
        Compute camera position (x, y, z), Euler angles (roll, pitch, yaw),
        quaternion (qx, qy, qz, qw), and 3x3 rotation matrix for given orbit angle.
        """
        theta = math.radians(angle_deg)
        x = self.radius * math.cos(theta)
        y = self.radius * math.sin(theta)
        z = self.height

        dx = self.target_x - x
        dy = self.target_y - y
        dz = self.target_z - z
        dist_xy = math.hypot(dx, dy)

        # Looking down toward target: pitch > 0 rotates about +Y to point down
        pitch = math.atan2(self.height - self.target_z, dist_xy)
        # Yaw points from (x, y) back to (0, 0)
        yaw = theta + math.pi
        yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
        roll = 0.0

        qx, qy, qz, qw = euler_to_quaternion(roll, pitch, yaw)
        R = quat_to_rot_matrix(qx, qy, qz, qw)
        pos = np.array([x, y, z], dtype=np.float64)

        return pos, (qx, qy, qz, qw), R, (roll, pitch, yaw)

    def set_camera_pose(self, angle_deg: float) -> bool:
        """Call Gazebo set_pose service to place camera at given angle."""
        pos, (qx, qy, qz, qw), _, _ = self.compute_camera_pose(angle_deg)
        x, y, z = pos

        if self.gz_node:
            req = Pose()
            req.name = self.model_name
            if self.model_id > 0:
                req.id = self.model_id
            req.position.x = float(x)
            req.position.y = float(y)
            req.position.z = float(z)
            req.orientation.x = float(qx)
            req.orientation.y = float(qy)
            req.orientation.z = float(qz)
            req.orientation.w = float(qw)

            res, rep = self.gz_node.request(self.service_name, req, Pose, Boolean, 2000)
            if res and rep.data:
                return True

        # Fallback to gz service CLI
        import subprocess
        id_str = f'id: {self.model_id}, ' if self.model_id > 0 else ''
        req_content = (
            f'name: "{self.model_name}", {id_str}'
            f'position: {{x: {x:.4f}, y: {y:.4f}, z: {z:.4f}}}, '
            f'orientation: {{x: {qx:.5f}, y: {qy:.5f}, z: {qz:.5f}, w: {qw:.5f}}}'
        )
        cmd = [
            'gz', 'service', '-s', self.service_name,
            '--reqtype', 'gz.msgs.Pose',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '2000',
            '--req', req_content
        ]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            return p.returncode == 0 and 'data: true' in p.stdout
        except Exception as e:
            self.get_logger().error(f'gz service CLI failed: {e}')
            return False

    def unpack_pointcloud2(self, msg: PointCloud2):
        """
        Unpack ROS PointCloud2 message into numpy arrays for coordinates and RGB colors.
        Handles both 16-byte packed formats and arbitrary field layouts.
        """
        n_pts = msg.width * msg.height
        if n_pts == 0 or len(msg.data) == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        data = np.frombuffer(msg.data, dtype=np.uint8)
        step = msg.point_step

        field_offsets = {f.name: f.offset for f in msg.fields}
        if 'x' in field_offsets and 'y' in field_offsets and 'z' in field_offsets:
            ox = field_offsets['x']
            oy = field_offsets['y']
            oz = field_offsets['z']
            
            # Fast vectorized extraction if standard 16-byte layout
            if step == 16 and ox == 0 and oy == 4 and oz == 8:
                pts = np.frombuffer(msg.data, dtype=np.float32).reshape(n_pts, 4)[:, :3].copy()
                bytes_2d = data.reshape(n_pts, step)
                if 'rgb' in field_offsets:
                    orgb = field_offsets['rgb']
                    colors = bytes_2d[:, orgb:orgb+3].astype(np.float32) / 255.0
                else:
                    colors = np.ones((n_pts, 3), dtype=np.float32) * 0.8
                return pts, colors

        # Generic field extraction
        bytes_2d = data.reshape(n_pts, step)
        x = np.frombuffer(bytes_2d[:, ox:ox+4].tobytes(), dtype=np.float32)
        y = np.frombuffer(bytes_2d[:, oy:oy+4].tobytes(), dtype=np.float32)
        z = np.frombuffer(bytes_2d[:, oz:oz+4].tobytes(), dtype=np.float32)
        pts = np.column_stack([x, y, z])

        if 'rgb' in field_offsets:
            orgb = field_offsets['rgb']
            colors = bytes_2d[:, orgb:orgb+3].astype(np.float32) / 255.0
        else:
            colors = np.ones((n_pts, 3), dtype=np.float32) * 0.8

        return pts, colors

    def capture_valid_cloud(self, timeout_sec: float = 5.0):
        """
        Wait in a non-blocking ROS spin loop until a settled, valid PointCloud2
        message with > 100 valid non-NaN points is received.
        """
        # Settle camera motion while processing callbacks
        settle_end = time.time() + self.settle_sec
        while rclpy.ok() and time.time() < settle_end:
            rclpy.spin_once(self, timeout_sec=0.02)

        # Wait for a fresh message
        self.has_new_pointcloud = False
        start_time = time.time()
        while rclpy.ok() and (time.time() - start_time) < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.02)
            if self.has_new_pointcloud and self.latest_pc_msg is not None:
                pts, colors = self.unpack_pointcloud2(self.latest_pc_msg)
                # Filter NaNs and Infs
                valid_mask = np.all(np.isfinite(pts), axis=1)
                pts_valid = pts[valid_mask]
                colors_valid = colors[valid_mask]

                if len(pts_valid) >= 100:
                    return pts_valid, colors_valid
                # Reset and wait for next frame if empty or corrupted
                self.has_new_pointcloud = False

        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    def transform_and_filter_cloud(self, pts_cam: np.ndarray, colors: np.ndarray,
                                   R_world_cam: np.ndarray, pos_cam: np.ndarray):
        """
        Transform local camera point cloud to world frame and strip out
        the ground plane and cylindrical base.
        """
        if len(pts_cam) == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

        # Check frame conventions: Gazebo camera_link (+X forward) vs ROS optical (+Z forward)
        # In Gazebo rgbd_camera sensor with camera_link frame_id, points are +X forward.
        median_x = np.median(pts_cam[:, 0])
        median_z = np.median(pts_cam[:, 2])

        if median_z > 0.4 and abs(median_x) < 0.3:
            # Standard optical frame convention: convert to camera_link
            # optical: X-right, Y-down, Z-forward -> link: X-forward, Y-left, Z-up
            pts_link = np.column_stack([pts_cam[:, 2], -pts_cam[:, 0], -pts_cam[:, 1]])
        else:
            # Already in camera_link (+X forward, +Y left, +Z up)
            pts_link = pts_cam

        # Transform to global world coordinates: P_world = R @ P_link + T
        pts_world = (R_world_cam @ pts_link.T).T + pos_cam

        # Filter 1: Strip out ground plane and cylindrical base (z <= z_floor_threshold)
        floor_mask = pts_world[:, 2] > self.z_floor_threshold

        # Filter 2: Strip out background points outside target radius (r > 0.22m)
        radius_mask = np.hypot(pts_world[:, 0], pts_world[:, 1]) < 0.22

        # Filter 3: Height cap (box top is at z ~ 0.14m, cap at 0.25m)
        height_mask = pts_world[:, 2] < 0.25

        valid = floor_mask & radius_mask & height_mask
        return pts_world[valid].astype(np.float32), colors[valid].astype(np.float32)

    def run_revolution(self):
        """Execute the complete 360-degree multi-view capture and fusion."""
        num_stops = int(round(360.0 / self.step_deg))
        cycle = 0

        while rclpy.ok():
            cycle += 1
            self.get_logger().info(f'=== Starting 360° Multi-View Scanning Cycle #{cycle} ({num_stops} stops) ===')

            all_pts_list = []
            all_colors_list = []
            capture_success = True

            for step_idx in range(num_stops):
                if not rclpy.ok():
                    break

                angle_deg = step_idx * self.step_deg
                pos, quat, R, angles = self.compute_camera_pose(angle_deg)

                set_ok = self.set_camera_pose(angle_deg)
                if not set_ok:
                    self.get_logger().warning(f'[{step_idx+1}/{num_stops}] Failed set_pose at {angle_deg:5.1f}°')

                # Non-blocking spin loop capture
                pts_cam, colors = self.capture_valid_cloud(timeout_sec=5.0)
                n_raw = len(pts_cam)

                if n_raw < 100:
                    self.get_logger().error(
                        f'[{step_idx+1}/{num_stops}] NULL/EMPTY cloud at {angle_deg:5.1f}°! (raw points: {n_raw})'
                    )
                    capture_success = False
                    continue

                # Transform to world coordinate frame and filter floor
                pts_world, colors_filtered = self.transform_and_filter_cloud(pts_cam, colors, R, pos)
                n_box = len(pts_world)

                self.get_logger().info(
                    f'[{step_idx+1:2d}/{num_stops}] Angle {angle_deg:5.1f}° | '
                    f'Cam ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | '
                    f'Raw Pts: {n_raw:6d} -> Box Pts: {n_box:5d}'
                )

                if n_box > 0:
                    all_pts_list.append(pts_world)
                    all_colors_list.append(colors_filtered)

            if len(all_pts_list) == 0:
                self.get_logger().error('No valid box points captured across entire 360° revolution!')
                if not self.loop:
                    break
                continue

            # Merge all stops
            merged_pts = np.vstack(all_pts_list)
            merged_colors = np.vstack(all_colors_list)
            self.get_logger().info(f'Total merged raw box points: {len(merged_pts):,}')

            # Statistical outlier removal and PLY export
            self.save_pointcloud(merged_pts, merged_colors, self.output_ply)

            if not self.loop:
                self.get_logger().info(' Multi-view capture & point cloud fusion complete!')
                break

    def save_pointcloud(self, points: np.ndarray, colors: np.ndarray, output_path: str):
        """Clean point cloud with statistical outlier removal and save as PLY."""
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

            # Statistical Outlier Removal
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            clean_pcd = pcd.select_by_index(ind)

            o3d.io.write_point_cloud(output_path, clean_pcd)
            # Also save to workspace root for convenience
            workspace_ply = os.path.join('/home/raghav/CMM_Automation', os.path.basename(output_path))
            if os.path.abspath(workspace_ply) != os.path.abspath(output_path):
                o3d.io.write_point_cloud(workspace_ply, clean_pcd)

            bbox = clean_pcd.get_axis_aligned_bounding_box()
            min_b = bbox.get_min_bound()
            max_b = bbox.get_max_bound()
            center = clean_pcd.get_center()

            self.get_logger().info(f'Saved filtered point cloud to: {output_path}')
            self.get_logger().info(f'  - Final point count: {len(clean_pcd.points):,}')
            self.get_logger().info(f'  - Centroid        : ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})m')
            self.get_logger().info(f'  - Bounding Box X  : [{min_b[0]:.4f}, {max_b[0]:.4f}] (span: {max_b[0]-min_b[0]:.4f}m)')
            self.get_logger().info(f'  - Bounding Box Y  : [{min_b[1]:.4f}, {max_b[1]:.4f}] (span: {max_b[1]-min_b[1]:.4f}m)')
            self.get_logger().info(f'  - Bounding Box Z  : [{min_b[2]:.4f}, {max_b[2]:.4f}] (span: {max_b[2]-min_b[2]:.4f}m)')
            return

        except ImportError:
            self.get_logger().warning('Open3D not available, falling back to direct binary PLY export.')

        # Direct binary PLY export fallback
        n = len(points)
        with open(output_path, 'wb') as f:
            header = (
                f"ply\n"
                f"format binary_little_endian 1.0\n"
                f"element vertex {n}\n"
                f"property float x\n"
                f"property float y\n"
                f"property float z\n"
                f"property uchar red\n"
                f"property uchar green\n"
                f"property uchar blue\n"
                f"end_header\n"
            )
            f.write(header.encode('ascii'))
            rgb_uint8 = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
            record = np.zeros(n, dtype=[
                ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
            ])
            record['x'] = points[:, 0]
            record['y'] = points[:, 1]
            record['z'] = points[:, 2]
            record['red'] = rgb_uint8[:, 0]
            record['green'] = rgb_uint8[:, 1]
            record['blue'] = rgb_uint8[:, 2]
            f.write(record.tobytes())
        self.get_logger().info(f'Saved raw point cloud to {output_path} ({n:,} points)')


def main(args=None):
    parser = argparse.ArgumentParser(description='Revolve RealSense camera 360° around origin & fuse point cloud.')
    parser.add_argument('--step-deg', type=float, default=20.0, help='Angle increment in degrees (default: 20.0)')
    parser.add_argument('--settle-sec', type=float, default=0.5, help='Settling time per stop in seconds (default: 0.5)')
    parser.add_argument('--radius', type=float, default=0.85, help='Orbit radius in meters (default: 0.85)')
    parser.add_argument('--height', type=float, default=0.45, help='Camera height in meters (default: 0.45)')
    parser.add_argument('--target-z', type=float, default=0.09, help='Z coordinate of target center (default: 0.09)')
    parser.add_argument('--z-floor', type=float, default=0.041, help='Z threshold to strip floor/base (default: 0.041)')
    parser.add_argument('--output', type=str, default='merged_box_cloud.ply', help='Output PLY file path')
    parser.add_argument('--world', type=str, default='cmm_world', help='Gazebo world name')
    parser.add_argument('--model-name', type=str, default='realsense_camera', help='Camera model name')
    parser.add_argument('--loop', action='store_true', help='Continuously loop 360° revolution')
    parsed_args, remaining_args = parser.parse_known_args()

    rclpy.init(args=remaining_args)
    revolver = CameraRevolver(
        step_deg=parsed_args.step_deg,
        settle_sec=parsed_args.settle_sec,
        radius=parsed_args.radius,
        height=parsed_args.height,
        target_z=parsed_args.target_z,
        z_floor_threshold=parsed_args.z_floor,
        output_ply=parsed_args.output,
        world_name=parsed_args.world,
        model_name=parsed_args.model_name,
        loop=parsed_args.loop
    )

    try:
        revolver.run_revolution()
    finally:
        revolver.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
