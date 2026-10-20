#!/usr/bin/env python3
"""
ROS 2 Python script to spawn the 10cm x 10cm 6-colored measured box
onto the cylindrical base in Gazebo Jetty.
"""

import sys
import os
import argparse
import subprocess
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError


class BoxSpawnerNode(Node):
    def __init__(self):
        super().__init__('box_spawner_node')
        self.get_logger().info('Initializing CMM Box Spawner Node...')

    def spawn(self, x: float, y: float, z: float, world_name: str = 'cmm_world', model_name: str = 'measured_box'):
        """Spawns the measured box model into the running Gazebo world via UserCommands service."""
        # 1. Locate the measured_box model SDF
        model_path = None
        try:
            share_dir = get_package_share_directory('cmm_env')
            candidate = os.path.join(share_dir, 'models', 'measured_box', 'model.sdf')
            if os.path.isfile(candidate):
                model_path = candidate
        except PackageNotFoundError:
            pass

        if not model_path:
            # Fallback to local workspace source path
            workspace_candidate = '/home/raghav/CMM_Automation/src/cmm_env/models/measured_box/model.sdf'
            if os.path.isfile(workspace_candidate):
                model_path = workspace_candidate

        if not model_path or not os.path.isfile(model_path):
            self.get_logger().error(f'Could not find measured_box model.sdf at {model_path}')
            return False

        self.get_logger().info(f'Using model SDF: {model_path}')

        # 2. Check if Gazebo world service is available
        service_name = f'/world/{world_name}/create'
        check_cmd = ['gz', 'service', '-l']
        try:
            check_proc = subprocess.run(check_cmd, capture_output=True, text=True, timeout=5)
            if service_name not in check_proc.stdout:
                self.get_logger().warning(
                    f"Service '{service_name}' not found yet. Is Gazebo running with world '{world_name}'?\n"
                    f"You can launch Gazebo with: gz sim <path_to_cmm_world.sdf> or ros2 launch cmm_env cmm_sim.launch.py"
                )
        except Exception as e:
            self.get_logger().warning(f'Could not query gz services: {e}')

        # 3. Construct EntityFactory request
        req_content = (
            f'sdf_filename: "{model_path}", '
            f'name: "{model_name}", '
            f'pose: {{ '
            f'position: {{ x: {x:.4f}, y: {y:.4f}, z: {z:.4f} }}, '
            f'orientation: {{ x: 0.0, y: 0.0, z: 0.0, w: 1.0 }} '
            f'}}'
        )

        spawn_cmd = [
            'gz', 'service',
            '-s', service_name,
            '--reqtype', 'gz.msgs.EntityFactory',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '5000',
            '--req', req_content
        ]

        self.get_logger().info(
            f"Spawning '{model_name}' at position (x={x:.4f}m, y={y:.4f}m, z={z:.4f}m) into world '{world_name}'..."
        )

        try:
            proc = subprocess.run(spawn_cmd, capture_output=True, text=True, timeout=10)
            if proc.returncode == 0 and 'data: true' in proc.stdout:
                self.get_logger().info(
                    f" Successfully spawned '{model_name}' at (x={x}, y={y}, z={z}) on top of cylindrical base!"
                )
                self.get_logger().info("Box Faces & Colors:")
                self.get_logger().info("  - Front (+X): Red")
                self.get_logger().info("  - Back  (-X): Green")
                self.get_logger().info("  - Right (+Y): Blue")
                self.get_logger().info("  - Left  (-Y): Yellow")
                self.get_logger().info("  - Top   (+Z): Cyan")
                self.get_logger().info("  - Bottom(-Z): Magenta")
                return True
            else:
                self.get_logger().error(f"Failed to spawn model. gz service output:\n{proc.stdout}\n{proc.stderr}")
                return False
        except subprocess.TimeoutExpired:
            self.get_logger().error("gz service call timed out after 10 seconds. Check if Gazebo is running.")
            return False
        except Exception as e:
            self.get_logger().error(f'Error executing spawn command: {e}')
            return False


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(description='Spawn 10cm measured box on cylindrical base in Gazebo')
    parser.add_argument('--x', type=float, default=0.0, help='X coordinate in meters (default: 0.0)')
    parser.add_argument('--y', type=float, default=0.0, help='Y coordinate in meters (default: 0.0)')
    parser.add_argument(
        '--z',
        type=float,
        default=0.045,
        help='Z coordinate in meters (default: 0.045m / 4.5cm, resting on the 4cm base)'
    )
    parser.add_argument('--world', type=str, default='cmm_world', help='World name (default: cmm_world)')
    parser.add_argument('--model-name', type=str, default='measured_box', help='Spawned model name')
    parser.add_argument(
        '--force-meters',
        action='store_true',
        help='Prevent auto-scaling when z > 1.0 (interpret value literally as meters)'
    )

    parsed_args, _ = parser.parse_known_args()

    x = parsed_args.x
    y = parsed_args.y
    z = parsed_args.z

    # If user passed z=4.5 without --force-meters, 4.5cm corresponds to 0.045m (on top of 4cm base)
    if z > 1.0 and not parsed_args.force_meters:
        print(f"[INFO] Notice: z={z} is greater than 1.0m. Interpreting as {z} cm ({z/100.0:.4f} m) on top of the 4cm base.")
        print("[INFO] (Use --force-meters if you intended literal meters in the sky).")
        z = z / 100.0

    node = BoxSpawnerNode()
    success = node.spawn(x, y, z, world_name=parsed_args.world, model_name=parsed_args.model_name)
    node.destroy_node()
    rclpy.shutdown()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
