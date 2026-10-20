#!/usr/bin/env python3
"""
Launch file for CMM Simulation in Gazebo Jetty with RealSense Point Cloud Bridge.
Launches:
- Gazebo simulation with CMM world (ground, lighting, cylindrical base, RealSense camera)
- RealSense camera bridge publishing PointCloud2 to ROS 2
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError


def generate_launch_description():
    try:
        pkg_share = get_package_share_directory('cmm_env')
        world_path = os.path.join(pkg_share, 'worlds', 'cmm_world.sdf')
        models_path = os.path.join(pkg_share, 'models')
    except PackageNotFoundError:
        world_path = '/home/raghav/CMM_Automation/src/cmm_env/worlds/cmm_world.sdf'
        models_path = '/home/raghav/CMM_Automation/src/cmm_env/models'

    world_arg = DeclareLaunchArgument(
        'world',
        default_value=world_path,
        description='Path to the SDF world file'
    )

    # Set up environment for Gazebo resource discovery and python bindings
    current_resource_path = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    gz_resource_path = f"{models_path}:{current_resource_path}" if current_resource_path else models_path

    env = {
        'GZ_SIM_RESOURCE_PATH': gz_resource_path
    }

    # Gazebo sim command
    gz_sim = ExecuteProcess(
        cmd=['gz', 'sim', '-r', LaunchConfiguration('world')],
        additional_env=env,
        output='screen'
    )

    # RealSense ROS 2 Bridge Node
    camera_bridge = Node(
        package='cmm_env',
        executable='camera_bridge',
        name='realsense_camera_bridge',
        output='screen'
    )

    return LaunchDescription([
        world_arg,
        gz_sim,
        camera_bridge
    ])
