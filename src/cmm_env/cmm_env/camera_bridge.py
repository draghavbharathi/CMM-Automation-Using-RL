#!/usr/bin/env python3
"""
RealSense ROS 2 Bridge Node with TF Broadcaster.
Bridges Intel RealSense camera streams and simulation poses from Gazebo Jetty (gz-transport)
to standard ROS 2 topics:
  - PointCloud2:  /camera/points & /camera/depth/color/points
  - RGB Image:    /camera/color/image_raw
  - Depth Image:  /camera/depth/image_raw
  - Camera Info:  /camera/color/camera_info
  - Transforms:   /tf and /tf_static (world -> cylindrical_base, world -> camera_link,
                  camera_link -> camera_depth_frame -> camera_depth_optical_frame,
                  camera_link -> camera_color_frame -> camera_color_optical_frame)
"""

import sys
import os
import time
import numpy as np

# Ensure Gazebo python libraries are accessible
gz_py_path = '/home/raghav/workspace/install/lib/python'
if gz_py_path not in sys.path:
    sys.path.insert(0, gz_py_path)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time as RosTime
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

import gz.transport
from gz.msgs.pointcloud_packed_pb2 import PointCloudPacked as GzPointCloud
from gz.msgs.image_pb2 import Image as GzImage
from gz.msgs.camera_info_pb2 import CameraInfo as GzCameraInfo
from gz.msgs.pose_v_pb2 import Pose_V as GzPoseV


class RealSenseCameraBridge(Node):
    def __init__(self):
        super().__init__('realsense_camera_bridge')

        # Standard ROS 2 Publishers (Reliable QoS with queue size 10)
        self.pub_pc = self.create_publisher(PointCloud2, '/camera/points', 10)
        self.pub_pc_alias = self.create_publisher(PointCloud2, '/camera/depth/color/points', 10)
        self.pub_rgb = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.pub_depth = self.create_publisher(Image, '/camera/depth/image_raw', 10)
        self.pub_info = self.create_publisher(CameraInfo, '/camera/color/camera_info', 10)

        # TF Broadcasters
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # Frame poses
        # Cylindrical base at origin (height 4cm, radius 20cm, center at z=0.02)
        self.base_pos = [0.0, 0.0, 0.02]
        self.base_rot = [0.0, 0.0, 0.0, 1.0]

        # Camera link spawned 85cm behind (-0.85m) and 45cm above (0.45m), pitch = 0.401 rad
        self.cam_pos = [-0.85, 0.0, 0.45]
        self.cam_rot = [0.0, 0.199159, 0.0, 0.979967]

        # Publish initial static transforms
        self._publish_static_transforms()

        # Timer to broadcast dynamic transforms on /tf at 30 Hz
        self.tf_timer = self.create_timer(1.0 / 30.0, self._timer_broadcast_tf)

        # Gazebo Transport Node
        self.gz_node = gz.transport.Node()

        # Connect callbacks for camera streams
        self.gz_node.subscribe(GzPointCloud, '/camera/points', self._on_point_cloud)
        self.gz_node.subscribe(GzImage, '/camera/image', self._on_rgb_image)
        self.gz_node.subscribe(GzImage, '/camera/depth_image', self._on_depth_image)
        self.gz_node.subscribe(GzCameraInfo, '/camera/camera_info', self._on_camera_info)

        # Subscribe to Gazebo simulation poses for dynamic tracking of camera / base
        self.gz_node.subscribe(GzPoseV, '/world/cmm_world/dynamic_pose/info', self._on_pose_info)
        self.gz_node.subscribe(GzPoseV, '/world/cmm_world/pose/info', self._on_pose_info)

        # Statistics
        self.pc_count = 0
        self.rgb_count = 0
        self.last_log_time = time.time()

        self.get_logger().info('RealSense Camera Bridge & TF Broadcaster successfully started!')
        self.get_logger().info('Subscribed to Gazebo Jetty topics: /camera/points, /camera/image, /camera/depth_image')
        self.get_logger().info('Publishing to ROS 2 topics: /camera/points, /camera/depth/color/points, /camera/color/image_raw, /camera/depth/image_raw')
        self.get_logger().info('Publishing TF tree on /tf and /tf_static: world -> cylindrical_base, world -> camera_link -> optical frames')

    def _convert_header(self, gz_stamp, default_frame: str):
        header_time = RosTime()
        # Always use current ROS clock time to keep sensor data and TF in perfect synchronization
        now = self.get_clock().now().to_msg()
        header_time.sec = now.sec
        header_time.nanosec = now.nanosec
        return header_time

    def _on_pose_info(self, gz_msg: GzPoseV):
        """Update dynamic poses of camera and cylindrical base from Gazebo. Note: Analyzed objects are excluded."""
        for p in gz_msg.pose:
            name = p.name
            if name == 'realsense_camera':
                self.cam_pos = [p.position.x, p.position.y, p.position.z]
                self.cam_rot = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
            elif name == 'cylindrical_base':
                self.base_pos = [p.position.x, p.position.y, p.position.z]
                self.base_rot = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]

    def _create_static_transforms(self, stamp):
        """Build optical and intermediate transforms from camera_link."""
        transforms = []

        # camera_link -> camera_depth_frame
        t_depth = TransformStamped()
        t_depth.header.stamp = stamp
        t_depth.header.frame_id = 'camera_link'
        t_depth.child_frame_id = 'camera_depth_frame'
        t_depth.transform.translation.x = 0.0
        t_depth.transform.translation.y = 0.0
        t_depth.transform.translation.z = 0.0
        t_depth.transform.rotation.x = 0.0
        t_depth.transform.rotation.y = 0.0
        t_depth.transform.rotation.z = 0.0
        t_depth.transform.rotation.w = 1.0
        transforms.append(t_depth)

        # camera_depth_frame -> camera_depth_optical_frame
        # Standard ROS optical frame rotation: qx=-0.5, qy=0.5, qz=-0.5, qw=0.5
        t_depth_opt = TransformStamped()
        t_depth_opt.header.stamp = stamp
        t_depth_opt.header.frame_id = 'camera_depth_frame'
        t_depth_opt.child_frame_id = 'camera_depth_optical_frame'
        t_depth_opt.transform.translation.x = 0.0
        t_depth_opt.transform.translation.y = 0.0
        t_depth_opt.transform.translation.z = 0.0
        t_depth_opt.transform.rotation.x = -0.5
        t_depth_opt.transform.rotation.y = 0.5
        t_depth_opt.transform.rotation.z = -0.5
        t_depth_opt.transform.rotation.w = 0.5
        transforms.append(t_depth_opt)

        # camera_link -> camera_color_frame (15mm offset along Y for D435 RGB sensor)
        t_color = TransformStamped()
        t_color.header.stamp = stamp
        t_color.header.frame_id = 'camera_link'
        t_color.child_frame_id = 'camera_color_frame'
        t_color.transform.translation.x = 0.0
        t_color.transform.translation.y = 0.015
        t_color.transform.translation.z = 0.0
        t_color.transform.rotation.x = 0.0
        t_color.transform.rotation.y = 0.0
        t_color.transform.rotation.z = 0.0
        t_color.transform.rotation.w = 1.0
        transforms.append(t_color)

        # camera_color_frame -> camera_color_optical_frame
        t_color_opt = TransformStamped()
        t_color_opt.header.stamp = stamp
        t_color_opt.header.frame_id = 'camera_color_frame'
        t_color_opt.child_frame_id = 'camera_color_optical_frame'
        t_color_opt.transform.translation.x = 0.0
        t_color_opt.transform.translation.y = 0.0
        t_color_opt.transform.translation.z = 0.0
        t_color_opt.transform.rotation.x = -0.5
        t_color_opt.transform.rotation.y = 0.5
        t_color_opt.transform.rotation.z = -0.5
        t_color_opt.transform.rotation.w = 0.5
        transforms.append(t_color_opt)

        return transforms

    def _publish_static_transforms(self):
        stamp = self.get_clock().now().to_msg()
        static_transforms = self._create_static_transforms(stamp)
        self.static_tf_broadcaster.sendTransform(static_transforms)

    def _broadcast_tf(self, stamp=None):
        """Broadcast complete TF tree for cylindrical_base and camera."""
        if stamp is None:
            stamp = self.get_clock().now().to_msg()

        transforms = []

        # 1. world -> cylindrical_base
        t_base = TransformStamped()
        t_base.header.stamp = stamp
        t_base.header.frame_id = 'world'
        t_base.child_frame_id = 'cylindrical_base'
        t_base.transform.translation.x = float(self.base_pos[0])
        t_base.transform.translation.y = float(self.base_pos[1])
        t_base.transform.translation.z = float(self.base_pos[2])
        t_base.transform.rotation.x = float(self.base_rot[0])
        t_base.transform.rotation.y = float(self.base_rot[1])
        t_base.transform.rotation.z = float(self.base_rot[2])
        t_base.transform.rotation.w = float(self.base_rot[3])
        transforms.append(t_base)

        # 2. world -> camera_link (dynamically updated when camera revolves)
        t_cam = TransformStamped()
        t_cam.header.stamp = stamp
        t_cam.header.frame_id = 'world'
        t_cam.child_frame_id = 'camera_link'
        t_cam.transform.translation.x = float(self.cam_pos[0])
        t_cam.transform.translation.y = float(self.cam_pos[1])
        t_cam.transform.translation.z = float(self.cam_pos[2])
        t_cam.transform.rotation.x = float(self.cam_rot[0])
        t_cam.transform.rotation.y = float(self.cam_rot[1])
        t_cam.transform.rotation.z = float(self.cam_rot[2])
        t_cam.transform.rotation.w = float(self.cam_rot[3])
        transforms.append(t_cam)

        # 3. Optical frames also broadcast on /tf for seamless sync
        transforms.extend(self._create_static_transforms(stamp))

        # Broadcast on /tf
        if not rclpy.ok():
            return
        try:
            self.tf_broadcaster.sendTransform(transforms)
        except Exception:
            pass

    def _timer_broadcast_tf(self):
        if rclpy.ok():
            self._broadcast_tf()

    def _on_point_cloud(self, gz_msg: GzPointCloud):
        try:
            now_stamp = self.get_clock().now().to_msg()
            self._broadcast_tf(now_stamp)

            ros_msg = PointCloud2()
            ros_msg.header.stamp = now_stamp
            # Gazebo rgbd_camera sensor outputs points in the camera body link frame
            # (+X forward, +Y left, +Z up). Setting frame_id to camera_link aligns
            # the point cloud perfectly with the world and cylindrical base.
            ros_msg.header.frame_id = 'camera_link'

            ros_msg.height = gz_msg.height
            ros_msg.width = gz_msg.width
            ros_msg.is_bigendian = gz_msg.is_bigendian
            ros_msg.point_step = gz_msg.point_step
            ros_msg.row_step = gz_msg.row_step if gz_msg.row_step > 0 else gz_msg.width * gz_msg.point_step
            ros_msg.is_dense = gz_msg.is_dense

            # Gazebo gz.msgs.PointCloudPacked.Field.DataType is 0-indexed (FLOAT32=6),
            # while ROS 2 sensor_msgs.msg.PointField is 1-indexed (FLOAT32=7).
            # RViz XYZ transformer requires PointField.FLOAT32 (7) for x, y, z fields.
            GZ_TO_ROS_DATATYPE = {
                0: PointField.INT8,
                1: PointField.UINT8,
                2: PointField.INT16,
                3: PointField.UINT16,
                4: PointField.INT32,
                5: PointField.UINT32,
                6: PointField.FLOAT32,
                7: PointField.FLOAT64,
            }

            # Fields mapping
            for f in gz_msg.field:
                pf = PointField()
                pf.name = f.name
                pf.offset = f.offset
                pf.datatype = GZ_TO_ROS_DATATYPE.get(f.datatype, PointField.FLOAT32)
                pf.count = f.count if f.count > 0 else 1
                ros_msg.fields.append(pf)

            # Ensure alpha channel is 255 (fully opaque) for RViz RGB8 color transformer.
            # Gazebo outputs 16 bytes per point: x(4), y(4), z(4), rgb(4).
            # The 4th byte of rgb (byte offset 15, 31, ...) is 0 (transparent) in Gazebo.
            # Setting it to 255 allows RViz RGB8 to render vivid opaque colors.
            raw_data = bytearray(gz_msg.data)
            if gz_msg.point_step == 16 and len(raw_data) >= 16:
                arr = np.frombuffer(raw_data, dtype=np.uint8)
                arr[15::16] = 255
                ros_msg.data = arr.tobytes()
            else:
                ros_msg.data = bytes(gz_msg.data)

            self.pub_pc.publish(ros_msg)
            self.pub_pc_alias.publish(ros_msg)

            self.pc_count += 1
            if self.pc_count == 1:
                self.get_logger().info(
                    f"Point Cloud Active! Publishing to ROS 2 [/camera/points]: {ros_msg.width}x{ros_msg.height} ({ros_msg.width * ros_msg.height} points, {len(ros_msg.data)} bytes)"
                )

            now = time.time()
            if now - self.last_log_time >= 5.0:
                fps = self.pc_count / (now - self.last_log_time)
                self.get_logger().info(f'Streaming PointCloud2: {ros_msg.width}x{ros_msg.height} pts @ {fps:.1f} FPS')
                self.pc_count = 0
                self.last_log_time = now
        except Exception as e:
            self.get_logger().error(f'Error processing point cloud: {e}')

    def _on_rgb_image(self, gz_msg: GzImage):
        try:
            now_stamp = self.get_clock().now().to_msg()
            ros_msg = Image()
            ros_msg.header.stamp = now_stamp
            ros_msg.header.frame_id = 'camera_color_optical_frame'
            ros_msg.height = gz_msg.height
            ros_msg.width = gz_msg.width
            ros_msg.encoding = 'rgb8'
            ros_msg.step = gz_msg.step if gz_msg.step > 0 else gz_msg.width * 3
            ros_msg.data = bytes(gz_msg.data)
            self.pub_rgb.publish(ros_msg)
        except Exception as e:
            self.get_logger().error(f'Error processing RGB image: {e}')

    def _on_depth_image(self, gz_msg: GzImage):
        try:
            now_stamp = self.get_clock().now().to_msg()
            ros_msg = Image()
            ros_msg.header.stamp = now_stamp
            ros_msg.header.frame_id = 'camera_depth_optical_frame'
            ros_msg.height = gz_msg.height
            ros_msg.width = gz_msg.width
            ros_msg.encoding = '32FC1'
            ros_msg.step = gz_msg.step if gz_msg.step > 0 else gz_msg.width * 4
            ros_msg.data = bytes(gz_msg.data)
            self.pub_depth.publish(ros_msg)
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {e}')

    def _on_camera_info(self, gz_msg: GzCameraInfo):
        try:
            now_stamp = self.get_clock().now().to_msg()
            ros_msg = CameraInfo()
            ros_msg.header.stamp = now_stamp
            ros_msg.header.frame_id = 'camera_color_optical_frame'
            ros_msg.width = gz_msg.width
            ros_msg.height = gz_msg.height

            # Intrinsics
            if gz_msg.intrinsics:
                ros_msg.k = [float(x) for x in gz_msg.intrinsics.k]
            if gz_msg.projection:
                ros_msg.p = [float(x) for x in gz_msg.projection.p]

            if not rclpy.ok():
                return
            self.pub_info.publish(ros_msg)
        except Exception as e:
            if rclpy.ok():
                self.get_logger().error(f'Error processing camera info: {e}')

    def stop(self):
        """Unsubscribe from Gazebo transport before node destruction."""
        try:
            self.gz_node.unsubscribe('/camera/points')
            self.gz_node.unsubscribe('/camera/image')
            self.gz_node.unsubscribe('/camera/depth_image')
            self.gz_node.unsubscribe('/camera/camera_info')
            self.gz_node.unsubscribe('/world/cmm_world/dynamic_pose/info')
            self.gz_node.unsubscribe('/world/cmm_world/pose/info')
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    bridge_node = RealSenseCameraBridge()
    try:
        while rclpy.ok():
            rclpy.spin_once(bridge_node, timeout_sec=0.01)
            time.sleep(0.002)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        bridge_node.stop()
        bridge_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
