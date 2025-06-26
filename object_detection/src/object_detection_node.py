#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
from sensor_msgs_py import point_cloud2
import time
import cv2
import csv
import os
import signal
import sys
from os.path import join
from numpy.lib.recfunctions import unstructured_to_structured
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, PointField
from geometry_msgs.msg import PoseArray, Pose, Quaternion

from object_detection_msgs.msg import (
    PointCloudArray,
    ObjectDetectionInfo,
    ObjectDetectionInfoArray,
)
from std_msgs.msg import Header

from object_detection.objectdetectorONNX import ObjectDetectorONNX
from object_detection.pointprojector import PointProjector
from object_detection.objectlocalizer import ObjectLocalizer
from object_detection.utils import *

# from object_detection.ros_numpy import *

from ament_index_python.packages import get_package_share_directory


class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__("object_detection_node")

        self.get_logger().info(
            "[ObjectDetection Node] Object Detector initilization starts ..."
        )

        # ---------- Initialize CSV data storage ----------
        self.csv_data = []
        self.csv_filename = "object_detection_results.csv"
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        # ---------- Initialize parameters ----------
        self.declare_parameters(
            namespace="",
            parameters=[
                ("verbose", True),
                ("project_object_points_to_image", True),
                ("project_all_points_to_image", False),
                ("camera_topic", "/rgb_camera/undistorted"),
                ("camera_info_topic", "/rgb_camera/camera_info"),
                ("lidar_topic", "/rslidar/points"),
                ("object_detection_pose_topic", "object_poses"),
                ("object_detection_output_image_topic", "detections_in_image"),
                ("object_detection_point_clouds_topic", "detection_point_clouds"),
                ("object_detection_info_topic", "detection_info"),
                ("camera_lidar_sync_queue_size", 10),
                ("camera_lidar_sync_slop", 0.05),
                ("project_config", "projector_config.yaml"),
                ("architecture", "yolo"),
                ("model", "yolov5n6"),
                ("model_dir_path", ""),
                ("device", "cpu"),
                ("confident", 0.4),
                ("iou", 0.1),
                ("model_method", "hdbscan"),
                ("ground_percentage", 25),
                ("bb_contract_percentage", 10),
                ("distance_estimator_type", "none"),
                ("distance_estimator_save_data", False),
                ("object_specific_file", "object_specific.yaml"),
                ("min_cluster_size", 5),
                ("cluster_selection_epsilon", 0.08),
                ("multiple_instance", False),
                ("classes", "[56]"), # only chair by default
                ("csv_filename", "object_detection_results.csv"),
            ],
        )

        # Get CSV filename from parameters
        self.csv_filename = self.get_parameter("csv_filename").value

        # ---------- Setup publishers ----------
        self.object_pose_pub = self.create_publisher(
            PoseArray, self.get_parameter("object_detection_pose_topic").value, 10
        )

        self.object_detection_img_pub = self.create_publisher(
            Image, self.get_parameter("object_detection_output_image_topic").value, 10
        )

        self.object_point_clouds_pub = self.create_publisher(
            PointCloudArray,
            self.get_parameter("object_detection_point_clouds_topic").value,
            10,
        )

        self.detection_info_pub = self.create_publisher(
            ObjectDetectionInfoArray,
            self.get_parameter("object_detection_info_topic").value,
            10,
        )

        # ---------- Setup subscribers ----------
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.camera_sub = Subscriber(
            self, Image, self.get_parameter("camera_topic").value
        )
        self.lidar_sub = Subscriber(
            self,
            PointCloud2,
            self.get_parameter("lidar_topic").value,
            qos_profile=qos_profile,
        )

        # ---------- Setup synchronizer ----------
        self.synchronizer = ApproximateTimeSynchronizer(
            [self.camera_sub, self.lidar_sub],
            queue_size=self.get_parameter("camera_lidar_sync_queue_size").value,
            slop=self.get_parameter("camera_lidar_sync_slop").value,
        )
        self.synchronizer.registerCallback(self.sync_callback)

        # ---------- Config Directory ----------
        self.config_dir = join(get_package_share_directory("object_detection"), "cfg")

        # ---------- Setup PointProjector ----------
        project_cfg = self.get_parameter("project_config").value
        self.point_projector = PointProjector(self, join(self.config_dir, project_cfg))

        # ---------- Setup 2D Object Detection ----------
        self.object_detector = ObjectDetectorONNX(
            {
                "architecture": self.get_parameter("architecture").value,
                "model": self.get_parameter("model").value,
                "model_dir_path": self.get_parameter("model_dir_path").value,
                "device": self.get_parameter("device").value,
                "confident": self.get_parameter("confident").value,
                "iou": self.get_parameter("iou").value,
                "checkpoint": None,
                # "classes": self.get_parameter("classes").value,
                "classes": [11,24,25,39,74],
                "multiple_instance": self.get_parameter("multiple_instance").value,
            },
        )

        # ---------- Setup 3D Object Localizer ----------
        self.object_localizer = ObjectLocalizer(
            self,
            {
                "model_method": self.get_parameter("model_method").value,
                "ground_percentage": self.get_parameter("ground_percentage").value,
                "bb_contract_percentage": self.get_parameter(
                    "bb_contract_percentage"
                ).value,
                "distance_estimator_type": self.get_parameter(
                    "distance_estimator_type"
                ).value,
                "distance_estimator_save_data": self.get_parameter(
                    "distance_estimator_save_data"
                ).value,
                "object_specific_file": self.get_parameter(
                    "object_specific_file"
                ).value,
                "min_cluster_size": self.get_parameter("min_cluster_size").value,
                "cluster_selection_epsilon": self.get_parameter(
                    "cluster_selection_epsilon"
                ).value,
            },
            self.config_dir,
        )

        # ---------- Initialize components ----------
        self.image_reader = CvBridge()
        self.image_info_received = False

        self.get_logger().info(
            "[ObjectDetection Node] Object Detector initilization done."
        )
        self.get_logger().info("[ObjectDetection Node] Waiting for image info ...")
        self.get_logger().info(
            f"[ObjectDetection Node] If this takes longer than a few seconds, make sure {self.get_parameter('camera_info_topic').value} is published."
        )

        # ---------- Check camera info ----------
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.get_parameter("camera_info_topic").value,
            self.image_info_callback,
            10,
        )

    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.get_logger().info(f"Received signal {signum}, shutting down gracefully...")
        self.save_csv_data()
        sys.exit(0)

    def save_csv_data(self):
        """Save collected detection data to CSV file"""
        try:
            if not self.csv_data:
                self.get_logger().info("No detection data to save")
                return
                
            with open(self.csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow([
                    'timestamp', 'object_id', 'class_id', 'confidence',
                    'pos_x', 'pos_y', 'pos_z', 'estimation_type',
                    'bbox_min_x', 'bbox_min_y', 'bbox_max_x', 'bbox_max_y'
                ])
                
                # Write data
                writer.writerows(self.csv_data)
                
            self.get_logger().info(f"Successfully saved {len(self.csv_data)} detection records to {self.csv_filename}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save CSV file: {str(e)}")

    def add_detection_to_csv(self, timestamp, obj_id, class_id, confidence, pos, estimation_type, bbox):
        """Add detection data to CSV storage"""
        self.csv_data.append([
            timestamp, obj_id, class_id, confidence,
            pos[0], pos[1], pos[2], estimation_type,
            bbox[0], bbox[1], bbox[2], bbox[3]
        ])

    def image_info_callback(self, msg):
        """Handle camera info message"""
        h = msg.height
        w = msg.width
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

        if check_validity_image_info(K, w, h):
            self.point_projector.set_intrinsic_params(K, [w, h])
            self.object_localizer.set_intrinsic_camera_param(K)
            self.optical_frame_id = msg.header.frame_id
            self.get_logger().info(
                "[ObjectDetection Node] Image info is set! Detection will start..."
            )
            self.image_info_received = True
        else:
            self.get_logger().error(
                " ------------------ camera_info not valid ------------------------"
            )

    def sync_callback(self, image_msg, lidar_msg):
        """Synchronized callback for image and point cloud"""
        if not self.image_info_received:
            self.get_logger().warn("Waiting for camera info...", once=True)
            return

        start_time = time.time()
        self.get_logger().info(
            "Got first image / pointcloud pair",
            once=True,
        )

        # If Image and Lidar messages are not empty
        if not image_msg.height > 0:
            self.get_logger().fatal(
                "[ObjectDetection Node] Image message is empty. Object detecion is on hold."
            )
            return
        if not lidar_msg.width > 0:
            self.get_logger().fatal(
                "[ObjectDetection Node] Lidar message is empty. Object detecion is on hold."
            )
            return

        try:
            # Read message
            cv_image = self.image_reader.imgmsg_to_cv2(image_msg, "bgr8")

            point_cloud_xyz = pointcloud2_to_xyz_array(lidar_msg)
            # Validate point cloud data
            if point_cloud_xyz is None or point_cloud_xyz.shape[0] == 0:
                self.get_logger().warn("Empty point cloud received")
                return

            # Ground filter
            # Upward direction is Z which 3rd column in the matrix
            # It is positive because it increases upwards
            point_cloud_xyz = filter_ground(
                point_cloud_xyz, self.get_parameter("ground_percentage").value
            )

            # Transform points
            transformed_points = self.point_projector.transform_points(
                point_cloud_xyz[:, :3]
            )
            point_cloud_xyz[:, :3] = transformed_points

            # Project points and validate results
            points_on_image, in_fov_indices = (
                self.point_projector.project_points_on_image(point_cloud_xyz[:, :3])
            )
            if len(in_fov_indices) == 0:
                self.get_logger().debug("No points projected within image frame")
                return

            # Get points in field of view
            pointcloud_in_fov = point_cloud_xyz[in_fov_indices]

            # Detect objects
            object_detection_result, object_detection_image = (
                self.object_detector.detect(cv_image)
            )
            if (
                object_detection_result is None
                or len(object_detection_result.get("name", [])) == 0
            ):
                self.get_logger().debug("No objects detected")
                return

            # Localize objects
            object_list = self.object_localizer.localize(
                object_detection_result,
                points_on_image,
                point_cloud_xyz[in_fov_indices],
                cv_image,
            )
            self.get_logger().info(f"Number of detected objects: {len(object_list)}")

            # Create and publish results
            header = Header()
            header.stamp = image_msg.header.stamp
            header.frame_id = self.optical_frame_id
            
            # Convert timestamp to string for CSV
            timestamp_str = f"{header.stamp.sec}.{header.stamp.nanosec:09d}"

            object_pose_array = PoseArray(header=header)
            object_info_array = ObjectDetectionInfoArray(header=header)
            point_cloud_array = PointCloudArray(header=header)

            # Populate messages
            for i, obj in enumerate(object_list):
                # Create pose
                object_pose = Pose()
                object_pose.position.x = float(obj.pos[0])
                object_pose.position.y = float(obj.pos[1])
                object_pose.position.z = float(obj.pos[2])
                object_pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                object_pose_array.poses.append(object_pose)

                # Create detection info
                object_information = ObjectDetectionInfo()
                object_information.class_id = str(
                    object_detection_result["name"][i]
                )  # Ensure string
                object_information.id = int(obj.id)  # Ensure integer

                object_information.position.x = float(obj.pos[0])
                object_information.position.y = float(obj.pos[1])
                object_information.position.z = float(obj.pos[2])
                object_information.pose_estimation_type = str(obj.estimation_type)
                object_information.confidence = float(
                    object_detection_result["confidence"][i]
                )
                object_information.bounding_box_min_x = int(
                    object_detection_result["xmin"][i]
                )
                object_information.bounding_box_min_y = int(
                    object_detection_result["ymin"][i]
                )
                object_information.bounding_box_max_x = int(
                    object_detection_result["xmax"][i]
                )
                object_information.bounding_box_max_y = int(
                    object_detection_result["ymax"][i]
                )
                object_info_array.info.append(object_information)

                # Add detection data to CSV storage
                self.add_detection_to_csv(
                    timestamp_str,
                    obj.id,
                    object_detection_result["name"][i],
                    object_detection_result["confidence"][i],
                    obj.pos,
                    obj.estimation_type,
                    [
                        object_detection_result["xmin"][i],
                        object_detection_result["ymin"][i],
                        object_detection_result["xmax"][i],
                        object_detection_result["ymax"][i]
                    ]
                )

                # Create point cloud
                object_point_cloud = pointcloud_in_fov[obj.pt_indices]
                # Debug logging to understand the shape
                self.get_logger().info(
                    f"Original object_point_cloud shape: {object_point_cloud.shape}"
                )
                # Convert to PointCloud2 message (utils)
                point_cloud_msg = array_to_pointcloud2(
                    object_point_cloud,
                    frame_id=self.optical_frame_id,
                    stamp=image_msg.header.stamp,
                )
                point_cloud_array.point_clouds.append(point_cloud_msg)

                # Visualize if enabled
                if (
                    not self.get_parameter("project_all_points_to_image").value
                    and self.get_parameter("project_object_points_to_image").value
                ):
                    object_points = points_on_image[obj.pt_indices]

                    if len(object_points.shape) == 1:
                        # Calculate number of points (total length must be even)
                        n_points = len(object_points) // 2
                        # Reshape to (n_points, 2) array
                        object_points = object_points.reshape(n_points, 2)
                    elif object_points.shape[1] != 2:
                        # If 2D but wrong shape, try to fix it
                        object_points = object_points.reshape(-1, 2)

                    for idx, pt in enumerate(object_points):
                        try:
                            dist = object_point_cloud[idx, 2]
                            color = depth_color(dist, min_d=0.5, max_d=20)
                            # Make a copy of the image before drawing
                            object_detection_image = object_detection_image.copy()

                            cv2.circle(
                                object_detection_image,
                                pt[:2].astype(np.int32),
                                2,
                                color,
                                -1,
                            )
                        except Exception as e:
                            self.get_logger().warn(f"Could not draw circle: {str(e)}")

            # Publish all points if enabled
            if self.get_parameter("project_all_points_to_image").value:
                for idx, pt in enumerate(points_on_image):
                    dist = pointcloud_in_fov[idx, 2]
                    color = depth_color(dist, min_d=0.5, max_d=30)
                    try:
                        cv2.circle(
                            object_detection_image,
                            pt[:2].astype(np.int32),
                            3,
                            color,
                            -1,
                        )
                    except Exception as e:
                        self.get_logger().warn(f"Could not draw circle: {str(e)}")
            # Publish results
            self.object_pose_pub.publish(object_pose_array)
            self.detection_info_pub.publish(object_info_array)
            self.object_point_clouds_pub.publish(point_cloud_array)
            self.object_detection_img_pub.publish(
                self.image_reader.cv2_to_imgmsg(object_detection_image, "bgr8")
            )

            self.get_logger().debug(f"Processing time: {time.time()-start_time:.3f}s")

        except Exception as e:
            self.get_logger().error(f"Error in sync_callback: {str(e)}")

    def __del__(self):
        """Destructor to ensure CSV is saved when object is destroyed"""
        self.save_csv_data()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = ObjectDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info("Keyboard interrupt received, shutting down...")
            node.save_csv_data()
    except Exception as e:
        if node:
            node.get_logger().fatal(f"Fatal error: {str(e)}")
            node.save_csv_data()
    finally:
        # Clean shutdown sequence
        if node:
            try:
                node.save_csv_data()  # Final attempt to save data
                node.destroy_node()
            except Exception as e:
                print(f"Error during node destruction: {e}")
        
        # Only call shutdown if it hasn't been called already
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            print(f"Error during rclpy shutdown: {e}")


if __name__ == "__main__":
    main()