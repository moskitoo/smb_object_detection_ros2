#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import csv
import os
from datetime import datetime
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs
from rclpy.time import Time
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import PointStamped, TransformStamped

from object_detection_msgs.msg import (
    ObjectDetectionInfo,
    ObjectDetectionInfoArray,
)

class DetectionProcessorNode(Node):
    def __init__(self):
        super().__init__("detection_processor_node")

        self.get_logger().info(
            "[Detection Processor Node] Initialization starts ..."
        )

        # ---------- Initialize TF ----------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # self.static_broadcaster = StaticTransformBroadcaster(self)
        # self.publish_camera_correction_transform()

                # ---------- Initialize parameters ----------
        self.declare_parameters(
            namespace="",
            parameters=[
                ("detection_info_topic", "detection_info"),
                ("target_classes", ["chair", "person"]),
                ("duplicate_distance_threshold", 0.0),
                ("csv_output_dir", "detections"),  # Relative path from current working directory
                ("marker_topic", "/detection_markers"),
                ("marker_lifetime", 30.0),  # seconds
                ("target_frame", "map"),  # Frame to transform to
                ("tf_timeout", 1.0),  # TF lookup timeout in seconds
            ],
        )

        # ---------- Initialize detection tracking ----------
        self.previous_detections = {}  # Dictionary to track previous detections by class
        self.detections_data = []  # List to store all detections for CSV

        # ---------- Initialize CSV logging ----------
        # Use relative path from current working directory
        self.csv_output_dir = self.get_parameter("csv_output_dir").value
        
        # Create directory if it doesn't exist
        os.makedirs(self.csv_output_dir, exist_ok=True)
        
        # Generate CSV filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = os.path.join(self.csv_output_dir, f"processed_detections_{timestamp}.csv")
        
        # Log the absolute path for clarity
        absolute_path = os.path.abspath(self.csv_filename)
        self.get_logger().info(f"[Detection Processor Node] CSV logging initialized. Output file: {absolute_path}")

        # ---------- Setup publishers ----------
        self.marker_pub = self.create_publisher(
            MarkerArray, 
            self.get_parameter("marker_topic").value, 
            10
        )

        # ---------- Setup subscribers ----------
        self.detection_info_sub = self.create_subscription(
            ObjectDetectionInfoArray,
            self.get_parameter("detection_info_topic").value,
            self.detection_info_callback,
            10,
        )

        # ---------- Create save CSV service ----------
        self.save_csv_service = self.create_service(
            Trigger, 
            'save_processed_detections_csv', 
            self.save_csv_callback
        )

        # ---------- Class colors for visualization ----------
        self.class_colors = {
            "chair": {"r": 1.0, "g": 0.0, "b": 0.0},
            "person": {"r": 0.0, "g": 1.0, "b": 0.0},
            "bottle": {"r": 0.0, "g": 0.0, "b": 1.0},
            "cup": {"r": 1.0, "g": 1.0, "b": 0.0},
            "book": {"r": 1.0, "g": 0.0, "b": 1.0},
            "laptop": {"r": 0.0, "g": 1.0, "b": 1.0},
        }

        self.marker_id_counter = 0

        self.get_logger().info(
            "[Detection Processor Node] Initialization complete. Waiting for detections..."
        )

    def publish_camera_correction_transform(self):
        """Publish static transform to correct camera frame orientation"""
        static_transform = TransformStamped()
        
        static_transform.header.stamp = self.get_clock().now().to_msg()
        static_transform.header.frame_id = "rgb_camera_link"
        static_transform.child_frame_id = "rgb_camera_link_corrected"
        
        # No translation needed
        static_transform.transform.translation.x = 0.0
        static_transform.transform.translation.y = 0.0
        static_transform.transform.translation.z = 0.0
        
        # Apply rotations: 90 degrees around Y (clockwise), then 90 degrees around Z (clockwise)
        # First rotation: 90 degrees clockwise around Y axis
        # Second rotation: 90 degrees clockwise around Z axis
        # Combined quaternion for these rotations
        import math
        
        # 90 degrees clockwise around Y = -90 degrees around Y
        y_rot = -math.pi / 2
        # 90 degrees clockwise around Z = -90 degrees around Z  
        z_rot = -math.pi / 2
        
        # Convert to quaternion (YZ rotation order)
        cy = math.cos(y_rot * 0.5)
        sy = math.sin(y_rot * 0.5)
        cz = math.cos(z_rot * 0.5)
        sz = math.sin(z_rot * 0.5)
        
        # Quaternion multiplication for Y then Z rotation
        static_transform.transform.rotation.x = sy * cz
        static_transform.transform.rotation.y = cy * sz
        static_transform.transform.rotation.z = cy * cz * (-sz/cz) + sy * sz
        static_transform.transform.rotation.w = cy * cz
        
        # Simpler approach - use known quaternion values for this specific rotation
        # 90 deg CW around Y, then 90 deg CW around Z
        static_transform.transform.rotation.x = -0.5
        static_transform.transform.rotation.y = 0.5
        static_transform.transform.rotation.z = -0.5
        static_transform.transform.rotation.w = 0.5
        
        self.static_broadcaster.sendTransform(static_transform)
        self.get_logger().info("Published camera correction transform: rgb_camera_link -> rgb_camera_link_corrected")        

    def detection_info_callback(self, msg):
        """Process incoming detection info messages"""
        try:
            target_classes = self.get_parameter("target_classes").value
            duplicate_threshold = self.get_parameter("duplicate_distance_threshold").value
            target_frame = self.get_parameter("target_frame").value
            tf_timeout = self.get_parameter("tf_timeout").value
            
            # Create marker array for visualization
            marker_array = MarkerArray()
            
            # Get timestamp for CSV logging
            detection_timestamp = datetime.fromtimestamp(
                msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            ).isoformat()

            processed_count = 0
            
            for detection_info in msg.info:

                class_name = detection_info.class_id
                
                # Check if this class is in our target classes
                if class_name not in target_classes:
                    self.get_logger().debug(f"Skipping object of class '{class_name}' - not in target classes")
                    continue

                # Create marker in original camera frame first
                original_marker = self.create_detection_marker(
                    detection_info.position,
                    "rgb_camera_link_corrected",
                    class_name,
                    msg.header.stamp,
                    "_original"
                )
                if original_marker:
                    marker_array.markers.append(original_marker)

                # Extract position and transform to target frame
                try:
                    # Create PointStamped message
                    point_stamped = PointStamped()
                    point_stamped.header = msg.header
                    point_stamped.point = detection_info.position

                    # point_stamped.header.frame_id = "rgb_camera_link"
                    # point_stamped.header.frame_id = "rgb_camera_link_corrected"
                    point_stamped.header.frame_id = "rgb_camera_optical_link"
                    
                    # Check if transform is available
                    if self.tf_buffer.can_transform(
                        target_frame, 
                        point_stamped.header.frame_id, 
                        msg.header.stamp,
                        timeout=rclpy.duration.Duration(seconds=tf_timeout)
                        # Time()
                    ):
                        # Transform to target frame
                        point_in_target_frame = self.tf_buffer.transform(
                            point_stamped, 
                            target_frame,
                            # timeout=rclpy.duration.Duration(seconds=tf_timeout)
                        )
                        transformed_position = point_in_target_frame.point
                        
                        self.get_logger().info(f"Transformed {class_name} from frame '{msg.header.frame_id}' to '{target_frame}': "
                                            f"({detection_info.position.x:.3f}, {detection_info.position.y:.3f}, {detection_info.position.z:.3f}) -> "
                                            f"({transformed_position.x:.3f}, {transformed_position.y:.3f}, {transformed_position.z:.3f})")
                    else:
                        self.get_logger().warn(f"Cannot transform from '{msg.header.frame_id}' to '{target_frame}' for {class_name} detection")
                        continue
                        
                except (LookupException, ConnectivityException, ExtrapolationException) as e:
                    self.get_logger().error(f"TF transform failed for {class_name} detection: {str(e)}")
                    continue
                
                # Check for duplicates using transformed position
                if not self.is_duplicate_detection(transformed_position, class_name, duplicate_threshold):
                    # Add to tracking dictionary (using transformed position)
                    self.add_detection_position(transformed_position, class_name)
                    
                    # Add to CSV data (using transformed position)
                    self.add_detection_to_csv_data(
                        class_name,
                        transformed_position.x,
                        transformed_position.y,
                        transformed_position.z,
                        detection_info.confidence,
                        detection_info.pose_estimation_type,
                        detection_timestamp,
                        target_frame  # Add frame info
                    )
                    
                    # Create visualization marker in target frame
                    transformed_marker = self.create_detection_marker(
                        transformed_position,
                        target_frame,
                        class_name,
                        msg.header.stamp,
                        "_transformed"
                    )
                    if transformed_marker:
                        marker_array.markers.append(transformed_marker)
                    
                    processed_count += 1
                    
                    self.get_logger().info(f"Processed new {class_name} detection at position ({target_frame}): "
                                        f"x={transformed_position.x:.2f}, y={transformed_position.y:.2f}, z={transformed_position.z:.2f}, "
                                        f"confidence={detection_info.confidence:.2f}")
                else:
                    self.get_logger().debug(f"Skipped duplicate {class_name} detection")

            # Publish markers if any were created
            if len(marker_array.markers) > 0:
                self.marker_pub.publish(marker_array)

            if processed_count > 0:
                self.get_logger().info(f"Processed {processed_count} new detections from {len(msg.info)} total detections")

        except Exception as e:
            self.get_logger().error(f"Error in detection_info_callback: {str(e)}")

    def is_duplicate_detection(self, position, class_name, threshold):
        """Check if a detection is too close to existing detections of the same class"""
        if class_name not in self.previous_detections:
            return False
        
        new_pos = np.array([position.x, position.y, position.z])
        
        for prev_pos in self.previous_detections[class_name]:
            prev_pos_array = np.array([prev_pos.x, prev_pos.y, prev_pos.z])
            distance = np.linalg.norm(new_pos - prev_pos_array)
            if distance < threshold:
                return True
        return False

    def add_detection_position(self, position, class_name):
        """Add a new detection position to the tracking dictionary"""
        if class_name not in self.previous_detections:
            self.previous_detections[class_name] = []
        self.previous_detections[class_name].append(position)

    def get_class_color(self, class_name):
        """Get color for a specific class"""
        if class_name in self.class_colors:
            return self.class_colors[class_name]
        else:
            # Default color if class not found
            return {"r": 0.5, "g": 0.5, "b": 0.5}

    def create_detection_marker(self, position, frame_id, class_name, stamp, namespace_suffix=""):
        """Create a visualization marker for a detection"""
        try:
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = f"detection_{class_name}{namespace_suffix}"
            marker.id = self.marker_id_counter
            self.marker_id_counter += 1
            
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            # Set position
            marker.pose.position = position
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            
            # Set scale - make original frame markers slightly smaller to distinguish
            if namespace_suffix == "_original":
                marker.scale.x = 0.2
                marker.scale.y = 0.2
                marker.scale.z = 0.2
            else:
                marker.scale.x = 0.3
                marker.scale.y = 0.3
                marker.scale.z = 0.3
            
            # Set color based on class
            color = self.get_class_color(class_name)
            marker.color.r = color["r"]
            marker.color.g = color["g"]
            marker.color.b = color["b"]
            
            # Make original frame markers more transparent to distinguish
            if namespace_suffix == "_original":
                marker.color.a = 0.5  # More transparent for original frame
            else:
                marker.color.a = 0.8  # Less transparent for transformed frame
            
            # Set lifetime
            # marker.lifetime.sec = int(self.get_parameter("marker_lifetime").value)
            
            return marker
            
        except Exception as e:
            self.get_logger().error(f"Failed to create marker: {str(e)}")
            return None

    # def add_detection_to_csv_data(self, class_name, x, y, z, confidence, estimation_type, timestamp=None, frame_id="map"):
    #     """Add a detection to the CSV data list"""
    #     if timestamp is None:
    #         timestamp = datetime.now().isoformat()
        
    #     detection_entry = {
    #         'timestamp': timestamp,
    #         'class': class_name,
    #         'x': x,
    #         'y': y,
    #         'z': z,
    #         'confidence': confidence,
    #         'estimation_type': estimation_type,
    #         'frame_id': frame_id
    #     }
    #     self.detections_data.append(detection_entry)
        
    #     self.get_logger().debug(f"Added detection to CSV data: {class_name} at ({x:.3f}, {y:.3f}, {z:.3f}) "
    #                            f"in frame '{frame_id}' with confidence {confidence:.3f}")

    def add_detection_to_csv_data(self, class_name, x, y, z, confidence, estimation_type, timestamp=None, frame_id="map"):
        """Add a detection to the CSV data list and immediately save it"""
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        detection_entry = {
            'timestamp': timestamp,
            'class': class_name,
            'x': x,
            'y': y,
            'z': z,
            'confidence': confidence,
            'estimation_type': estimation_type,
            'frame_id': frame_id
        }
        self.detections_data.append(detection_entry)

        self.get_logger().debug(
            f"Added detection to CSV data: {class_name} at ({x:.3f}, {y:.3f}, {z:.3f}) "
            f"in frame '{frame_id}' with confidence {confidence:.3f}"
        )

        # 💾 Save CSV after every addition
        try:
            self.save_detections_csv()
        except Exception as e:
            self.get_logger().error(f"Failed to auto-save CSV after detection: {str(e)}")


    def save_csv_callback(self, request, response):
        """Service callback to save CSV file on demand"""

        # self.add_detection_to_csv_data(
        #     "test",
        #     0.0,
        #     0.0,
        #     0.0,
        #     0.0,
        #     "test",
        #     0.0,
        #     "test"  # Add frame info
        # )

        try:
            self.save_detections_csv()
            response.success = True
            response.message = f"Successfully saved {len(self.detections_data)} detections to {self.csv_filename}"
            self.get_logger().info(response.message)
        except Exception as e:
            response.success = False
            response.message = f"Failed to save CSV: {str(e)}"
            self.get_logger().error(response.message)
        
        return response

    def save_detections_csv(self):
        """Save all detections to CSV file"""
        
        try:
            with open(self.csv_filename, 'w', newline='') as csvfile:
                fieldnames = ['timestamp', 'class', 'x', 'y', 'z', 'confidence', 'estimation_type', 'frame_id']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for detection in self.detections_data:
                    writer.writerow(detection)
            
            self.get_logger().info(f"Saved {len(self.detections_data)} detections to {self.csv_filename}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save CSV file: {str(e)}")
            raise

    def __del__(self):
        """Destructor - save CSV when node is destroyed"""
        try:
            if hasattr(self, 'detections_data') and self.detections_data:
                self.save_detections_csv()
        except:
            pass  # Ignore errors during destruction


def main(args=None):
    rclpy.init(args=args)

    node = None
    try:
        node = DetectionProcessorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info("Shutting down - saving CSV file...")
            try:
                node.save_detections_csv()
            except Exception as e:
                node.get_logger().error(f"Failed to save CSV on shutdown: {str(e)}")
    except Exception as e:
        if node:
            node.get_logger().fatal(f"Fatal error: {str(e)}")
            try:
                node.save_detections_csv()
            except:
                pass
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()