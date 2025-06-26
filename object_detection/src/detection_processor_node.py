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
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

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

        # ---------- Initialize parameters ----------
        self.declare_parameters(
            namespace="",
            parameters=[
                ("detection_info_topic", "detection_info"),
                ("target_classes", ["backpack", "umbrella", "stop sign", "clock", "bottle"]),
                ("csv_output_dir", "detections"),
                ("marker_topic", "/detection_markers"),
                ("refined_marker_topic", "/refined_detection_markers"),
                ("marker_lifetime", 30.0),
                ("target_frame", "map"),
                ("tf_timeout", 1.0),
                # Updated clustering parameters for better object grouping
                ("cluster_eps", 0.8),  # Increased from 0.5 - larger search radius
                ("cluster_min_samples", 3),  # Increased from 2 - more robust clusters
                ("confidence_weight", 0.0),  # Reduced from 0.3 - focus on spatial clustering
                ("outlier_confidence_threshold", 0.25),  # Reduced from 0.3 - less strict
                ("processing_interval", 5.0),
                ("merge_distance", 0.5),  # New parameter for post-clustering merge
            ],
        )

        # ---------- Initialize detection storage ----------
        self.all_detections = {}  # Dictionary: class_name -> list of detection data
        self.refined_positions = {}  # Dictionary: class_name -> list of refined positions
        self.detections_data = []  # List for CSV output

        # ---------- Initialize CSV logging ----------
        self.csv_output_dir = self.get_parameter("csv_output_dir").value
        os.makedirs(self.csv_output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = os.path.join(self.csv_output_dir, f"processed_detections_{timestamp}.csv")
        self.refined_csv_filename = os.path.join(self.csv_output_dir, f"refined_detections_{timestamp}.csv")
        
        absolute_path = os.path.abspath(self.csv_filename)
        self.get_logger().info(f"[Detection Processor Node] CSV logging initialized. Output file: {absolute_path}")

        # ---------- Setup publishers ----------
        self.marker_pub = self.create_publisher(
            MarkerArray, 
            self.get_parameter("marker_topic").value, 
            10
        )
        
        self.refined_marker_pub = self.create_publisher(
            MarkerArray,
            self.get_parameter("refined_marker_topic").value,
            10
        )

        # ---------- Setup subscribers ----------
        self.detection_info_sub = self.create_subscription(
            ObjectDetectionInfoArray,
            self.get_parameter("detection_info_topic").value,
            self.detection_info_callback,
            10,
        )

        # ---------- Setup processing timer ----------
        self.processing_timer = self.create_timer(
            self.get_parameter("processing_interval").value,
            self.process_detection_clusters
        )

        # ---------- Create services ----------
        self.save_csv_service = self.create_service(
            Trigger, 
            'save_processed_detections_csv', 
            self.save_csv_callback
        )
        
        self.process_clusters_service = self.create_service(
            Trigger,
            'process_detection_clusters',
            self.process_clusters_service_callback
        )

        # ---------- Class colors for visualization ----------
        self.class_colors = {
            "chair": {"r": 1.0, "g": 0.0, "b": 0.0},
            "person": {"r": 0.0, "g": 1.0, "b": 0.0},
            "bottle": {"r": 0.0, "g": 0.0, "b": 1.0},
            "cup": {"r": 1.0, "g": 1.0, "b": 0.0},
            "book": {"r": 1.0, "g": 0.0, "b": 1.0},
            "laptop": {"r": 0.0, "g": 1.0, "b": 1.0},
            "backpack": {"r": 0.8, "g": 0.4, "b": 0.0},
            "umbrella": {"r": 0.5, "g": 0.0, "b": 0.8},
            "stop sign": {"r": 1.0, "g": 0.2, "b": 0.2},
            "clock": {"r": 0.2, "g": 0.8, "b": 0.2},
        }

        self.marker_id_counter = 0

        self.get_logger().info(
            "[Detection Processor Node] Initialization complete. Waiting for detections..."
        )

    def detection_info_callback(self, msg):
        """Process incoming detection info messages - store ALL detections"""
        try:
            target_frame = self.get_parameter("target_frame").value
            tf_timeout = self.get_parameter("tf_timeout").value
            
            # Create marker array for visualization of raw detections
            marker_array = MarkerArray()
            
            # Get timestamp for logging
            detection_timestamp = datetime.fromtimestamp(
                msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            ).isoformat()

            processed_count = 0
            
            for detection_info in msg.info:
                class_name = detection_info.class_id
                
                # Transform to target frame
                try:
                    point_stamped = PointStamped()
                    point_stamped.header = msg.header
                    point_stamped.point = detection_info.position
                    point_stamped.header.frame_id = "rgb_camera_optical_link"
                    
                    if self.tf_buffer.can_transform(
                        target_frame, 
                        point_stamped.header.frame_id, 
                        msg.header.stamp,
                        timeout=rclpy.duration.Duration(seconds=tf_timeout)
                    ):
                        point_in_target_frame = self.tf_buffer.transform(
                            point_stamped, 
                            target_frame,
                        )
                        transformed_position = point_in_target_frame.point
                        
                        # Store ALL detections (no duplicate filtering)
                        detection_data = {
                            'timestamp': detection_timestamp,
                            'position': np.array([transformed_position.x, transformed_position.y, transformed_position.z]),
                            'confidence': detection_info.confidence,
                            'estimation_type': detection_info.pose_estimation_type,
                            'frame_id': target_frame,
                            'ros_timestamp': msg.header.stamp
                        }
                        
                        # Add to class-specific detection list
                        if class_name not in self.all_detections:
                            self.all_detections[class_name] = []
                        self.all_detections[class_name].append(detection_data)
                        
                        # Add to CSV data for raw detections
                        self.add_detection_to_csv_data(
                            class_name,
                            transformed_position.x,
                            transformed_position.y,
                            transformed_position.z,
                            detection_info.confidence,
                            detection_info.pose_estimation_type,
                            detection_timestamp,
                            target_frame
                        )
                        
                        # Create visualization marker for raw detection
                        marker = self.create_detection_marker(
                            transformed_position,
                            target_frame,
                            class_name,
                            msg.header.stamp,
                            "_raw"
                        )
                        if marker:
                            marker_array.markers.append(marker)
                        
                        processed_count += 1
                        
                        self.get_logger().debug(f"Stored {class_name} detection at position ({target_frame}): "
                                            f"x={transformed_position.x:.2f}, y={transformed_position.y:.2f}, z={transformed_position.z:.2f}, "
                                            f"confidence={detection_info.confidence:.2f}")
                    else:
                        self.get_logger().warn(f"Cannot transform from '{point_stamped.header.frame_id}' to '{target_frame}' for {class_name} detection")
                        continue
                        
                except (LookupException, ConnectivityException, ExtrapolationException) as e:
                    self.get_logger().error(f"TF transform failed for {class_name} detection: {str(e)}")
                    continue

            # Publish raw detection markers
            if len(marker_array.markers) > 0:
                self.marker_pub.publish(marker_array)

            if processed_count > 0:
                self.get_logger().info(f"Stored {processed_count} detections from {len(msg.info)} total detections")
                # Log current detection counts
                for class_name, detections in self.all_detections.items():
                    self.get_logger().debug(f"Total {class_name} detections stored: {len(detections)}")

        except Exception as e:
            self.get_logger().error(f"Error in detection_info_callback: {str(e)}")

    def process_detection_clusters(self):
        """Process stored detections using clustering to filter outliers and refine positions"""
        try:
            eps = self.get_parameter("cluster_eps").value
            min_samples = self.get_parameter("cluster_min_samples").value
            confidence_weight = self.get_parameter("confidence_weight").value
            confidence_threshold = self.get_parameter("outlier_confidence_threshold").value
            
            refined_marker_array = MarkerArray()
            
            for class_name, detections in self.all_detections.items():
                if len(detections) < min_samples:
                    self.get_logger().debug(f"Not enough detections for {class_name} clustering: {len(detections)} < {min_samples}")
                    continue
                
                self.get_logger().info(f"Processing {len(detections)} {class_name} detections for clustering")
                
                # Prepare data for clustering
                positions = np.array([d['position'] for d in detections])
                confidences = np.array([d['confidence'] for d in detections])
                
                # Use ONLY spatial coordinates for clustering (remove confidence from features)
                # The confidence weighting was causing over-segmentation
                features = positions  # Only x, y, z coordinates
                
                # Apply DBSCAN clustering with relaxed parameters
                # Increase eps for larger search radius to group nearby detections
                adjusted_eps = max(eps, 0.3)  # Ensure minimum eps of 0.3 meters
                clustering = DBSCAN(eps=adjusted_eps, min_samples=min_samples).fit(features)
                labels = clustering.labels_
                
                # Get unique clusters (excluding noise points labeled as -1)
                unique_labels = set(labels)
                n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
                n_noise = list(labels).count(-1)
                
                self.get_logger().info(f"{class_name}: Found {n_clusters} initial clusters and {n_noise} outliers")
                
                # Process each cluster and collect initial refined positions
                initial_clusters = []
                
                for label in unique_labels:
                    if label == -1:  # Skip noise points
                        continue
                    
                    # Get points in this cluster
                    cluster_mask = (labels == label)
                    cluster_positions = positions[cluster_mask]
                    cluster_confidences = confidences[cluster_mask]
                    
                    # Filter by confidence threshold
                    high_confidence_mask = cluster_confidences >= confidence_threshold
                    if not np.any(high_confidence_mask):
                        self.get_logger().debug(f"Cluster {label} for {class_name} has no high-confidence detections")
                        continue
                    
                    filtered_positions = cluster_positions[high_confidence_mask]
                    filtered_confidences = cluster_confidences[high_confidence_mask]
                    
                    # Calculate weighted centroid
                    weights = filtered_confidences / np.sum(filtered_confidences)
                    refined_position = np.average(filtered_positions, axis=0, weights=weights)
                    
                    # Calculate cluster statistics
                    cluster_size = len(filtered_positions)
                    avg_confidence = np.mean(filtered_confidences)
                    max_confidence = np.max(filtered_confidences)
                    position_std = np.std(filtered_positions, axis=0)
                    
                    initial_clusters.append({
                        'position': refined_position,
                        'cluster_size': cluster_size,
                        'avg_confidence': avg_confidence,
                        'max_confidence': max_confidence,
                        'position_std': position_std,
                        'cluster_id': label,
                        'original_positions': filtered_positions,
                        'original_confidences': filtered_confidences
                    })
                
                # Post-processing: Merge nearby clusters that likely represent the same object
                merged_clusters = self.merge_nearby_clusters(initial_clusters, merge_distance=0.5)
                
                self.get_logger().info(f"{class_name}: After merging, {len(merged_clusters)} final clusters")
                
                # Create refined data for final clusters
                refined_positions_for_class = []
                
                for i, cluster in enumerate(merged_clusters):
                    refined_data = {
                        'class_name': class_name,
                        'position': cluster['position'],
                        'cluster_size': cluster['cluster_size'],
                        'avg_confidence': cluster['avg_confidence'],
                        'max_confidence': cluster['max_confidence'],
                        'position_std': cluster['position_std'],
                        'cluster_id': i  # Renumber after merging
                    }
                    
                    refined_positions_for_class.append(refined_data)
                    
                    self.get_logger().info(f"Final {class_name} cluster {i}: "
                                        f"pos=({cluster['position'][0]:.2f}, {cluster['position'][1]:.2f}, {cluster['position'][2]:.2f}), "
                                        f"size={cluster['cluster_size']}, avg_conf={cluster['avg_confidence']:.2f}")
                    
                    # Create refined marker
                    refined_marker = self.create_refined_marker(refined_data)
                    if refined_marker:
                        refined_marker_array.markers.append(refined_marker)
                
                # Update refined positions for this class
                self.refined_positions[class_name] = refined_positions_for_class
            
            # Publish refined markers
            if len(refined_marker_array.markers) > 0:
                self.refined_marker_pub.publish(refined_marker_array)
                self.get_logger().info(f"Published {len(refined_marker_array.markers)} refined detection markers")
            
            # Save refined positions to CSV
            self.save_refined_detections_csv()
            
        except Exception as e:
            self.get_logger().error(f"Error in process_detection_clusters: {str(e)}")

    def merge_nearby_clusters(self, clusters, merge_distance=0.5):
        """Merge clusters that are closer than merge_distance to each other"""
        if len(clusters) <= 1:
            return clusters
        
        merged = []
        used = [False] * len(clusters)
        
        for i, cluster1 in enumerate(clusters):
            if used[i]:
                continue
            
            # Start a new merged cluster
            merged_cluster = {
                'position': cluster1['position'].copy(),
                'cluster_size': cluster1['cluster_size'],
                'avg_confidence': cluster1['avg_confidence'],
                'max_confidence': cluster1['max_confidence'],
                'position_std': cluster1['position_std'],
                'original_positions': cluster1['original_positions'].copy(),
                'original_confidences': cluster1['original_confidences'].copy()
            }
            
            total_weight = cluster1['cluster_size'] * cluster1['avg_confidence']
            weighted_position = cluster1['position'] * total_weight
            
            used[i] = True
            
            # Find nearby clusters to merge
            for j, cluster2 in enumerate(clusters):
                if used[j] or i == j:
                    continue
                
                # Calculate distance between cluster centroids
                distance = np.linalg.norm(cluster1['position'] - cluster2['position'])
                
                if distance < merge_distance:
                    self.get_logger().info(f"Merging clusters {i} and {j} (distance: {distance:.3f}m)")
                    
                    # Merge the clusters
                    cluster2_weight = cluster2['cluster_size'] * cluster2['avg_confidence']
                    total_weight += cluster2_weight
                    weighted_position += cluster2['position'] * cluster2_weight
                    
                    # Combine original data
                    merged_cluster['original_positions'] = np.vstack([
                        merged_cluster['original_positions'],
                        cluster2['original_positions']
                    ])
                    merged_cluster['original_confidences'] = np.concatenate([
                        merged_cluster['original_confidences'],
                        cluster2['original_confidences']
                    ])
                    
                    merged_cluster['cluster_size'] += cluster2['cluster_size']
                    merged_cluster['max_confidence'] = max(
                        merged_cluster['max_confidence'], 
                        cluster2['max_confidence']
                    )
                    
                    used[j] = True
            
            # Recalculate merged cluster properties
            if total_weight > 0:
                merged_cluster['position'] = weighted_position / total_weight
            
            merged_cluster['avg_confidence'] = np.mean(merged_cluster['original_confidences'])
            merged_cluster['position_std'] = np.std(merged_cluster['original_positions'], axis=0)
            
            merged.append(merged_cluster)
        
        return merged

    def create_refined_marker(self, refined_data):
        """Create a visualization marker for refined detection"""
        try:
            marker = Marker()
            marker.header.frame_id = self.get_parameter("target_frame").value
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = f"refined_{refined_data['class_name']}"
            marker.id = self.marker_id_counter
            self.marker_id_counter += 1
            
            marker.type = Marker.CYLINDER  # Use cylinder to distinguish from raw detections
            marker.action = Marker.ADD
            
            # Set position
            pos = refined_data['position']
            marker.pose.position.x = float(pos[0])
            marker.pose.position.y = float(pos[1])
            marker.pose.position.z = float(pos[2])
            marker.pose.orientation.w = 1.0
            
            # Scale based on cluster size and confidence
            base_scale = 0.2
            confidence_scale = min(refined_data['avg_confidence'] * 0.5, 0.3)
            size_scale = min(refined_data['cluster_size'] * 0.05, 0.2)
            total_scale = base_scale + confidence_scale + size_scale
            
            marker.scale.x = total_scale
            marker.scale.y = total_scale
            marker.scale.z = total_scale * 2  # Make cylinder taller
            
            # Set color based on class (brighter for refined)
            color = self.get_class_color(refined_data['class_name'])
            marker.color.r = min(color["r"] * 1.2, 1.0)
            marker.color.g = min(color["g"] * 1.2, 1.0)
            marker.color.b = min(color["b"] * 1.2, 1.0)
            marker.color.a = 0.9  # More opaque for refined detections
            
            return marker
            
        except Exception as e:
            self.get_logger().error(f"Failed to create refined marker: {str(e)}")
            return None

    # def save_refined_detections_csv(self):
    #     """Save refined detection clusters to CSV"""
    #     try:
    #         refined_data_for_csv = []
            
    #         for class_name, refined_positions in self.refined_positions.items():
    #             for refined_data in refined_positions:
    #                 pos = refined_data['position']
    #                 entry = {
    #                     'timestamp': datetime.now().isoformat(),
    #                     'class': class_name,
    #                     'x': float(pos[0]),
    #                     'y': float(pos[1]),
    #                     'z': float(pos[2]),
    #                     'cluster_size': refined_data['cluster_size'],
    #                     'avg_confidence': refined_data['avg_confidence'],
    #                     'max_confidence': refined_data['max_confidence'],
    #                     'position_std_x': float(refined_data['position_std'][0]),
    #                     'position_std_y': float(refined_data['position_std'][1]),
    #                     'position_std_z': float(refined_data['position_std'][2]),
    #                     'cluster_id': refined_data['cluster_id'],
    #                     'frame_id': self.get_parameter("target_frame").value
    #                 }
    #                 refined_data_for_csv.append(entry)
            
    #         if refined_data_for_csv:
    #             with open(self.refined_csv_filename, 'w', newline='') as csvfile:
    #                 fieldnames = ['timestamp', 'class', 'x', 'y', 'z', 'cluster_size', 
    #                             'avg_confidence', 'max_confidence', 'position_std_x', 
    #                             'position_std_y', 'position_std_z', 'cluster_id', 'frame_id']
    #                 writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
    #                 writer.writeheader()
    #                 for entry in refined_data_for_csv:
    #                     writer.writerow(entry)
                
    #             self.get_logger().info(f"Saved {len(refined_data_for_csv)} refined detections to {self.refined_csv_filename}")
        
    #     except Exception as e:
    #         self.get_logger().error(f"Failed to save refined CSV: {str(e)}")
    def save_refined_detections_csv(self):
        """Save refined detection clusters to CSV"""
        try:
            refined_data_for_csv = []
            
            for class_name, refined_positions in self.refined_positions.items():
                for refined_data in refined_positions:
                    pos = refined_data['position']
                    entry = {
                        'class': class_name,
                        'x': float(pos[0]),
                        'y': float(pos[1]),
                        'z': float(pos[2])
                    }
                    refined_data_for_csv.append(entry)
            
            if refined_data_for_csv:
                with open(self.refined_csv_filename, 'w', newline='') as csvfile:
                    fieldnames = ['class', 'x', 'y', 'z']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    writer.writeheader()
                    for entry in refined_data_for_csv:
                        writer.writerow(entry)
                
                self.get_logger().info(f"Saved {len(refined_data_for_csv)} refined detections to {self.refined_csv_filename}")
        
        except Exception as e:
            self.get_logger().error(f"Failed to save refined CSV: {str(e)}")

    def process_clusters_service_callback(self, request, response):
        """Service to manually trigger cluster processing"""
        try:
            self.process_detection_clusters()
            total_refined = sum(len(positions) for positions in self.refined_positions.values())
            response.success = True
            response.message = f"Processed clusters successfully. Found {total_refined} refined positions."
            self.get_logger().info(response.message)
        except Exception as e:
            response.success = False
            response.message = f"Failed to process clusters: {str(e)}"
            self.get_logger().error(response.message)
        
        return response

    # Keep existing methods with minimal changes...
    def get_class_color(self, class_name):
        """Get color for a specific class"""
        if class_name in self.class_colors:
            return self.class_colors[class_name]
        else:
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
            
            marker.pose.position = position
            marker.pose.orientation.w = 1.0
            
            # Smaller markers for raw detections
            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.15
            
            color = self.get_class_color(class_name)
            marker.color.r = color["r"]
            marker.color.g = color["g"]
            marker.color.b = color["b"]
            marker.color.a = 0.6  # Semi-transparent for raw detections
            
            return marker
            
        except Exception as e:
            self.get_logger().error(f"Failed to create marker: {str(e)}")
            return None

    def add_detection_to_csv_data(self, class_name, x, y, z, confidence, estimation_type, timestamp=None, frame_id="map"):
        """Add a detection to the CSV data list and immediately save it"""
        detection_entry = {
            'class': class_name,
            'x': x,
            'y': y,
            'z': z
        }
        self.detections_data.append(detection_entry)

        # Auto-save CSV after every addition
        try:
            self.save_detections_csv()
        except Exception as e:
            self.get_logger().error(f"Failed to auto-save CSV after detection: {str(e)}")

    def save_csv_callback(self, request, response):
        """Service callback to save CSV file on demand"""
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
                fieldnames = ['class', 'x', 'y', 'z']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for detection in self.detections_data:
                    writer.writerow(detection)
            
            self.get_logger().debug(f"Saved {len(self.detections_data)} detections to {self.csv_filename}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save CSV file: {str(e)}")
            raise

    def __del__(self):
        """Destructor - save CSV when node is destroyed"""
        try:
            if hasattr(self, 'detections_data') and self.detections_data:
                self.save_detections_csv()
            if hasattr(self, 'refined_positions') and self.refined_positions:
                self.save_refined_detections_csv()
        except Exception as e:
            # Use print instead of logger since node might be destroyed
            print(f"Error saving CSV in destructor: {e}")


def main(args=None):
    rclpy.init(args=args)

    node = None
    try:
        node = DetectionProcessorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.get_logger().info("Shutting down - saving CSV files...")
            try:
                node.save_detections_csv()
                node.save_refined_detections_csv()
            except Exception as e:
                node.get_logger().error(f"Failed to save CSV on shutdown: {str(e)}")
    except Exception as e:
        if node:
            node.get_logger().fatal(f"Fatal error: {str(e)}")
            try:
                node.save_detections_csv()
                node.save_refined_detections_csv()
            except Exception as save_e:
                if node:
                    node.get_logger().error(f"Failed to save CSV on error: {str(save_e)}")
    finally:
        if node:
            try:
                node.destroy_node()
            except Exception as destroy_e:
                print(f"Error destroying node: {destroy_e}")
        
        # Only shutdown if rclpy is still initialized
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as shutdown_e:
            print(f"Error during shutdown: {shutdown_e}")


if __name__ == "__main__":
    main()