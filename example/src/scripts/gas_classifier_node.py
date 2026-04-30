#!/usr/bin/env python3
"""
Live Gas Classification ROS2 Node (Python)

Collects sensor data for 5 seconds, preprocesses it matching the training pipeline,
and classifies using a trained LOF model. Publishes predictions with location info.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float64MultiArray, Float32MultiArray
from geometry_msgs.msg import PoseStamped

import numpy as np
import onnxruntime as ort
from pathlib import Path
from datetime import datetime
from collections import deque
import threading
from scipy.signal import butter, filtfilt
import json
import warnings

warnings.filterwarnings('ignore')


class GasClassifierNode(Node):
    """ROS2 node for real-time gas classification with buffering and preprocessing"""

    def __init__(self):
        super().__init__('gas_classifier_node')

        # Declare parameters
        self.declare_parameter('model_path', '/home/tolu/Documents/saved_models/lof_basic_model_2026-03-20_17-23-54.onnx')
        self.declare_parameter('buffer_duration_sec', 5.0)
        self.declare_parameter('n_baseline', 100)
        self.declare_parameter('baseline_method', 'conductance_change')
        self.declare_parameter('lowpass_cutoff', 0.5)
        self.declare_parameter('lowpass_fs', 20.0)
        self.declare_parameter('lowpass_order', 2)
        self.declare_parameter('sensor_topic', '/gas_sensors_resistance')
        self.declare_parameter('pose_topic', '/utlidar/robot_pose')
        self.declare_parameter('output_topic', '/gas_predictions')
        self.declare_parameter('output_dir', '/home/tolu/Documents/gas_classifier_outputs')

        # Get parameters
        self.model_path = self.get_parameter('model_path').value
        self.buffer_duration_sec = self.get_parameter('buffer_duration_sec').value
        self.n_baseline = self.get_parameter('n_baseline').value
        self.baseline_method = self.get_parameter('baseline_method').value
        self.lowpass_cutoff = self.get_parameter('lowpass_cutoff').value
        self.lowpass_fs = self.get_parameter('lowpass_fs').value
        self.lowpass_order = self.get_parameter('lowpass_order').value
        self.output_topic = self.get_parameter('output_topic').value
        self.output_dir = Path(self.get_parameter('output_dir').value)

        # Setup output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model
        try:
            self.ort_session = ort.InferenceSession(
            self.model_path,
            providers=["CPUExecutionProvider"],
            )
            self.ort_input_name = self.ort_session.get_inputs()[0].name
            self.ort_output_names = [out.name for out in self.ort_session.get_outputs()]

            self.get_logger().info(f"ONNX model loaded from: {self.model_path}")
            self.get_logger().info(f"ONNX input: {self.ort_input_name}")
            self.get_logger().info(f"ONNX outputs: {self.ort_output_names}")
        except FileNotFoundError:
            self.get_logger().error(f"Model not found: {self.model_path}")
            raise
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            raise

        for out in self.ort_session.get_outputs():
            self.get_logger().info(
                f"ONNX output: name={out.name}, shape={out.shape}, type={out.type}"
        )

        # Data buffers
        self.sensor_buffer = deque()
        self.timestamps_buffer = deque()
        self.pose_buffer = deque()  # Store position for each sensor reading
        self.current_pose = {'x': 0.0, 'y': 0.0, 'theta': 0.0}
        self.lock = threading.Lock()

        # Baseline values
        self.baseline_values = None
        self.baseline_calculated = False

        # Quality of Service Profile
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers
        self.sensor_sub = self.create_subscription(
            Float64MultiArray,
            self.get_parameter('sensor_topic').value,
            self.sensor_callback,
            qos_profile
        )


        self.pose_sub = self.create_subscription(
            PoseStamped,
            self.get_parameter('pose_topic').value,
            self.pose_callback,
            qos_profile
        )

        # Publisher for predictions
        self.pred_pub = self.create_publisher(
            Float32MultiArray,
            self.output_topic,
            qos_profile
        )

        # Timer for periodic batch processing
        self.batch_timer = self.create_timer(
            self.buffer_duration_sec,
            self.process_batch
        )

        self.get_logger().info(f"Gas Classifier Node initialized")
        self.get_logger().info(f"  Model: {self.model_path}")
        self.get_logger().info(f"  Buffer duration: {self.buffer_duration_sec} seconds")
        self.get_logger().info(f"  Baseline samples: {self.n_baseline}")
        self.get_logger().info(f"  Baseline method: {self.baseline_method}")
        self.get_logger().info(f"  Output topic: {self.output_topic}")

    # =========================================================================
    # Callbacks
    # =========================================================================

    def sensor_callback(self, msg: Float64MultiArray):
        """Handle incoming sensor data"""
        with self.lock:
            if not msg.data:
                self.get_logger().warn("Received empty sensor message")
                return

            sensor_values = np.array(msg.data, dtype=np.float32)
            timestamp = self.get_clock().now().nanoseconds / 1e9  # Convert to seconds

            self.sensor_buffer.append(sensor_values)
            self.timestamps_buffer.append(timestamp)
            self.pose_buffer.append(self.current_pose.copy())

            self.get_logger().debug(
                f"Received sensor data: {len(msg.data)} values at position "
                f"({self.current_pose['x']:.2f}, {self.current_pose['y']:.2f}), "
                f"buffer size: {len(self.sensor_buffer)}"
            )


    def pose_callback(self, msg: PoseStamped):
        """Handle PoseStamped messages"""
        with self.lock:
            self.current_pose = {
                'x': float(msg.pose.position.x),
                'y': float(msg.pose.position.y),
                'theta': 0.0  # Extract from quaternion if needed
            }
            self.get_logger().debug(
                f"Pose updated: X={msg.pose.position.x:.3f}, Y={msg.pose.position.y:.3f}"
            )

    # =========================================================================
    # Batch Processing
    # =========================================================================

    def process_batch(self):
        """Process buffered sensor data and generate predictions"""
        # Make copies with minimal lock holding
        with self.lock:
            if len(self.sensor_buffer) == 0:
                self.get_logger().warn("Empty buffer")
                return

            X_raw = np.array(list(self.sensor_buffer), dtype=np.float32)
            timestamps = list(self.timestamps_buffer)
            poses = list(self.pose_buffer)


            # Clear buffers immediately
            self.sensor_buffer.clear()
            self.timestamps_buffer.clear()
            self.pose_buffer.clear()
        # Lock RELEASED

        # WITHOUT lock
        if not self.baseline_calculated:
            self._calculate_baseline(X_raw)
            return


        X_processed = self._preprocess_data(X_raw)
        predictions = self._classify_batch(X_processed)
        self._publish_predictions(predictions, poses, timestamps)

    # =========================================================================
    # Baseline Calculation
    # =========================================================================

    def _calculate_baseline(self, X: np.ndarray):
        """Calculate baseline as median of first n_baseline samples"""
        n_samples = min(self.n_baseline, len(X))

        # Median of first n_baseline samples for each feature
        self.baseline_values = np.median(X[:n_samples, :], axis=0)

        # Prevent division by zero
        self.baseline_values = np.where(
            self.baseline_values == 0,
            1e-6,
            self.baseline_values
        )

        self.baseline_calculated = True
        self.get_logger().info(
            f"Baseline calculated from {n_samples} samples. Can start path now!"
        )

    # =========================================================================
    # Preprocessing
    # =========================================================================

    def _preprocess_data(self, X_raw: np.ndarray) -> np.ndarray:
        """Apply preprocessing pipeline matching training"""
        # Step 1: Baseline correction
        X_corrected = self._correct_baseline(X_raw)

        # Step 2: Low-pass filtering
        X_filtered = self._apply_lowpass_filter(X_corrected)

        return X_filtered

    def _correct_baseline(self, X: np.ndarray) -> np.ndarray:
        """Apply baseline correction using configured method"""
        if self.baseline_method == "conductance_change":
            # (R0/R) - 1
            X_corrected = (self.baseline_values / (X + 1e-9)) - 1.0

        elif self.baseline_method == "relative_resistance":
            # R/R0 - 1
            X_corrected = (X / self.baseline_values) - 1.0

        elif self.baseline_method == "difference_subtraction":
            # R - R0
            X_corrected = X - self.baseline_values

        else:
            self.get_logger().warn(
                f"Unknown baseline method: {self.baseline_method}, using raw data"
            )
            X_corrected = X

        return X_corrected

    def _apply_lowpass_filter(self, X: np.ndarray) -> np.ndarray:
        """Apply Butterworth low-pass filter to each feature"""
        if X.size == 0:
            return X

        # Design Butterworth filter
        nyquist = 0.5 * self.lowpass_fs
        normalized_cutoff = self.lowpass_cutoff / nyquist

        # Ensure normalized cutoff is valid (0 < wn < 1)
        normalized_cutoff = np.clip(normalized_cutoff, 0.01, 0.99)


        b, a = butter(self.lowpass_order, normalized_cutoff, btype='low', analog=False)

        # filtfilt needs more than padlen samples
        padlen = 3 * max(len(a), len(b))
        n_samples = X.shape[0]

        if n_samples <= padlen:
            self.get_logger().warn(
                f"Skipping low-pass filter: need > {padlen} samples, got {n_samples}"
            )
            return X

        # filter each column (sensor) independently
        return filtfilt(b, a, X, axis=0)

    # =========================================================================
    # Classification
    # =========================================================================

    def _classify_batch_joblib(self, X_processed: np.ndarray) -> list:
        """Perform batch predictions using loaded model"""
        try:
            # Get predictions
            predictions = self.model.predict(X_processed)

            # Get decision scores/distances
            if hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(X_processed)
            elif hasattr(self.model, 'score_samples'):
                scores = self.model.score_samples(X_processed)
            else:
                scores = np.ones(len(predictions)) * 0.5
                self.get_logger().warn(
                "Model has neither decision_function nor score_samples; ")

            # # Normalize scores to [0, 1] confidence
            # min_score = np.min(scores)
            # max_score = np.max(scores)

            # if max_score > min_score:
            #     confidences = (scores - min_score) / (max_score - min_score)
            # else:
            #     confidences = np.ones_like(scores) * 0.5

            # Build result list
            results = []
            for pred, conf in zip(predictions, scores):
                results.append({
                    'pred': int(pred),
                    'conf': float(conf)
                })

            self.get_logger().info(f"Predictions computed: {len(results)} samples")
            return results

        except Exception as e:
            self.get_logger().error(f"Prediction failed: {e}")
            return []

    def _classify_batch(self, X_processed: np.ndarray) -> list:
        """Perform batch predictions using ONNX Runtime"""
        try:
            if X_processed.size == 0:
                return []

            X_input = X_processed.astype(np.float32, copy=False)

            labels, scores = self.ort_session.run(
                ["label", "scores"],
                {self.ort_input_name: X_input}
            )

            labels = np.asarray(labels).reshape(-1)
            scores = np.asarray(scores).reshape(-1)

            results = [
                {
                    'pred': int(pred),
                    'conf': float(score)
                }
                for pred, score in zip(labels, scores)
            ]

            self.get_logger().info(f"Predictions computed: {len(results)} samples")
            return results

        except Exception as e:
            self.get_logger().error(f"ONNX prediction failed: {e}")
            return []

    # =========================================================================
    # Publishing
    # =========================================================================

    def _publish_predictions(self, predictions: list, poses: list, timestamps: list):
        """Publish predictions to ROS2 topic"""
        if not predictions:
            self.get_logger().warn("No predictions to publish")
            return

        for i, pred in enumerate(predictions):
            x_pos = poses[i]['x'] if i < len(poses) else 0.0
            y_pos = poses[i]['y'] if i < len(poses) else 0.0
            timestamp = timestamps[i] if i < len(timestamps) else 0.0

            msg = Float32MultiArray()
            msg.data = [
                float(pred['pred']),
                float(pred['conf']),
                float(x_pos),
                float(y_pos),
                float(timestamp)
            ]
            self.pred_pub.publish(msg)

        self.get_logger().info(
            f"Published {len(predictions)} predictions with indexed position matching"
        )

        # # Optionally save debug info
        # self._save_debug_info(predictions, x_pos, y_pos)

    def _save_debug_info(self, predictions: list, x_pos: float, y_pos: float):
        """Save debug information to file"""
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

            # Create summary
            summary = {
                'timestamp': timestamp,
                'position': {'x': x_pos, 'y': y_pos},
                'n_predictions': len(predictions),
                'predictions': predictions,
                'inliers': sum(1 for p in predictions if p['pred'] == 1),
                'outliers': sum(1 for p in predictions if p['pred'] == -1),
            }

            # Save to JSON
            output_file = self.output_dir / f"predictions_{timestamp}.json"
            with open(output_file, 'w') as f:
                json.dump(summary, f, indent=2)

            self.get_logger().debug(f"Saved debug info to: {output_file}")

        except Exception as e:
            self.get_logger().warn(f"Failed to save debug info: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = GasClassifierNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
