#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


class ClassificationMapNode(Node):
    def __init__(self):
        super().__init__("classification_map_node")

        self.declare_parameter("input_topic", "/gas_predictions")
        self.declare_parameter("output_dir", "/home/tolu/Documents/gas_classification_maps")
        self.declare_parameter("output_file", "classification_map.svg")
        self.declare_parameter("title", "Spatial Classification Results")
        self.declare_parameter("save_period_sec", 5.0)

        self.input_topic = self.get_parameter("input_topic").value
        self.output_dir = Path(self.get_parameter("output_dir").value)
        self.output_file = self.get_parameter("output_file").value
        self.title = self.get_parameter("title").value
        self.save_period_sec = float(self.get_parameter("save_period_sec").value)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.points = []

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub = self.create_subscription(
            Float32MultiArray,
            self.input_topic,
            self.prediction_callback,
            qos_profile,
        )

        self.timer = self.create_timer(self.save_period_sec, self.save_plot)

        self.get_logger().info("classification_map_node started")
        self.get_logger().info(f"  input_topic: {self.input_topic}")
        self.get_logger().info(f"  output_dir: {self.output_dir}")
        self.get_logger().info(f"  output_file: {self.output_file}")

    def prediction_callback(self, msg: Float32MultiArray):
        if len(msg.data) < 5:
            self.get_logger().warn(
                f"Prediction message has {len(msg.data)} entries, expected at least 5"
            )
            return

        pred = int(msg.data[0])
        conf = float(msg.data[1])
        x = float(msg.data[2])
        y = float(msg.data[3])
        timestamp = float(msg.data[4])

        self.points.append((pred, conf, x, y, timestamp))

    def save_plot(self):
        if not self.points:
            self.get_logger().warn("No prediction data yet, skipping visualization")
            return

        points = np.array(self.points, dtype=np.float64)
        preds = points[:, 0].astype(int)
        coords = points[:, 2:4]

        inlier_mask = preds == 1
        outlier_mask = preds != 1

        fig, ax = plt.subplots(figsize=(10, 8))

        if np.any(inlier_mask):
            ax.scatter(
                coords[inlier_mask, 0],
                coords[inlier_mask, 1],
                c="green",
                s=30,
                alpha=0.7,
                label=f"Inlier (+1): {int(np.sum(inlier_mask))}",
                edgecolors="darkgreen",
                linewidth=0.5,
            )

        if np.any(outlier_mask):
            ax.scatter(
                coords[outlier_mask, 0],
                coords[outlier_mask, 1],
                c="red",
                s=30,
                alpha=0.7,
                label=f"Outlier (-1): {int(np.sum(outlier_mask))}",
                edgecolors="darkred",
                linewidth=0.5,
            )

        ax.set_xlabel("X Position [m]")
        ax.set_ylabel("Y Position [m]")
        ax.set_title(self.title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

        params_text = (
            f"Total points: {len(points)}\n"
            f"Inliers: {int(np.sum(inlier_mask))}\n"
            f"Outliers: {int(np.sum(outlier_mask))}"
        )

        ax.text(
            0.02,
            0.98,
            params_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        timestamp = self.get_clock().now().nanoseconds
        output_path = self.output_dir / f"classification_map_{timestamp}.svg"
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

        self.get_logger().info(f"Saved visualization to {output_path}")


def main(args=None):
    rclpy.init(args=args)
    node = ClassificationMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
