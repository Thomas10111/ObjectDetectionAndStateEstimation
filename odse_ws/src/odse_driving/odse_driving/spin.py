"""Run a node until Ctrl+C or launch shutdown, without a traceback."""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


def spin(node: Node) -> None:
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
