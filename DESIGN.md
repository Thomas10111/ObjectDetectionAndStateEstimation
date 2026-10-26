# ObjectDetectionAndStateEstimation

Real-time object detection **and** onboard state estimation on an **NVIDIA Jetson
Orin**, driven by a **single camera** whose frames are shared by two consumers, and
fused with an onboard **IMU**.

This is the successor to the [`ObjectDetection`](../ObjectDetection) project. Where
that project answered *"what is in front of the robot and which way should it turn?"*,
this project adds *"where is the robot itself, and how is it moving?"* — using the same
camera it already carries.

---

## Motivation

The existing `ObjectDetection` project runs YOLOv8 (via OpenVINO) on a camera stream to
detect objects and emit navigation hints. It has no sense of **ego-motion**: it does not
know how the robot is translating or rotating in the world.

Two things change here:

1. **An IMU is added** to the Orin board — an **MPU-6050** (3-axis accelerometer +
   3-axis gyroscope) over **I2C**. This gives high-rate motion measurements between
   camera frames.
2. **The camera stream is reused.** Instead of adding a second camera for pose tracking,
   the *same* video stream feeds two networks/pipelines concurrently:
   - an **object-detection** pipeline (what the current project already does), and
   - a **state-estimation** pipeline (visual-inertial odometry / pose).

The result is a robot that can both perceive obstacles/targets **and** localize and track
its own 6-DoF motion — without extra optical hardware.

**A note on odometry vs. SLAM.** Visual-inertial odometry gives pose **relative to the
starting point**, and its error **accumulates without bound** — good for *"how have I moved
since I started,"* insufficient for navigating an environment over time. Actual navigation
needs **SLAM**: a persistent **map**, **loop closure** to bound the drift, and
**relocalization** to place the robot in a previously built map. This project therefore
treats VIO as the **odometry front-end** and adds a **SLAM back-end** on top of it. VIO is
built and validated **first** (Phase 1); SLAM is layered on as **Phase 2** — a validated
odometry source de-risks the map, since most SLAM failures trace back to bad odometry.

---

## Goals

- **Shared camera, two consumers.** One capture source, fanned out to two pipelines with
  minimal copies and no frame contention.
- **Visual-inertial state estimation (Phase 1).** Fuse camera + IMU into a 6-DoF pose
  estimate (position, orientation) and velocity, robust to the ~30 FPS camera rate by
  leaning on the high-rate IMU between frames. This is odometry — pose relative to start.
- **SLAM for navigation (Phase 2).** Build and persist a map, close loops to bound drift,
  and relocalize within a saved map, so a planner can navigate the environment.
- **Keep object detection working.** No regression in the detection/navigation behavior
  inherited from `ObjectDetection`.
- **Run on the Orin in real time.** Budget the two pipelines so both keep up on the
  Jetson's GPU/DLA/CPU without starving each other.

## Non-Goals (initial phase)

- SLAM in **Phase 1**. The first milestone is **odometry** (VIO) only — drift-prone but
  simpler. Full SLAM (map building, loop closure, relocalization, map persistence) is a
  planned **Phase 2** that builds on the validated VIO front-end, not out of scope.
- Multi-camera or stereo rigs. This is deliberately a **single-camera** design.
- Retraining detection models. Reuse the existing YOLOv8n / OpenVINO assets.
- Path planning / autonomous navigation itself. This project provides the map and pose that
  a navigation stack (e.g. Nav2) would consume; the planner is downstream.

---

## System Overview

```
                 ┌───────────────────────────────────────────────┐
                 │                 Jetson Orin                    │
                 │                                                │
   Camera  ──────┤  ┌───────────────┐                             │
  (single)       │  │ Frame Source  │  one capture, fanned out    │
                 │  │  / Publisher  │────────────┬───────────────┐│
                 │  └───────────────┘            │               ││
                 │                               ▼               ▼│
                 │                     ┌──────────────┐  ┌──────────────────┐
                 │                     │  Object      │  │  State Estimation │
                 │                     │  Detection   │  │  (Visual-Inertial │
                 │                     │  (YOLOv8 /   │  │   Odometry)       │
                 │                     │   OpenVINO)  │  │                   │
                 │                     └──────┬───────┘  └─────────┬────────┘
                 │                            │                    │
   IMU  ─────────┼────────────────────────────────────────────────┤
 (accel/gyro)    │                            │          fuses IMU + vision
                 │                            ▼                    ▼
                 │                   detections + nav      6-DoF pose + velocity
                 └───────────────────────────────────────────────┘
```

### Components

1. **Frame Source / Publisher.**
   Owns the single camera capture. Timestamps each frame and hands it to both consumers
   (e.g. shared-memory ring buffer, a small pub/sub bus, or a bounded queue per consumer).
   Design intent: each pipeline reads frames without blocking the other, and a slow
   consumer drops frames rather than stalling capture.

2. **Object-Detection Pipeline.**
   Ported from `ObjectDetection` (`robot_detection*.py`): YOLOv8n exported to OpenVINO,
   producing bounding boxes + navigation hints. Consumes frames from the publisher instead
   of opening the camera directly.

3. **IMU Driver.**
   Reads the newly added IMU on the Orin at high rate (target: hundreds of Hz). Provides
   timestamped angular velocity and linear acceleration; calibration (bias, alignment)
   handled here or in the estimator.

4. **State-Estimation Pipeline (Visual-Inertial Odometry).**
   Consumes the same camera frames plus IMU samples and produces a 6-DoF pose and velocity
   estimate. Vision supplies drift-corrected structure; the IMU supplies high-rate motion
   and scale/gravity observability between frames.

5. **Time Synchronization.**
   A shared, monotonic clock timestamps camera frames and IMU samples so the estimator can
   correctly interleave them. This is critical — VIO accuracy is very sensitive to
   camera↔IMU time offset.

---

## Key Design Questions (to resolve before coding)

- **How is the frame shared?** Shared-memory ring buffer vs. per-consumer bounded queues
  vs. a lightweight message bus. Zero/low-copy matters on the Orin.
- **Which VIO approach?** Options span classic (filter-based, e.g. MSCKF-style; or
  optimization-based, e.g. VINS-Mono / OpenVINS-style) through to learning-based
  visual-inertial odometry. Trade-off: accuracy vs. Orin compute budget vs. integration effort.
- **Compute placement.** Detection and VIO both want the accelerators. Decide GPU vs. DLA
  vs. CPU per stage, and the frame-rate budget each pipeline gets.
- **Camera model & calibration.** Intrinsics, distortion, and camera↔IMU extrinsics +
  time offset must be calibrated once and stored.
- **Sync strategy.** Hardware timestamps vs. software timestamps; how to estimate/track the
  camera–IMU time offset.
- **Outputs & consumers.** What downstream consumes the pose (navigation, logging, a fusion
  node)? Define the interface early.

---

## Hardware

- **Compute:** NVIDIA Jetson Orin.
- **Camera:** single camera (reused for both pipelines) — same class of device as the
  `ObjectDetection` project (RGB webcam; RealSense optional).
- **IMU:** **InvenSense MPU-6050**, newly added to the Orin. 6-DoF (3-axis
  accelerometer + 3-axis gyroscope; no magnetometer). Connected over **I2C**.

## Software Environment

- **ROS 2 runs in a Docker container on the Orin.** This is the natural home for the
  frame publisher, IMU driver, detection, and state-estimation nodes: ROS 2 topics provide
  the pub/sub fan-out of the shared camera stream to both consumers, message timestamps
  support the required time synchronization, and standard message types
  (`sensor_msgs/Image`, `sensor_msgs/Imu`, `nav_msgs/Odometry`, TF) give ready-made
  interfaces between stages. Containerizing ROS 2 keeps its dependencies isolated from the
  host JetPack environment. Camera and IMU devices must be passed through to the container,
  and any existing VIO packages (e.g. Isaac ROS Visual SLAM, OpenVINS) can be layered into
  the same image.

---

## Relationship to `ObjectDetection`

| | `ObjectDetection` | `ObjectDetectionAndStateEstimation` |
|---|---|---|
| Camera | one, one consumer | one, **two** consumers |
| IMU | none | **added**, fused for pose |
| Detection | YOLOv8 + OpenVINO | **reused** |
| Ego-motion | none | **6-DoF pose + velocity (VIO)** |
| Platform | Jetson Orin | Jetson Orin |

The detection code is intended to be carried over largely intact; the new work is the
frame publisher, the IMU driver, the state-estimation pipeline, and time synchronization.

---

## Proposed Roadmap

1. **Frame publisher.** One capture → two consumers; validate no frame contention and
   measure per-consumer FPS on the Orin.
2. **Detection on shared frames.** Re-point the existing detection pipeline at the
   publisher; confirm parity with `ObjectDetection`.
3. **IMU driver + calibration.** Bring the IMU online; log synchronized camera + IMU data.
4. **Camera–IMU calibration.** Intrinsics, extrinsics, and time offset.
5. **State estimation (VIO) — Phase 1.** Integrate a VIO approach; validate pose against a
   known trajectory. Delivers drift-prone odometry relative to the start.
6. **Joint run + budgeting.** Both pipelines concurrently on the Orin in real time;
   profile and tune the compute split.
7. **SLAM back-end — Phase 2.** Layer a SLAM stack on top of the VIO front-end: build and
   persist a map, add loop closure to bound drift, and support relocalization in a saved
   map. Publish the `map → odom` correction so the pose is globally consistent.

---

## TODO — Nodes to Build

The system is decomposed into the following ROS 2 nodes:

1. **IMU node.** Reads data from the MPU-6050 over I2C and publishes it (e.g.
   `sensor_msgs/Imu`).
2. **Object-detection network.** Reads the camera stream and detects objects (the YOLOv8 /
   OpenVINO pipeline carried over from `ObjectDetection`).
3. **Visual pose network.** A second network that uses visual cues from the same camera
   stream to estimate the camera's pose.
4. **Fusion node.** Combines the IMU data with the camera-derived pose.
5. **Robot-pose node.** Outputs the final robot pose after fusing IMU and camera data —
   this is the `odom → base_link` odometry (e.g. `nav_msgs/Odometry` + TF).
6. **SLAM node (Phase 2).** Consumes VIO odometry + camera (+ IMU), builds and persists a
   map, closes loops, relocalizes, and publishes the `map → odom` correction so the pose is
   globally consistent and usable for navigation.

---

> This document describes intent and architecture only. No implementation exists yet —
> source code, environment setup, and per-module docs will follow once the design questions
> above are settled.
