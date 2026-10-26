# Implementation Prompt — ObjectDetectionAndStateEstimation

Use this prompt to drive the implementation of the robot described in [`DESIGN.md`](./DESIGN.md).
It lists every ROS 2 package to create, the nodes inside them, their topics/message types,
and the order in which to build and verify them.

---

## Context (read `DESIGN.md` first)

Build a ROS 2 system that runs on an **NVIDIA Jetson Orin**. A **single camera** stream is
fanned out to **two** consumers — an object-detection network and a visual pose network —
and the visual pose is fused with an **MPU-6050 IMU** (connected over **I2C**) to produce a
6-DoF robot pose. **ROS 2 runs inside a Docker container** on the Orin; the camera and the
I2C bus must be passed through to the container.

**Hard constraints:**
- Single camera, one capture, shared by both pipelines — no second camera, no frame contention.
- Reuse the existing YOLOv8n + OpenVINO detection assets from the sibling `ObjectDetection`
  project; do not retrain.
- Real-time on the Orin: both pipelines must keep up. Budget GPU/DLA/CPU per stage.
- **Phased scope.** Phase 1 delivers **odometry** (VIO) — pose relative to start, drift-prone.
  Phase 2 layers a **SLAM back-end** on the VIO front-end: map building, loop closure,
  relocalization, and map persistence, publishing the `map → odom` correction. Build and
  validate VIO first; SLAM builds on it.
- All inter-node data uses standard ROS 2 messages where possible: `sensor_msgs/Image`,
  `sensor_msgs/CameraInfo`, `sensor_msgs/Imu`, `nav_msgs/Odometry`, `geometry_msgs/PoseWithCovarianceStamped`,
  `vision_msgs/Detection2DArray`, and TF2.
- Time synchronization is critical: one monotonic clock timestamps every camera frame and
  IMU sample so the estimator can interleave them.

---

## Workspace & Package Layout

Create a ROS 2 (target: **Humble** or the version shipped in the Orin container) colcon
workspace. Use the prefix `odse_` (ObjectDetection**A**nd**S**tate**E**stimation).

```
odse_ws/
└── src/
    ├── odse_bringup/          # launch files, params, orchestration, RViz configs
    ├── odse_interfaces/       # custom msgs/srvs (only what standard msgs don't cover)
    ├── odse_camera/           # single-camera frame source / publisher node
    ├── odse_imu/              # MPU-6050 I2C driver node
    ├── odse_detection/        # YOLOv8/OpenVINO object-detection node
    ├── odse_visual_odometry/  # visual pose network node
    ├── odse_fusion/           # IMU + visual-pose fusion node (EKF/UKF)
    ├── odse_state_publisher/  # odom->base_link pose output node (Odometry + TF)
    ├── odse_slam/             # Phase 2: SLAM back-end (map, loop closure, map->odom)
    └── odse_calibration/      # camera intrinsics + camera↔IMU extrinsics/time-offset tools
docker/                        # Dockerfile + compose + device passthrough for the Orin
```

---

## Packages to Create

### 1. `odse_camera` — Frame Source / Publisher
- **Purpose:** own the single camera capture; publish timestamped frames once, consumed by
  both downstream pipelines.
- **Node:** `camera_publisher`
- **Publishes:** `/camera/image_raw` (`sensor_msgs/Image`), `/camera/camera_info`
  (`sensor_msgs/CameraInfo`).
- **Requirements:** stamp every frame from a single monotonic clock; use a QoS / bounded
  buffer so a slow consumer drops frames rather than stalling capture; support the same
  camera class as `ObjectDetection` (RGB webcam; RealSense optional). Consider ROS 2
  intra-process comms / zero-copy to avoid double-copying frames on the Orin.
- **Deliverable check:** both a detection subscriber and a VO subscriber receive frames
  concurrently with no contention; measure per-consumer FPS.

### 2. `odse_imu` — MPU-6050 I2C Driver
- **Purpose:** read the MPU-6050 over I2C at high rate and publish IMU data.
- **Node:** `mpu6050_driver`
- **Publishes:** `/imu/data_raw` (`sensor_msgs/Imu` — angular velocity + linear acceleration).
- **Requirements:** configurable I2C bus/address (default `0x68`); target hundreds of Hz;
  configurable accel/gyro full-scale ranges and DLPF; publish covariances; timestamp from
  the same monotonic clock as the camera. Provide a bias/offset calibration routine (store
  results to a params file).
- **Deliverable check:** stable, correctly-scaled data at the configured rate; gravity reads
  ~9.81 m/s² on the up axis at rest.

### 3. `odse_detection` — Object-Detection Network
- **Purpose:** port the YOLOv8n + OpenVINO pipeline from `ObjectDetection` to consume shared
  frames instead of opening the camera directly.
- **Node:** `object_detector`
- **Subscribes:** `/camera/image_raw`.
- **Publishes:** `/detections` (`vision_msgs/Detection2DArray`), optional annotated
  `/detections/image` and the existing navigation hint.
- **Requirements:** reuse existing OpenVINO IR (`yolov8n_openvino_model`); frame-skip / rate
  control so it does not starve the VO pipeline; decide GPU vs. DLA vs. CPU placement.
- **Deliverable check:** detection parity with `ObjectDetection` on the same input.

### 4. `odse_visual_odometry` — Visual Pose Network
- **Purpose:** the second network — estimate the camera's 6-DoF pose from visual cues on the
  same shared stream.
- **Node:** `visual_odometry`
- **Subscribes:** `/camera/image_raw`, `/camera/camera_info`.
- **Publishes:** `/vo/pose` (`geometry_msgs/PoseWithCovarianceStamped`) and/or `/vo/odometry`
  (`nav_msgs/Odometry`).
- **Requirements:** pick and integrate a VO/VIO approach (evaluate: classic feature-based VO,
  OpenVINS, VINS-Fusion, or NVIDIA Isaac ROS Visual SLAM — trade accuracy vs. Orin compute).
  Output realistic covariances. Use calibrated intrinsics from `odse_calibration`.
- **Deliverable check:** pose tracks a known trajectory within acceptable drift.

### 5. `odse_fusion` — IMU + Visual-Pose Fusion
- **Purpose:** fuse high-rate IMU with the lower-rate visual pose into a single state estimate.
- **Node:** `state_fusion` (EKF/UKF — e.g. `robot_localization`, or a custom filter, or the
  tightly-coupled estimator inside the chosen VIO stack).
- **Subscribes:** `/imu/data_raw` (or bias-corrected `/imu/data`), `/vo/pose` (or `/vo/odometry`).
- **Publishes:** `/fusion/odometry` (`nav_msgs/Odometry`) with covariance.
- **Requirements:** correct time interleaving using synchronized stamps; apply the camera↔IMU
  extrinsics and time offset from calibration; handle vision dropouts by propagating on IMU alone.
- **Deliverable check:** smoother, lower-drift pose than VO alone; survives brief vision loss.

### 6. `odse_state_publisher` — Robot Pose Output
- **Purpose:** publish the fused odometry and the `odom → base_link` TF consumers rely on.
- **Node:** `robot_pose_publisher`
- **Subscribes:** `/fusion/odometry`.
- **Publishes:** `/robot/odometry` (`nav_msgs/Odometry`) and TF `odom → base_link`
  (per REP-105); optional `/robot/pose` (`geometry_msgs/PoseStamped`).
- **Deliverable check:** correct, continuous TF visualizable in RViz.

### 6b. `odse_slam` — SLAM Back-End (Phase 2)
- **Purpose:** turn drift-prone odometry into a globally consistent, navigable estimate by
  building a map, closing loops, and relocalizing.
- **Node:** `slam` (or the launch-wrapped node of the chosen stack).
- **Subscribes:** `/robot/odometry` (or `/fusion/odometry`) as the odometry source,
  `/camera/image_raw` + `/camera/camera_info`, and `/imu/data_raw` if the stack uses it.
- **Publishes:** the map (e.g. `/map`, point cloud / occupancy grid) and the TF
  **`map → odom`** correction; supports **map save/load** for persistence and
  **relocalization** into a saved map.
- **Requirements:** pick a Jetson-friendly stack — **RTAB-Map**, **NVIDIA Isaac ROS
  cuVSLAM/nvblox**, or **ORB-SLAM3** — that takes external odometry as its front-end, closes
  loops, and persists maps. Own only `map → odom`; leave `odom → base_link` to
  `odse_state_publisher` (REP-105).
- **Deliverable check:** revisiting a place triggers loop closure and corrects accumulated
  drift; a saved map reloads and the robot relocalizes into it.

### 7. `odse_calibration` — Calibration Tooling
- **Purpose:** produce and store the parameters the estimator needs.
- **Deliverables:** camera intrinsics + distortion (e.g. `camera_calibration` / checkerboard),
  camera↔IMU extrinsics and time offset (e.g. Kalibr), saved as YAML loaded by
  `odse_camera` / `odse_visual_odometry` / `odse_fusion`.

### 8. `odse_interfaces` — Custom Messages
- **Purpose:** only for data standard messages don't cover (e.g. a combined detections+pose
  message, or diagnostics). Prefer standard `sensor_msgs` / `vision_msgs` / `nav_msgs` first.

### 9. `odse_bringup` — Orchestration
- **Purpose:** launch the whole graph and hold configuration.
- **Deliverables:** `robot.launch.py` bringing up all nodes; per-node param YAMLs; an RViz
  config; smaller launch files for staged bring-up (camera-only, camera+imu, detection-only,
  vo-only, full).

### 10. `docker/` — Container on the Orin
- **Deliverables:** a Dockerfile based on a JetPack/ROS 2-for-Jetson base image with
  OpenVINO/CUDA and the chosen VIO stack layered in; docker-compose (or run script) that
  passes through the camera (`/dev/video*`) and I2C bus (`/dev/i2c-*`), grants the needed
  device permissions, and mounts the workspace; build/run instructions.

---

## Build Order (Steps)

Follow the roadmap in `DESIGN.md`, package by package:

1. **Container + workspace.** Build `docker/`; get a ROS 2 shell in the container with camera
   and I2C passthrough working. Create the empty colcon workspace and all packages; confirm
   `colcon build` succeeds.
2. **Camera publisher (`odse_camera`).** One capture → two subscribers; verify no contention
   and measure per-consumer FPS.
3. **Detection on shared frames (`odse_detection`).** Re-point the ported pipeline at
   `/camera/image_raw`; confirm parity with `ObjectDetection`.
4. **IMU driver (`odse_imu`).** Bring the MPU-6050 online over I2C; log synchronized
   camera + IMU data; run bias calibration.
5. **Calibration (`odse_calibration`).** Camera intrinsics, then camera↔IMU extrinsics and
   time offset; store YAMLs.
6. **Visual odometry (`odse_visual_odometry`).** Integrate the chosen VIO approach; validate
   pose against a known trajectory.
7. **Fusion (`odse_fusion`) + output (`odse_state_publisher`).** Fuse IMU + VO; publish
   `/robot/odometry` and the `odom → base_link` TF. **End of Phase 1 (odometry).**
8. **Joint run + budgeting (`odse_bringup`).** Launch the full graph; run both pipelines
   concurrently in real time on the Orin; profile and tune the GPU/DLA/CPU split.
9. **SLAM back-end (`odse_slam`) — Phase 2.** Layer the chosen SLAM stack on the VIO
   odometry; verify loop closure corrects drift and that maps save, reload, and relocalize;
   publish `map → odom`. Re-budget compute now that SLAM shares the accelerators.

---

## Cross-Cutting Requirements

- **Time sync:** single monotonic clock stamps all frames and IMU samples; track/apply the
  camera–IMU time offset. Use `message_filters` where streams must be paired.
- **Frames/TF:** follow REP-103 (units/axes) and REP-105 (`map`/`odom`/`base_link`).
- **QoS:** sensor streams use sensor-data QoS; size buffers so slow consumers drop, not block.
- **Params:** every tunable (I2C address, rates, thresholds, model paths, calibration files)
  is a ROS 2 parameter loaded from `odse_bringup` YAML.
- **Verification per package:** each package ships a minimal test/launch proving its
  deliverable check above before moving on.

---

## Deliverables Summary

A working `odse_ws` with the packages above, launched via `odse_bringup/robot.launch.py`
inside the Orin Docker container:
- **Phase 1:** live `/detections`, a fused `/robot/odometry` with `odom → base_link` TF, and
  both pipelines running concurrently in real time from a single camera.
- **Phase 2 (`odse_slam`):** a persistent map with loop closure and relocalization, publishing
  the `map → odom` correction — a globally consistent pose ready for a downstream navigation
  stack (e.g. Nav2).
