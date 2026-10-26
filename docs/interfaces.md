# Interfaces of the driving chain

Specification of every module in the chain from camera to motors: inputs,
outputs, units, conventions and rates. The training simulator
(`policy/lego_car_env.py`) must produce exactly what the real chain feeds the
policy network, so section 4 compares the two field by field, and section 5
lists where they differ today.

Status: October 2026. Modules marked *planned* do not exist yet; their
interface here is the specification to build them against.

## 1. The chain

```
 Camera ──images──▶ YOLO ──boxes──▶ Target observation ──9 numbers──▶ Policy ──steering, speed──▶ Hub bridge ──Bluetooth──▶ LEGO hub ──▶ motors
  M1                 M2               M3                                M4                           M5                         M6
```

| # | Module | Training (steps 2 to 4) | Gazebo (step 5) | Real car (step 6) |
|---|---|---|---|---|
| M1 | Camera | simulated geometry in `lego_car_env.py` | Gazebo cameras (`odse_sim`) | F200 driver node, *planned* |
| M2 | YOLO detector | simulated ("fake YOLO") | YOLO node (`yolo_node.py`, runs on the Orin) | same YOLO node |
| M3 | Target observation | `LegoCarEnv._observe()` | target node (`target_node.py`); `sim_target` as a stand in from true positions | same target node |
| M4 | Policy | PyTorch network in the training loop | policy node with the ONNX file | same, with TensorRT on the Orin |
| M5 | Command output | `LegoCarEnv._drive()` (car model) | sim adapter to `/cmd_vel` | hub bridge node, *planned* |
| M6 | Hub and motors | car model | Gazebo Ackermann plugin | LEGO Technic hub, motors A, B, D |

In Gazebo and on the real car, every module is a ROS 2 node and the arrows are
ROS 2 topics. During training there is no ROS 2: environment and network run
in one Python program.

A second sensor branch runs alongside: the **IMU** (M7), an MPU-6050 on the
Orin's I2C bus. It feeds the state estimation of `DESIGN.md` (visual inertial
odometry, fusion, robot pose), **not** the driving policy so far. Section 3
specifies it; section 5 (F15) discusses giving the policy the yaw rate.

| # | Module | Training | Gazebo | Real car |
|---|---|---|---|---|
| M7 | IMU | not modeled | IMU sensor in `odse_sim` (verified, 200 Hz) | MPU-6050 driver node, *planned* |

## 2. Conventions

| Topic | Convention |
|---|---|
| Car frame | `base_link` on the floor below the center of the rear axle; x forward, y left, z up (REP-103) |
| Image coordinates | pixels from the top left corner; u to the right, v down |
| Normalized image coordinates | x and y from -1 to 1: **x = -1 left edge, +1 right edge; y = -1 top, +1 bottom**. Sizes as a fraction of the image width or height |
| Policy steering | **+1 = full right, -1 = full left** |
| ROS steering angle (`ackermann_msgs`) | radians, **positive = left** (REP-103), so `steering_angle = -steer * 0.395` |
| Speed | policy: fraction of motor power, -1 to 1; ROS: m/s |
| Distance units | meters everywhere in the chain; the old ObjectDetection code used mm |
| Time stamps | every message carries the stamp of the camera image it is based on |
| Middleware | Zenoh (`rmw_zenoh_cpp`), see `docs/rmw_comparison.md` |
| Rate | policy steps at **10 Hz** (one command per 0.1 s); everything upstream must deliver at least that |

## 3. Modules

### M1 Camera

| | |
|---|---|
| Real | RealSense F200, driver node *planned* (V4L2; `librealsense2` does not support the F200, see `docs/camera_f200.md`) |
| Gazebo | color and depth sensors in `odse_sim/urdf/lego_car.urdf.xacro` |

Outputs:

| Topic | Type | Content | Gazebo (verified) | Real F200 |
|---|---|---|---|---|
| `/camera/color/image_raw` | `sensor_msgs/msg/Image` | color image, frame `camera_color_optical_frame` | 640×360 `rgb8`, 70° × 43°, 30 Hz | up to 1920×1080, **resolution and field of view of the stream used are still to be verified** |
| `/camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | intrinsics of the color image | yes | from calibration |
| `/camera/depth/image_rect_raw` | `sensor_msgs/msg/Image` | depth, frame `camera_depth_optical_frame` | 640×480 `32FC1` in m, 68° × 54°, 0.2 to 1.0 m, 30 Hz | 640×480, native 16 bit (`16UC1`, mm expected) |
| `/camera/depth/camera_info` | `sensor_msgs/msg/CameraInfo` | intrinsics of the depth image | yes | from calibration |

Color and depth are **two different cameras** with different fields of view
(and, on the real F200, a few cm apart). A pixel in the color image is not the
same pixel in the depth image; M3 handles the mapping.

### M2 YOLO detector

| | |
|---|---|
| Input | `/camera/color/image_raw` |
| Output | `/detections`, `vision_msgs/msg/Detection2DArray` |
| Node | `odse_driving/yolo_node.py`, detector code in `yolo.py` (numpy, OpenCV and ONNX Runtime only; same boxes as ultralytics) |
| Model | YOLOv8n as ONNX with a fixed 384 × 640 input (`models/yolov8n_384x640.onnx`, made by `scripts/export_yolo_onnx.py`), ONNX Runtime on the Orin's CPU; TensorRT engine later |
| Rate | at least 10 Hz. Measured on the Orin's CPU: 87 ms alone, about 120 ms with the other nodes running (F25) |
| Confidence threshold | 0.25 (`robot_detection.py` uses 0.5; the Gazebo ball often scores between the two, F7) |

The subscription keeps only the newest image: when YOLO is slower than the
camera, frames are skipped, never queued.

Per detection in `/detections`:

| Field | Meaning |
|---|---|
| `header` | copied from the input image (stamp and frame) |
| `results[0].hypothesis.class_id` | COCO class name, for example `sports ball`, `person` |
| `results[0].hypothesis.score` | confidence, 0 to 1 |
| `bbox.center.position.x`, `.y` | box center in **pixels** of the color image |
| `bbox.size_x`, `size_y` | box width and height in **pixels** |

All detected objects are published; choosing the target is M3's job.

### M3 Target observation

Turns YOLO boxes and the depth image into the observation the policy was
trained on.

| | |
|---|---|
| Inputs | `/detections`, `/camera/depth/image_rect_raw`, `/camera/color/camera_info`, `/camera/depth/camera_info` |
| Parameter | `target_class` (COCO name) |
| Output | `/target`, `odse_interfaces/msg/TargetObservation` (custom message, below) |
| Rate | one message per `/detections` message |

Processing:

1. **Select** the detection of `target_class` with the **highest score**. None:
   publish `visible = false`.
2. **Normalize** with the color image size W × H:
   `x = 2 u / W - 1`, `y = 2 v / H - 1`, `w = box width / W`, `h = box height / H`.
3. **Depth:** map the central 20 % of the box into the depth image using both
   camera infos (and the extrinsics between the two cameras on the real F200),
   take the median of the valid depth values, convert to meters.
   `z_valid` only if the value is between 0.2 and 1.0 m **and** the depth image
   is at most 0.2 s older than the color image.

4. **Depth hold** (F23): YOLO loses the ball when it is close and fills the
   image. If the target had a valid depth within the last `hold_for` seconds
   (parameter, 1.0 s; 0 switches it off) and YOLO misses it now, measure the
   depth again where the box was. If it moved less than 0.15 m, publish the
   target as visible, with the box scaled to the new distance (offsets from
   the image center and sizes times old depth / new depth) and `score = 0`
   to mark the held observation. Each such reading renews the hold.

Implemented in `odse_driving/target_node.py`. In Gazebo both cameras sit at
the same point, so step 3 maps with the two camera infos alone; the real F200
also needs the extrinsics.

Message `odse_interfaces/msg/TargetObservation`:

```
std_msgs/Header header   # stamp of the color image
bool visible
float32 x                # -1 left .. 1 right
float32 y                # -1 top .. 1 bottom
float32 w                # box width / image width
float32 h                # box height / image height
bool z_valid
float32 z                # m, distance to the object surface; 0 if not valid
float32 score            # YOLO confidence
string class_id
```

When `visible` is false, x, y, w, h, z are 0 and `z_valid` is false.

### M4 Policy

| | |
|---|---|
| Input | `/target`, plus its own previous command |
| Output | `/drive/command`, `ackermann_msgs/msg/AckermannDriveStamped` |
| Model | ONNX file exported from training; ONNX Runtime on the laptop, TensorRT on the Orin |
| Rate | 10 Hz timer, uses the latest `/target` |

Network input, `float32[1, 9]`, ONNX input name `obs`, in exactly this order
(`lego_car_env.OBS_NAMES`):

| Index | Name | From |
|---|---|---|
| 0 | `visible` | `/target.visible` as 0 or 1 |
| 1 | `x` | `/target.x` |
| 2 | `y` | `/target.y` |
| 3 | `w` | `/target.w` |
| 4 | `h` | `/target.h` |
| 5 | `z_valid` | `/target.z_valid` as 0 or 1 |
| 6 | `z` | `/target.z` |
| 7 | `prev_steer` | own previous output |
| 8 | `prev_speed` | own previous output |

If the latest `/target` is older than 0.5 s, the node uses `visible = 0`.

Network output, `float32[1, 2]`, ONNX output name `action`, each clipped to
-1 .. 1: `[steer, speed]` (steer +1 = right; speed = fraction of motor power).

**Speed limit:** the node clips `speed` to **±0.4** (40 % motor power) before
publishing, exactly as the training simulator does (`SPEED_LIMIT` in
`lego_car_env.py`). This is required: the PPO network asks for 1.0 most of the
time and relies on the clip. `prev_speed` in the next observation is the
network's raw output, not the clipped value, as in training. The limit stays
until faster speeds are measured on the car (F8).

Published as `/drive/command`:

| Field | Value |
|---|---|
| `header.stamp` | stamp of the `/target` used |
| `drive.steering_angle` | `-steer * 0.395` rad (positive = left) |
| `drive.speed` | `speed * 1.0` m/s |

Physical units keep the command independent of the hardware; M5 converts back.

### M5a Sim adapter, Gazebo

| | |
|---|---|
| Input | `/drive/command` |
| Output | `/cmd_vel`, `geometry_msgs/msg/Twist` (bridged into Gazebo, `odse_sim/config/bridge.yaml`) |

`linear.x = speed`, `angular.z = speed * tan(steering_angle) / 0.25`
(wheelbase 0.25 m). Gazebo's Ackermann plugin turns this back into a steering
angle. Limitation: at speed 0 the steering command is lost, so the simulated
wheels cannot pre-steer while standing.

### M5b Hub bridge, real car (*planned*)

| | |
|---|---|
| Input | `/drive/command` |
| Output | Bluetooth LE (LWP3) commands to the hub, via `LegoTechnicHub/technic_hub` |
| Rate | at most 10 Hz; only send when a value changed noticeably (each update is 3 Bluetooth writes) |

Conversion, using the settings from `LegoTechnicHub/examples/navigation_hints.py`:

| Output | Formula | Values |
|---|---|---|
| steering motor **D** | `run_to_position(center - STEER_LEFT_SIGN * steer * steer_deg)` with `steer = -steering_angle / 0.395` | `center` and `steer_deg` (80 % of half the range, about 89°) from `calibrate_steering()` at startup; `STEER_LEFT_SIGN = -1` |
| drive motors **A** (rear), **B** (front) | `run(DRIVE[port] * percent)` with `percent = round(speed / 1.0 * 100)`, clipped to **±40** (the speed limit, again here as a safety net) | `DRIVE = {"A": 1, "B": 1}` |
| stop | `speed == 0`: `stop()` on A and B | |
| watchdog | no `/drive/command` for 0.5 s: stop A and B | |

### M6 LEGO hub and motors

| Motor | Port | Command | Measured |
|---|---|---|---|
| steering | D | position, ±89° around the calibrated center | at 89° from the center (full left; the motor turns backward for left, `STEER_LEFT_SIGN = -1`) the car drives a **1.20 m circle**, so 22.6° at the wheels |
| drive, rear | A | regulated speed, % | 30 % gives **0.31 m/s** |
| drive, front | B | regulated speed, % | same |

### M7 IMU (*planned* as a node)

| | |
|---|---|
| Hardware | InvenSense **MPU-6050**: 3 axis accelerometer and 3 axis gyroscope, no magnetometer, on the Orin's **I2C** bus (`DESIGN.md`) |
| Real | driver node `mpu6050_driver` in package `odse_imu` (`prompt.md`), *planned* |
| Gazebo | IMU sensor in `odse_sim/urdf/lego_car.urdf.xacro`, link `imu_link`, verified at 200 Hz |
| Output | `/imu/data_raw`, `sensor_msgs/msg/Imu` |
| Rate | 200 Hz (Gazebo); real target several hundred Hz (`DESIGN.md`) |
| Consumers | state estimation (visual inertial odometry, fusion node, robot pose), see `DESIGN.md`; not the policy so far |

Fields of `/imu/data_raw` (REP-145 for raw IMU data):

| Field | Content |
|---|---|
| `header.stamp` | time of the reading, same clock as the camera images (important for fusion) |
| `header.frame_id` | `imu_link` |
| `angular_velocity` | rad/s, in `imu_link` axes |
| `linear_acceleration` | m/s², in `imu_link` axes; reads about +9.81 on the up axis at rest |
| `orientation` | not provided (no fusion in the driver): `orientation_covariance[0] = -1` |
| covariances | from the sensor's noise figures, measured at rest |

`imu_link` has to be placed in the robot model where the chip actually sits,
with its axes rotated so they match the chip's printed axes. Gazebo currently
assumes the car's center between the axles, 8 cm high, axes aligned with
`base_link`; both are assumptions until measured (F16).

## 4. Training simulator versus real chain

What `LegoCarEnv._observe()` computes for each input of the policy, compared
with what M2 and M3 deliver.

| Input | Real chain (M2 + M3) | Training simulator | Same? |
|---|---|---|---|
| `visible` | target class detected with score ≥ 0.5 | object in the color field of view and box ≥ 10 px high; random misses of 0 to 15 % | yes, if YOLO's real miss rate is in that range |
| `x` | `2 u / W - 1` from the box center | `-(lateral / forward) / tan(35°)`, plus noise 0.02 | yes, for a pinhole camera with 70° horizontal view |
| `y` | `2 v / H - 1` | `(camera height - object radius) / forward / tan(21.5°)`, plus noise | yes, same condition |
| `w`, `h` | box size / image size | object diameter / forward / (2 tan(half view)), ±5 % noise | yes for round objects; other shapes give other boxes |
| `z_valid` | depth found at the box, 0.2 to 1.0 m, fresh | object in the 68° depth view and 0.2 to 1.0 m away | yes, if depth is aligned and fresh (see F3 to F5) |
| `z` | median depth in the box, m | forward distance minus radius, noise 0.01 m | yes, if aligned and accurate |
| `prev_steer`, `prev_speed` | policy's own previous output | previous action | yes |
| delay | YOLO time + Bluetooth delay | 0 to 0.2 s, random per episode | to be measured (F9) |
| command → motion | hub bridge, motors | bicycle model with measured wheelbase, steering limit and speed | yes, within the measured points (F8) |

## 5. Verification findings

Checked on 8 October 2026 against `ObjectDetection/robot_detection.py`,
`LegoTechnicHub/examples/navigation_hints.py`, `odse_sim` and
`policy/lego_car_env.py`.

| # | Finding | Effect | Status |
|---|---|---|---|
| F1 | `robot_detection.py` has **no numeric output**: it only draws boxes and shows hint strings (`TURN LEFT`, `APPROACH`, ...) on screen | nothing to feed into the policy | **fixed**: M2 and M3 nodes (`yolo_node.py`, `target_node.py`) |
| F2 | The simulator computed boxes with the **depth** camera's field of view (68° × 54°), but YOLO runs on the **color** camera (70° × 43°) | x, y, w, h slightly wrong in training | **fixed** in `lego_car_env.py`; the rule based controller still arrives in 200 of 200 episodes |
| F3 | `robot_detection.py` reads the depth at the **color** pixel coordinates of the box center, without mapping between the two cameras | depth taken from the wrong place, especially away from the image center | open: M3 maps between the cameras |
| F4 | Depth arrives as **8 bit** on Windows and is scaled linearly to 0 to 1200 mm | coarse (about 5 mm steps at best) and not calibrated; the simulator assumes about 1 cm noise | open: read 16 bit through V4L2 on the Orin |
| F5 | Depth runs at **about 1 FPS** on Windows and is reused for up to 2 s | z can be 2 s old; the simulator assumes fresh depth every 0.1 s | open: V4L2 on the Orin; M3 rejects depth older than 0.2 s |
| F6 | With several boxes of the target class, `robot_detection.py` keeps the **last** one | target may jump between objects | **fixed**: M3 takes the highest score |
| F7 | The simulator uses round objects of 12 to 30 cm; YOLO needs a **COCO class**, and the Gazebo world has none (red box, blue cylinder, green ball are plain shapes) | Gazebo test impossible as is | **mostly fixed** (23 October 2026): YOLOv8n calls the plain orange `target_ball` a `sports ball`, with scores of 0.1 to 0.9, often below 0.5. Over 63 poses (0.35 to 4 m, ±30°) it finds it in 65 % at threshold 0.25, all of them from 1.2 to 2.2 m, few below 0.6 m (F23). Where found, the box matches the training camera model to 0.02 in x and y and about 8 % in size. A soccer ball texture scored worse (33 %), YOLOv8s did not help below 0.6 m |
| F8 | Speed: only **30 % = 0.31 m/s** measured; the simulator assumes 1 m/s at 100 %, linear. Without a limit, PPO learned to drive at 100 % (1.1 m/s in the simulator) | wrong speeds above 30 % if the motors are not linear | mitigated: speed limited to 40 % in training, policy node and hub bridge. Open: measure at 20, 50, 70 % |
| F19 | Gazebo and the training simulator move the car differently: Gazebo accelerates and brakes harder (car plugin limit 2 m/s²; training: motor lag 0.25 s), and the car answers steering more slowly, so the network steers longer and then overshoots (`policy/compare_gazebo.py`) | small: both still arrive (PPO 3.2 s training, 2.9 s Gazebo) | open: measure the real car's acceleration and steering response, then match both simulators to it |
| F20 | When the object is not visible, the PPO network drives **straight on at full allowed speed** (training episodes start with the object in view, so it never learned to search) | the car runs away or into obstacles when it loses the object | **addressed** by `ppo_search` (scenario `mixed`): with the ball anywhere around the car it arrives in 87 % of the episodes (rule based search 92 %), in a median 5.4 s instead of 13.3 s |
| F21 | `ppo_search` searches by **reversing** with right lock, which swings the nose left; reversing, the camera sees nothing behind the car, so 11 % of the episodes end with backing into the ball. A ball inside its search circle is never seen | blind reversing is dangerous on the real car | open: forbid reverse while searching (or penalize it), or more cost for collisions; possibly memory of where the ball was last seen |
| F22 | In Gazebo, `ppo_search` found a ball behind the car and stopped 0.30 m from it, but 15 cm off to the side; it then wants to steer while barely moving, which Gazebo cannot do (F14), so it stays there. In training it creeps forward and corrects | not counted as arrived in Gazebo | open: let the sim adapter pass the steering angle at standstill, or teach the network to back up and retry |
| F9 | **Bluetooth delay** not measured; the simulator assumes 0 to 0.2 s in total | policy may oscillate if the real delay is larger | open: measure on the car |
| F10 | Steering ±1 corresponds to ±89° at motor D (80 % of the range), where the 1.20 m circle was measured | consistent with the simulator's 0.395 rad | **verified** |
| F11 | The Orin container lacks `vision_msgs` and `ackermann_msgs` (both installed on the laptop) | nodes using them cannot start on the Orin | **fixed**: both in `docker/Dockerfile` |
| F12 | Real color stream: **resolution and field of view not verified**. `robot_detection.py` opens it through DirectShow without setting a resolution. A 4:3 crop of the 16:9 sensor would have a narrower field of view than 70° | x would be scaled differently than in training | open: read the stream size and calibrate the camera |
| F13 | Gazebo publishes depth as `32FC1` in m; the real driver will likely use `16UC1` in mm | M3 must handle both | **fixed**: M3 reads both |
| F14 | Gazebo's `/cmd_vel` cannot express steering at standstill | small sim and real difference when starting | accepted |
| F15 | The **IMU is not a policy input**, and the training simulator does not model it. Its yaw rate (gyroscope z) would tell the policy how fast the car actually turns, which helps with steering delay and slip, and to keep turning while the object is out of view | none for now; a possible improvement | open: decide after the first trained policy; adding it means a 10th observation, simulated in `lego_car_env.py` |
| F16 | IMU **mounting position and axis orientation** on the car are not measured; Gazebo uses a guessed position | wrong signs or axes in the fusion, wrong Gazebo readings | open: measure, then update `imu_link` in the robot model |
| F17 | The MPU-6050 has **not been checked on the I2C bus** (bus number, address 0x68 or 0x69, chip ID); the Orin was off when this section was written | driver configuration unknown | open: `i2cdetect` and a read of register 0x75 (should be 0x68) |
| F18 | The Orin container has no I2C access settings or driver packages for the IMU yet; it runs `--privileged`, which exposes `/dev/i2c-*` | none yet | open: with the driver node |
| F23 | YOLO **loses the ball below about 0.6 m**, where it is large and cut off by the image edge (21 % found from 0.35 to 0.6 m). The car stops with the ball about 0.49 m from the camera, and the policy reads `visible = 0` as "drive on" | collision at the end of an otherwise good approach (twice in Gazebo) | mitigated: depth hold in M3. Open: train with misses that depend on the distance, so the policy also copes without the hold |
| F24 | With Zenoh, **camera images stopped reaching the Orin** while small messages (`/clock`, commands) still did, after runs had been stopped hard and nodes were left over on both machines. Restarting the laptop's Zenoh router fixed it; the cause was not pinned down. Also: the Orin's clock runs about 0.3 to 0.7 s ahead of the laptop, so the laptop router logs `Error treating timestamp` (and restamps) | the YOLO node gets no images, the car drives blind | worked around: `orin_policy_windows.py stop` now makes sure no node is left; if images do not arrive, `systemctl --user restart zenohd` in WSL. Open: find the cause; set the Orin's clock (NTP or from the laptop) |
| F25 | YOLO on the Orin's CPU takes about 120 ms with the chain running (87 ms alone), about 6 frames per second of wall time, below the 10 Hz spec. In Gazebo this does not matter yet, because the rendered cameras slow the simulation to about 0.2 of real time | too slow for the real car | open: TensorRT on the Orin's GPU |

## 6. Where the numbers live

| Value | Code |
|---|---|
| Car, camera and task constants, observation order | `policy/lego_car_env.py` |
| Rule based controller | `policy/expert.py` |
| Gazebo cameras and car | `odse_ws/src/odse_sim/urdf/lego_car.urdf.xacro` |
| Gazebo topics | `odse_ws/src/odse_sim/config/bridge.yaml` |
| Hub ports, signs, steering calibration | `LegoTechnicHub/examples/navigation_hints.py` |
| Camera specifications | `docs/camera_f200.md` |
| IMU, state estimation design | `DESIGN.md`, `prompt.md` (package `odse_imu`) |
