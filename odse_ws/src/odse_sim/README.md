# odse_sim

Gazebo (Jetty) simulation of the LEGO Technic car with front axle steering,
the RealSense F200 depth camera and an IMU, bridged to ROS 2 Lyrical.

## Run

In WSL (`Ubuntu-26.04`). Build once, and again after changing files:

```bash
cd /mnt/c/Users/fisch/MyGit/ObjectDetectionAndStateEstimation/odse_ws
colcon build --packages-select odse_sim
```

Every terminal needs:

```bash
source /mnt/c/Users/fisch/MyGit/ObjectDetectionAndStateEstimation/odse_ws/install/setup.bash
```

ROS 2 uses Zenoh; `~/.bashrc` sets `RMW_IMPLEMENTATION=rmw_zenoh_cpp` and the
router runs as the service `zenohd` (`systemctl --user status zenohd`), both
set up by `scripts/wsl/setup_zenoh.sh`.

Terminal 1, the simulation:

```bash
ros2 launch odse_sim sim.launch.py              # Gazebo window
ros2 launch odse_sim sim.launch.py gui:=false   # server only, faster
ros2 launch odse_sim sim.launch.py rviz:=true   # plus RViz
```

Terminal 2, drive with the keyboard:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Why Zenoh: with WSL mirrored networking, the default Fast DDS cannot reach
other ROS 2 programs on the same laptop over UDP; `docs/rmw_comparison.md`
has the measurements.

Because the laptop's router is connected to the Orin's, the simulation's
topics are visible on the Orin too (when the USB cable is plugged in). Use a
different `ROS_DOMAIN_ID` in all terminals to keep a simulation separate.

## Topics

Names match what the real robot will publish.

| Topic | Type | Rate (sim time) |
|---|---|---|
| `/cmd_vel` (in) | `geometry_msgs/msg/Twist` | |
| `/odom` | `nav_msgs/msg/Odometry` | 30 Hz |
| `/tf` | `odom` to `base_link` | 30 Hz |
| `/joint_states` | `sensor_msgs/msg/JointState` | |
| `/imu/data_raw` | `sensor_msgs/msg/Imu` | 200 Hz |
| `/camera/color/image_raw`, `/camera/color/camera_info` | 640×360 `rgb8` | 30 Hz |
| `/camera/depth/image_rect_raw`, `/camera/depth/camera_info` | 640×480 `32FC1`, meters | 30 Hz |
| `/camera/depth/points` | `sensor_msgs/msg/PointCloud2` | 30 Hz |
| `/clock` | `rosgraph_msgs/msg/Clock` | |

`/cmd_vel` is interpreted by Gazebo's Ackermann plugin: `linear.x` is the
speed in m/s, `angular.z` the yaw rate. The steering angle follows from
`atan(wheelbase * angular.z / linear.x)`, limited to 0.40 rad.

## Model

`urdf/lego_car.urdf.xacro`. `base_link` is on the floor below the center of
the rear axle.

| | Value | Source |
|---|---|---|
| Wheelbase | 0.25 m | measured |
| Track width | 0.18 m | measured |
| Wheel diameter | 0.085 m | measured |
| Length, width, height | 0.42, 0.28, 0.15 m | measured |
| Turning circle | 1.20 m, so max steering 0.40 rad (22.6°) | measured |
| Speed at 30 % motor power | about 0.31 m/s | measured (circle time and motor rotation agree) |
| Max speed | 1.0 m/s | estimated from the 30 % value |
| Camera position | 0.15 m above the floor, over the front axle | assumed; launch arguments `camera_x`, `camera_z` |
| Depth camera | 640×480, 68° × 54°, 0.2 to 1.0 m, 30 FPS | `docs/camera_f200.md` |
| Masses | body 1.2 kg, wheels 0.05 kg | estimated |
| Drive | rear wheels only | the real car drives both axles (motors A and B) |

## Verified (October 2026, laptop with Intel graphics)

- All topics publish at the rates above (in simulation time).
- Depth: 640×480, field of view 68.0° × 53.7°, values 0.28 to 0.99 m.
- Circle at 0.31 m/s and full left steering: 1.31 m diameter (real car
  1.20 m), back within 12 cm of the start after one lap.
- Real time factor about **0.27** with both cameras, without the Gazebo
  window. The integrated graphics are the bottleneck. Everything that uses
  `use_sim_time` (all nodes started here) stays consistent; it is just slow.

## Known issues

- **Circle 9 % larger than on the real car.** Likely the steering joints lag
  behind the command, or wheel slip. Tune later if it matters for planning.
- **The bridge sometimes crashes on shutdown** (`Failed to publish`). Only
  seen when stopping the launch; harmless.
- **`GZ_IP=127.0.0.1` is set by the launch file.** Without it, Gazebo's own
  transport fails under WSL mirrored networking ("Exception sending a
  multicast message: Network is unreachable") and no data reaches ROS 2.
  When using the `gz` command line tool in another terminal, export it too.
