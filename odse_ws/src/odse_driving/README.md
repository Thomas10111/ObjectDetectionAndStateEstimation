# odse_driving

The trained driving network as ROS 2 nodes, and an experiment in Gazebo that
lets it drive the simulated car to a ball. Interfaces: `docs/interfaces.md`.

## Nodes

| Node | Input | Output | Spec |
|---|---|---|---|
| `sim_target` | `/world_poses` (true positions from Gazebo) | `/target` (`odse_interfaces/TargetObservation`) | stand in for M1 to M3 |
| `yolo` | `/camera/color/image_raw` | `/detections` (`vision_msgs/Detection2DArray`) | M2 |
| `target` | `/detections`, `/camera/depth/image_rect_raw`, both `camera_info` | `/target` | M3 |
| `policy` | `/target` | `/drive/command` (`ackermann_msgs/AckermannDriveStamped`) | M4 |
| `sim_adapter` | `/drive/command` | `/cmd_vel` (`geometry_msgs/Twist`) | M5a |
| `episode_monitor` | `/world_poses`, `/odom`, `/drive/command` | log output, CSV in `~/odse_experiments/` | experiment only |

`sim_target` replaces camera, YOLO and depth lookup with geometry
(`camera_model.py`, the same as the training simulator), so the experiment
shows the network's driving without detection errors. `yolo` and `target`
are the real chain: YOLO on the rendered camera images, then the depth
lookup. They run on the Orin (section "YOLO on the Orin" below).

## Setup (once)

```bash
bash /mnt/c/Users/fisch/MyGit/ObjectDetectionAndStateEstimation/scripts/setup_wsl_ros_python.sh
cd /mnt/c/Users/fisch/MyGit/ObjectDetectionAndStateEstimation/odse_ws
colcon build
```

The policy node needs onnxruntime, which lives in `~/.venvs/odse_ros`; the
launch file starts it with that Python.

## Run

```bash
source /mnt/c/Users/fisch/MyGit/ObjectDetectionAndStateEstimation/odse_ws/install/setup.bash
ros2 launch odse_driving experiment.launch.py
```

Gazebo opens with the car at the origin and an orange ball 1.8 m ahead. The
car waits 5 s of simulated time (`start_delay`), then drives. The terminal
shows, once per simulated second, what the policy sees and does, and finally
`OUTCOME: ARRIVED ...` (or `COLLISION`, `TIMEOUT`). The `WARN Watchdog ...`
lines from Zenoh's shared memory are harmless in WSL.

Compare the run with the same scenario in the training simulator:

```bash
source ~/.venvs/odse_policy/bin/activate
cd /mnt/c/Users/fisch/MyGit/ObjectDetectionAndStateEstimation/policy
python compare_gazebo.py $(ls -t ~/odse_experiments/*.csv | head -1)                      # PPO
python compare_gazebo.py $(ls -t ~/odse_experiments/*.csv | head -1) --model out/bc_policy.onnx
```

It writes `policy/out/compare_gazebo.png`: both paths from above, and gap,
speed and steering over time.

## Results (October 2026, ball at 1.8, -0.3)

| Network | Training simulator | Gazebo |
|---|---|---|
| PPO | arrived after 3.2 s, gap 0.31 m | arrived after 2.9 s, gap 0.33 m |
| Imitation | arrived after 4.3 s, gap 0.34 m | arrived after 4.3 s, gap 0.36 m |

Differences seen in the comparison plot: Gazebo's car accelerates and brakes
harder, and turns more slowly for the same steering command, so the network
steers longer and then overshoots (`docs/interfaces.md`, F19).

Search, ball behind the car:

```bash
ros2 launch odse_driving experiment.launch.py model:=ppo_search world:=arena target_x:=-0.5 target_y:=1.5
```

`ppo_search` reverses with right lock until the ball shows up, then drives to
it. In this run it stopped 0.30 m from the ball after 6.4 s, but 15 cm to the
side, so the monitor reports no arrival (F21, F22 in `docs/interfaces.md`).

Options:

| Argument | Default | Meaning |
|---|---|---|
| `model` | `ppo` | `ppo` (reinforcement learning, ball ahead), `ppo_search` (also searches when the ball is out of view), `bc` (imitation), `ppo_h4` and `ppo_search_h4` (the same with 4 hidden units instead of 64 × 64) |
| `world` | `room` | `room` (with other objects) or `arena` (empty, 6 × 6 m, for searching) |
| `target_x`, `target_y` | `1.8`, `-0.3` | ball position in m; the car starts at the origin facing +x |
| `gui` | `true` | Gazebo window |
| `cameras` | `false` | rendered cameras; slow on the laptop's integrated graphics |
| `noise` | `0.0` | detection noise, `1.0` as in training |
| `dropout` | `0.0` | share of missed detections, training uses 0 to 0.15 |
| `rviz` | `false` | RViz with car model and odometry |
| `start_delay` | `5` | simulated seconds before the car starts |
| `policy` | `laptop` | `orin`: start no policy node here, the network runs on the Orin (next section) |

To try other positions, stop the launch (Ctrl+C), move the ball in the
Gazebo window (translate tool) or edit `odse_sim/worlds/room.sdf`, and start
again. The car always starts at the origin, facing +x.

## Network on the Orin

Gazebo, `sim_target`, `sim_adapter` and the monitor stay on the laptop; only
the policy node runs in the Orin's `ros2_lyrical` container, connected over
Ethernet and Zenoh (`docs/orin_ssh_docker.md`). The container image has
`ackermann_msgs` and Ubuntu's `python3-onnxruntime` (CPU), see
`docker/Dockerfile`.

In PowerShell on the laptop (the SSH key for the Orin is on Windows):

```powershell
python scripts\orin_policy_windows.py deploy                       # once, and after code or model changes
python scripts\orin_policy_windows.py run --model ppo_search_h4    # leave running, Ctrl+C stops it
```

Then in WSL:

```bash
ros2 launch odse_driving experiment.launch.py policy:=orin world:=arena target_x:=-0.5 target_y:=1.5
```

The node on the Orin follows Gazebo's `/clock`. It can stay running: when
Gazebo is restarted, the simulated time jumps back and the node starts a new
episode. The ball position and world are set on the laptop, the model and
`start_delay` on the Orin.

Timing is logged on both sides: the policy node every 5 s (`target age`: how
old the latest `/target` was when the network used it; `network`: inference
time), the sim adapter every 50 commands (`command age`: how old a command
was on arrival). Measured on 14 October 2026:

| | Value |
|---|---|
| Network, Orin CPU (ONNX Runtime 1.23) | 0.2 ms median, 1.5 ms max (first call) |
| Command age, Orin to laptop | 2 ms median, single spikes to 90 to 110 ms |
| Target age | 2 or 33 ms, steady within a run: the 30 Hz target and the 10 Hz network timer line up differently per start |

Results are the same as with the network on the laptop: `ppo`, ball ahead,
arrived after 2.9 s and 3.0 s with gaps of 0.33 m and 0.34 m (laptop: 2.9 s,
0.33 m). `ppo_search_h4` with the ball at (-0.5, 1.5) reversed into the ball
after 5.6 s, on the Orin and on the laptop alike, with identical positions
each second: the blind reversing of F21, not an effect of the Orin.

The policy node on the Orin prints several hundred `Schema error: Trying to
register schema ... already registered` lines at start. They come from
Ubuntu's onnxruntime package and are harmless.

## YOLO on the Orin

YOLO, the target node and the policy run on the Orin; Gazebo renders the
cameras on the laptop and `sim_target` stays off. Commands, results and the
detection measurements: section 9.5 of `docs/how_it_works.md`.

```powershell
python scripts\orin_policy_windows.py run --target yolo                  # PowerShell
```

```bash
ros2 launch odse_driving experiment.launch.py policy:=orin target:=orin  # WSL
```

`ppo`, ball ahead: arrived after 2.9 s, gap 0.33 m, the same as with
`sim_target`. This needs the depth hold of the target node (`hold_for`,
default 1.0 s), because YOLO loses the ball in the last half meter.

## Models

`models/ppo_policy.onnx`, `models/ppo_search_policy.onnx` and `models/bc_policy.onnx` are copies of the files
written by `policy/train_ppo.py` and `policy/train_bc.py`. After retraining,
copy them again and rebuild:

```bash
cp ../../policy/out/*_policy.onnx src/odse_driving/models/ && colcon build --packages-select odse_driving
```

`ppo_h4_policy.onnx` and `ppo_search_h4_policy.onnx` are the best seeds of the
network size sweep (`policy/out/ppo_h4_s0_policy.onnx` and
`policy/out/search_h4_s2_policy.onnx`, section 8.1 of `docs/how_it_works.md`).
After changing models, run `orin_policy_windows.py deploy` again for the Orin.
