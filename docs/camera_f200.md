# Camera: Intel RealSense F200

The robot's depth camera is an **Intel RealSense F200**. The sibling project
`ObjectDetection` uses it (`robot_detection.py`); `DESIGN.md` only says
"RGB webcam; RealSense optional". This page collects what the code assumes,
what public sources say, and what follows for this project.

Checked October 2026.

## Specifications

| | Value | Source |
|---|---|---|
| Depth method | Coded light IR: a MEMS mirror projects a 2D Gray code pattern | Intel legacy specs |
| Depth range | **0.2 to 1.0 m** indoors (paper: 20 to 120 cm recommended) | Intel legacy specs, paper |
| Depth resolution | 640×480 | paper |
| Depth frame rate | 30 or 60 FPS | Intel legacy specs |
| Depth field of view | **68° × 54°** | Intel legacy specs |
| Infrared frame rate | 5 to 300 FPS, global shutter | Intel legacy specs |
| Color resolution | 1920×1080, rolling shutter | paper, Intel legacy specs |
| Color frame rate | about 30 FPS | paper |
| Depth format on Linux | `INVZ`, 16 bit, through the standard `uvcvideo` driver | librealsense issues |
| Outdoor use | not possible, sunlight drowns the IR pattern | Intel legacy specs |
| Several cameras | strong interference when pointed at the same scene | Intel legacy specs |
| Range control | trading frame rate for range can go down to about 2 FPS | Intel legacy specs |
| Status | end of life | Intel software support model |

The paper also lists the field of view as "77° × 43° × 70°", most likely the
color camera and its diagonal. For the depth stream, use Intel's 68° × 54°.

## Code versus sources

| | `ObjectDetection` code | Sources | Verdict |
|---|---|---|---|
| Depth range | 200 to 1200 mm | 0.2 to 1.0 m (Intel), 0.2 to 1.2 m (paper) | Matches; Intel's figure is shorter |
| Depth rate | about 1 FPS, called a "hardware limit" | 30 or 60 FPS | **Code comment is wrong.** The 1 FPS comes from reading through OpenCV with DirectShow on Windows |
| Depth values | 8 bit (0 to 255), mapped linearly to mm | 16 bit natively | **Precision is lost** somewhere in the Windows stack; the linear mm mapping is only approximate |
| Interface | OpenCV `VideoCapture` as a webcam (`CAP_DSHOW`) | `librealsense2` does not support the F200 | Consistent: the code works around the missing SDK support |

Affected places in `ObjectDetection`: `robot_detection.py` (module docstring,
the comment above `DEPTH_MAX_AGE_MS`, the `F200Camera` docstring, a startup
message and the on-screen text "~1 FPS is normal for F200") and `CLAUDE.md`
("F200 depth note: ... (hardware limit)").

## Software support

- **`librealsense2` (SDK 2.x) does not support the F200.** It starts with the
  SR300 and the D400 series. So the official ROS 2 driver `realsense-ros`,
  which builds on SDK 2.x, cannot use it either.
- **`librealsense` 1.x (legacy)** supports the F200, but it is unmaintained and
  likely hard to build on Ubuntu 26.04 and ARM (the Orin).
- **Plain V4L2 works without any SDK.** The F200 is a USB Video Class device;
  on Linux its color, depth and IR streams show up as `/dev/video*` nodes.
- On the Orin (October 2026), `rs-enumerate-devices` (SDK 2.x, built from
  `~/MyGit/librealsense`) reports "No device detected". Not yet tested with
  the camera plugged in.

## Consequences for this project

**Simulation (Gazebo):** model the depth camera as 640×480, 68° × 54° field
of view, 0.2 to 1.0 m range, 30 FPS. Use a reduced color resolution to spare
the laptop's integrated graphics.

**Navigation:** with about 1 m of depth range, Nav2 sees obstacles only
shortly ahead. The car has to drive slowly; obstacles beyond 1 m are invisible
to the costmap.

**Orin driver:** write a small ROS 2 node that reads the 16 bit depth and the
color stream through V4L2 and publishes `sensor_msgs/Image` and
`sensor_msgs/CameraInfo`. Expected: about 30 FPS depth instead of 1 FPS, with
full precision. Open questions to test with the camera on the Orin:

1. Which `/dev/video*` nodes and formats appear (`v4l2-ctl --list-devices`,
   `v4l2-ctl -d /dev/videoN --list-formats-ext`).
2. How `INVZ` values convert to meters (scale factor).
3. Camera intrinsics: the F200 stores calibration on the device, but reading
   it needs the legacy SDK; otherwise calibrate with a checkerboard.

**Alternative:** a current RealSense **D435i** or **D455** works with the
official ROS 2 driver, sees several meters, and has a built-in IMU that would
help the visual inertial odometry in `DESIGN.md`.

## Sources

- [Intel librealsense legacy branch: camera specs](https://github.com/IntelRealSense/librealsense/blob/legacy/doc/camera_specs.md):
  F200 range, field of view, frame rates, depth method, outdoor and multi camera limits.
- [Depth accuracy of the RealSense F200: Low-cost 4D facial imaging, Scientific Reports (2017)](https://www.nature.com/articles/s41598-017-16608-7):
  recommended range 20 to 120 cm, resolutions, about 30 FPS.
- [Fedora Project wiki: librealsense2](https://fedoraproject.org/wiki/Changes/librealsense2):
  SDK 2.x does not support the F200.
- [Intel RealSense software support model](https://dev.intelrealsense.com/docs/software-support-model):
  SDK 1.x versus 2.x device split, end of life status.
- [librealsense issue #913](https://github.com/IntelRealSense/librealsense/issues/913):
  uvcvideo driver on Linux.
