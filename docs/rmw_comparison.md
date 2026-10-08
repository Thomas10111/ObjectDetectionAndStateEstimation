# ROS 2 middleware: Fast DDS versus Zenoh

Which ROS 2 middleware (RMW) to use for camera images on this robot. Measured
on 7 October 2026 with ROS 2 Lyrical.

**Result:** use **Zenoh with shared memory** on both machines. With default
settings, Fast DDS lost half of all depth image sized messages on the Orin,
delivered under 20 % from the laptop to the Orin, and does not work at all
between programs on the laptop (WSL mirrored networking). Zenoh delivered 98 to
100 % in every depth image test. Over the **Ethernet link, Zenoh delivered
100 % of all sizes**, including 1080p color sized messages, which had failed
over USB with both middlewares (section 5).

## Setup

| | Laptop | Orin |
|---|---|---|
| Hardware | Intel Core Ultra 7 165U, 16 GB, integrated graphics | Jetson Orin Nano 8 GB, 6 cores |
| ROS 2 | Lyrical in WSL `Ubuntu-26.04`, mirrored networking | Lyrical in container `ros2_lyrical` (`ros:lyrical-perception`) |
| Fast DDS | `rmw_fastrtps_cpp`, default settings | same |
| Zenoh | `rmw_zenoh_cpp` 0.10.6 | same |
| Link | USB device mode (RNDIS), laptop `192.168.55.100`, Orin `192.168.55.1` | |

## 1. Image sized messages on the Orin

Publisher and subscriber both in the `ros2_lyrical` container. 450 messages at
30 Hz per run, `sensor_msgs/Image` with sensor data QoS (best effort, keep last
5). The subscriber acknowledges each message, the publisher measures the round
trip. CPU is percent of one core, measured per process with `os.times()`.

| Message | Middleware | Delivered | Rate | Round trip median / p95 | CPU pub | CPU sub |
|---|---|---|---|---|---|---|
| 0.6 MB (F200 depth, 640×480 16 bit) | Fast DDS | **61 %** | 18.3 Hz | 7.5 / 11.9 ms | 17 % | 11 % |
| | Zenoh | 100 % | 30.0 Hz | 9.5 / 11.3 ms | 25 % | 14 % |
| | Zenoh + shared memory | **100 %** | **30.0 Hz** | **4.2 / 5.5 ms** | **12 %** | **10 %** |
| 1.2 MB (sim depth, 640×480 float) | Fast DDS | **51 %** | 15.4 Hz | 10.0 / 19.0 ms | 16 % | 16 % |
| | Zenoh | 100 % | 30.1 Hz | 11.0 / 15.1 ms | 30 % | 23 % |
| | Zenoh + shared memory | **100 %** | **30.0 Hz** | **8.6 / 10.0 ms** | **20 %** | **16 %** |
| 6.2 MB (F200 color, 1920×1080 RGB) | Fast DDS | 95 % | 29.1 Hz | 15.5 / 27.2 ms | 29 % | 39 % |
| | Zenoh | 100 % | 30.3 Hz | 20.6 / 22.5 ms | 45 % | 51 % |
| | Zenoh + shared memory | **100 %** | **30.1 Hz** | **13.4 / 28.4 ms** | **26 %** | **27 %** |

The Zenoh router (`rmw_zenohd`) used 0.06 s of CPU over all six Zenoh runs.

Findings:

1. **Fast DDS with default settings loses large messages.** Only 51 to 61 %
   of depth sized messages arrived. Likely cause: messages larger than the
   default shared memory segment go over UDP in many fragments, and with best
   effort QoS one lost fragment drops the whole message.
2. **Zenoh delivered every message** at every size.
3. **Zenoh without shared memory costs more CPU per send** (25 % instead of
   17 % for 0.6 MB), but it delivers about twice as many messages, so the cost
   per delivered image is similar.
4. **Zenoh with shared memory is best in every row:** lowest CPU, lowest
   latency, no losses. Enabled with
   `ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true'`.

Limitations:

- **Python test nodes.** Real C++ camera and detection nodes need less CPU.
  The comparison between the middlewares still holds.
- **Default settings, one run each.** Fast DDS has a large data mode
  (`FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA`) that likely reduces its losses;
  not tested yet.
- **Same machine only.** Laptop to Orin is in section 4.

## 2. Laptop (WSL) on its own

Talker and listener (`demo_nodes_cpp`), both in WSL, 6 messages sent:

| Configuration | Received |
|---|---|
| Fast DDS, default | 0 |
| Fast DDS, `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` | 0 |
| Fast DDS, `ROS_STATIC_PEERS=127.0.0.1` | 0 |
| Fast DDS, UDP only (`FASTDDS_BUILTIN_TRANSPORTS=UDPv4`) | 0 |
| Fast DDS, shared memory only (`FASTDDS_BUILTIN_TRANSPORTS=SHM`) | 6, but with the Gazebo simulation the test node received nothing (the `ros2` command line tools did get `/clock` and `/odom`) |
| Zenoh | **6** |

Before switching WSL to mirrored networking, Fast DDS worked locally. With
mirrored networking, Fast DDS advertises the laptop's network addresses, and
the WSL firewall (Hyper-V firewall, inbound blocked by default) apparently
drops that traffic even when it comes from WSL itself. Laptop to Orin over
Fast DDS did work, thanks to the `ROS2-Orin-USB` firewall rule.

The Gazebo simulation (`odse_ws/src/odse_sim`) delivered all topics at full
rate with Zenoh.

## 3. USB link between laptop and Orin

Raw TCP throughput, 64 KB blocks from the laptop to the Orin:

| From | Throughput while running | Outcome |
|---|---|---|
| Windows | 53 to 81 MB/s, mean 66.5 MB/s (530 Mbit/s) | connection reset after 7.6 s |
| WSL | about 75 MB/s | stalled after 107 MB (about 1.5 s), then timed out |
| WSL, first attempt | 2.9 MB/s average | stalled after 92 MB |

**The link dropped about every 27 seconds.** The Orin's DHCP server log showed
a `DHCPDISCOVER` from the laptop's RNDIS adapter (MAC `4a:10:08:f3:79:74`)
every 20 to 30 s, all day. This also explains the SSH timeouts and
`Connection reset` errors seen throughout the day.

**Cause:** not the USB connection itself (the Orin's `usb0` counted only 2
carrier changes since boot) but the DHCP lease time. NVIDIA's USB device mode
hands out leases of **15 s** (`net_dhcp_lease_time=15` in
`/opt/nvidia/l4t-usb-device-mode/nv-l4t-usb-device-mode-config.sh`). Windows
did not renew such short leases: it let the address expire, dropped it, and
started over, breaking open connections each time.

**Fix (applied 7 October 2026):** lease time **60 s**. Windows now renews with
`DHCPREQUEST` about every 40 s; no `DHCPDISCOVER` in 3 minutes of watching.
After the fix, 1028 MB went from WSL to the Orin in 20 s at 51 MB/s
(409 Mbit/s), without a single stall (per 0.5 s: 28 to 75 MB/s).

NVIDIA keeps the lease short on purpose: the pool has one address, and a host
with a different MAC (Linux hosts randomize it) has to wait for the old lease to
expire. With 60 s, that wait is at most a minute. Orin reboots are no problem,
because the start script deletes all leases. **A JetPack update may reset the
value to 15**; see `docs/orin_ssh_docker.md` for the command.

The bandwidth is enough for both simulated cameras uncompressed (about
58 MB/s) only with little margin; plan for compression.

## 4. Laptop (WSL) to Orin over USB

After the DHCP fix. Publisher in WSL, subscriber in the Orin container, same
method as section 1 (`scripts/bench/run_cross.sh`). Fast DDS with
`ROS_STATIC_PEERS` on both sides; Zenoh with the WSL router connecting to the
Orin router (`connect/endpoints=["tcp/192.168.55.1:7447"]`). 15 s per run,
sensor data QoS unless marked reliable.

| Message | Middleware | Delivered | Rate received | Round trip median / p95 | CPU laptop | CPU Orin |
|---|---|---|---|---|---|---|
| 0.6 MB @ 30 Hz | Fast DDS | **19 %** | bursts, then nothing | 27.5 / 30.8 ms | 10 % | 24 % |
| | Fast DDS LARGE_DATA | **0 %** | | | | |
| | **Zenoh** | **100 %** | **30.0 Hz** | **19.1 / 21.7 ms** | 7 % | 17 % |
| 1.2 MB @ 30 Hz | Fast DDS | **18 %** | publisher blocked at 18 Hz | **13 s** (backlog) | 8 % | 23 % |
| | Fast DDS LARGE_DATA | **0 %** | | | | |
| | **Zenoh** | **98 %** | **29.4 Hz** | **33.2 / 76.7 ms** | 12 % | 21 % |
| 6.2 MB @ 5 Hz | Fast DDS | 21 % | | 14 s | 5 % | 29 % |
| | Fast DDS, reliable | 28 % | 3.4 Hz | 13 s | 9 % | 21 % |
| | Fast DDS LARGE_DATA | 0 % | | | | |
| | Zenoh | **1 %** | | | 7 % | |
| | Zenoh, reliable | 8 % | 0.4 Hz | 139 ms | 7 % | 3 % |

Findings:

1. **Depth images (0.6 and 1.2 MB): Zenoh delivers, Fast DDS does not.**
   Zenoh: 98 to 100 % at full rate, 19 to 33 ms round trip. Fast DDS with
   default settings: under 20 %, with a backlog of up to 13 s.
2. **Fast DDS LARGE_DATA received nothing.** That mode adds TCP connections,
   and the WSL firewall rule `ROS2-Orin-USB` only allows UDP; most likely the
   cause, not verified.
3. **6.2 MB (1080p color) fails with both**, also with reliable QoS, although
   5 Hz needs only 31 MB/s and the link carried 51 MB/s in the raw TCP test.
   Cause not found. A plausible factor for Zenoh best effort: it drops a
   message when the transmit queue is full, and one 6.2 MB message is far
   larger than the default queue. Either way, **color images must be
   compressed or scaled down before crossing the link**; 1080p at 30 Hz
   uncompressed would need 187 MB/s.

## 5. Laptop (WSL) to Orin over Ethernet, Zenoh

Same method as section 4 (`LINK=ethernet scripts/bench/run_cross.sh zenoh`),
over the gigabit Ethernet link (USB dongle on the laptop, fixed addresses
`192.168.56.100` and `192.168.56.1`), with the permanent Zenoh setup: both
routers always running, shared memory on. Fast DDS was not repeated: the WSL
firewall rule it needs only covers the USB network.

| Message | Delivered | Rate received | Round trip median / p95 | CPU laptop | CPU Orin | Same over USB (section 4) |
|---|---|---|---|---|---|---|
| 0.6 MB @ 30 Hz | **100 %** | 30.0 Hz | 11.9 / 12.9 ms | 8 % | 8 % | 100 %, 19.1 / 21.7 ms |
| 1.2 MB @ 30 Hz | **100 %** | 30.1 Hz | 18.9 / 20.0 ms | 10 % | 12 % | 98 %, 33.2 / 76.7 ms |
| 6.2 MB @ 5 Hz | **100 %** | 5.0 Hz | 83.4 / 93.2 ms | 7 % | 7 % | **1 %** |
| 6.2 MB @ 5 Hz, reliable | **100 %** | 5.0 Hz | 82.9 / 91.6 ms | 7 % | 7 % | 8 % |

Findings:

1. **Everything arrives over Ethernet**, including 1080p color sized
   messages. The 6.2 MB failures in section 4 came from the USB link, not
   from Zenoh.
2. **Round trips are about 40 % shorter** than over USB, and the 95th
   percentile stays close to the median (no spikes like 77 ms over USB).
3. **CPU on both sides stays at 7 to 12 % of one core** (Python test nodes).
4. Raw link capacity is 118 MB/s, so uncompressed 1080p color still fits only
   up to about 15 Hz; at 30 Hz it needs compression or a lower resolution.

## Recommendations

1. **Zenoh with shared memory on both machines.**
2. **Use Ethernet as the main link** (done 7 October 2026): a USB Ethernet
   dongle on the laptop to the Orin's Ethernet port, fixed addresses
   `192.168.56.100` and `192.168.56.1`. Raw TCP from WSL: **117.8 MB/s
   (941 Mbit/s), steady** over 20 s, more than twice the USB link, and Zenoh
   delivers all message sizes (section 5). Keep the USB DHCP lease at 60 s
   (section 3) for when USB is used.
3. **On the robot, run camera and detection in one process** (composable
   nodes with intra process communication). Images are then passed without
   copying, whichever middleware is used.
4. **For simulation with the Orin in the loop**, plan for reduced resolution
   or compressed images (`image_transport`) over the link.

## Open questions

- Fast DDS with `FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA`: on the Orin alone
  (no firewall involved) not tested yet; across the link it needs a TCP
  firewall rule.
- Why 6.2 MB messages fail over the USB link with both middlewares (section
  4). They work over Ethernet (section 5), so this only matters when USB is
  used.
- Why the throughput falls from about 75 to about 40 MB/s after 10 s of
  continuous transfer.

## Reproduce

Scripts in `scripts/bench/`:

| Script | Purpose |
|---|---|
| `bench_pub.py SIZE HZ SECONDS` | publishes `/bench_image`, measures round trip via `/bench_ack` |
| `bench_sub.py SECONDS` | receives, acknowledges, reports rate, throughput and CPU |
| `run_local.sh` | runs section 1 inside the container |
| `run_cross.sh` | runs sections 4 and 5 from WSL: `LINK=usb run_cross.sh` (all middlewares), `run_cross.sh zenoh` (Ethernet); needs the scripts in `/root/bench` in the container |
| `tcp_sink.py`, `tcp_source.py` | raw TCP throughput (section 3); sink on the Orin, source on the laptop |

Section 1 on the Orin (runs detached, so USB dropouts cannot stop it):

```bash
docker exec ros2_lyrical mkdir -p /root/bench
docker cp scripts/bench/. ros2_lyrical:/root/bench/
setsid nohup docker exec ros2_lyrical bash /root/bench/run_local.sh > ~/bench_local.log 2>&1 < /dev/null &
cat ~/bench_local.log   # after about 4 minutes
```

Section 3: start `python3 tcp_sink.py` on the Orin, then
`python3 tcp_source.py 192.168.55.1` on the laptop.

Sources for the camera sizes: `docs/camera_f200.md`.
