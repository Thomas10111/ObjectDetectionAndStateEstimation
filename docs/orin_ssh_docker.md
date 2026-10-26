# Connect to the Orin and run the ROS 2 container

How to reach the Jetson Orin Nano (`fishbox`) from the laptop over SSH and work
in its ROS 2 Docker container. The Orin keeps its own copy of the container
commands in `~/ROS2_DOCKER.md`.

| | |
|---|---|
| User | `fish` |
| **Ethernet address** (main link) | `192.168.56.1` (laptop side: `192.168.56.100`), 1 Gbit/s, 118 MB/s measured |
| USB address (alternative) | `192.168.55.1` (laptop side: `192.168.55.100`), about 51 MB/s |
| Wi-Fi address | `192.168.1.100` (not reachable from the laptop so far) |
| Container | `ros2_lyrical` from image `odse/ros:lyrical-zenoh` (repo `docker/`), ROS 2 Lyrical on Ubuntu 26.04, starts with the Orin |

## 1. Connect the cable

**Ethernet (main link):** the laptop has no Ethernet port, so a USB Ethernet
dongle (Windows: *Realtek USB GbE Family Controller*, adapter `Ethernet 7`) is
connected with a network cable to the Orin's Ethernet port. There is no DHCP
on this cable; both sides have fixed addresses, set once (October 2026):

```bash
# Orin
sudo nmcli connection modify 'Wired connection 1' ipv4.method manual ipv4.addresses 192.168.56.1/24 ipv4.never-default yes
sudo nmcli connection up 'Wired connection 1'
```

```powershell
# Laptop, PowerShell as Administrator
New-NetIPAddress -InterfaceAlias "Ethernet 7" -IPAddress 192.168.56.100 -PrefixLength 24
```

No gateway on either side, so both keep using Wi-Fi for the internet. The
laptop address belongs to adapter `Ethernet 7`; in another USB port Windows may
create a new adapter without it. Then run the command again with the new name
(`Get-NetAdapter` lists them).

**USB (alternative):** a USB-C **data** cable into the Orin's USB-C port. After
about a minute Windows shows *Remote NDIS Compatible Device* with
`192.168.55.100`. Needs the DHCP lease fix (see "SSH and the USB link" under
Lessons learned).

## 2. Log in

In PowerShell on the laptop:

```powershell
ssh fish@192.168.56.1                          # Ethernet
ssh -o ConnectionAttempts=5 fish@192.168.55.1  # USB
```

The laptop's key (`~/.ssh/id_ed25519`) is installed on the Orin, so no password
is asked. Over USB, `ConnectionAttempts=5` helps right after plugging in, while
Windows is still getting its address.

If the IPv4 addresses are missing, the Orin is still reachable over the
Ethernet cable through IPv6 link-local:
`ssh fish@fe80::f024:7d2f:c06f:854f%68` (`68` is the laptop's interface index
of the dongle; `Get-NetAdapter` shows it as `ifIndex`).

## 3. Start the container and open a shell

On the Orin (after step 2):

```bash
docker start ros2_lyrical
docker exec -it ros2_lyrical bash
```

The shell sets up ROS 2 by itself; `echo $ROS_DISTRO` prints `lyrical`.
Type `exit` twice to leave the container and then the Orin.

**Shortcut:** steps 2 and 3 in one command from PowerShell:

```powershell
ssh -t fish@192.168.56.1 "docker start ros2_lyrical > /dev/null && docker exec -it ros2_lyrical bash"
```

`-t` gives the remote shell a terminal, which `docker exec -it` needs.

## Other useful commands

From PowerShell, without logging in first:

```powershell
ssh fish@192.168.56.1 "docker ps -a"                 # container status
ssh fish@192.168.56.1 "docker stop ros2_lyrical"     # stop the container
ssh -t nvidia_orin "sudo shutdown now"              # switch the Orin off; -t lets sudo ask for the password
scp .\file.txt fish@192.168.56.1:/tmp/              # copy a file to the Orin
```

Everything in `~/ros2_ws` on the Orin appears in the container at
`/root/ros2_ws`.

## ROS 2 between the laptop (WSL) and the Orin

ROS 2 Lyrical runs in WSL (`Ubuntu-26.04`) on the laptop and in the
`ros2_lyrical` container on the Orin. They talk over the USB link using the
**Zenoh** middleware (`rmw_zenoh_cpp`); `docs/rmw_comparison.md` explains why.
Each side runs a Zenoh router; ROS 2 nodes connect to their local router, and
the laptop's router connects out to the Orin's router at
`tcp/192.168.56.1:7447` (Ethernet). To use the USB link instead, change the
address in `scripts/wsl/zenohd.service` to `192.168.55.1` and run
`scripts/wsl/setup_zenoh.sh` again. Listing both did not work: Zenoh always
picked the slower USB link.

| | Laptop (WSL) | Orin (container) |
|---|---|---|
| Middleware | `~/.bashrc`: `RMW_IMPLEMENTATION=rmw_zenoh_cpp` | set in the image (`docker/Dockerfile`) |
| Shared memory | `~/.bashrc`: `ZENOH_CONFIG_OVERRIDE='transport/shared_memory/enabled=true'` | set in the image |
| Router | systemd user service `zenohd`, connects to the Orin | started by the container's entrypoint |
| Set up by | `scripts/wsl/setup_zenoh.sh` | `docker/run_orin.sh` |

Check the routers:

```bash
systemctl --user status zenohd                          # laptop (WSL)
docker exec ros2_lyrical pgrep -af rmw_zenohd           # Orin
```

The laptop's router retries the Orin forever, so it may start before the Orin
is plugged in.

**Mirrored networking** in `C:\Users\fisch\.wslconfig` is still needed, so WSL
shares the laptop's network adapters, including the USB link:

```ini
[wsl2]
networkingMode=mirrored
```

Changes take effect after `wsl --shutdown`.

**No longer needed with Zenoh** (left from the Fast DDS setup):

- The firewall rule `ROS2-Orin-USB` (UDP from `192.168.55.0/24` into WSL).
  The laptop connects out to the Orin, which the firewall allows anyway. It is
  harmless; remove it in PowerShell as Administrator with
  `Remove-NetFirewallHyperVRule -Name "ROS2-Orin-USB"` if you like.
- `ROS_STATIC_PEERS`, removed from both `.bashrc` files.

To go back to Fast DDS for a test, set `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`
and `ROS_STATIC_PEERS` (laptop: `192.168.55.1`, Orin: `192.168.55.100`) in
both terminals; Fast DDS then needs the firewall rule again.

Test, with one terminal on each side. In the Orin container:

```bash
ros2 topic pub -r 2 /chatter std_msgs/msg/String "{data: hello from orin}"
```

In WSL:

```bash
ros2 topic echo /chatter std_msgs/msg/String
```

Always name the message type in `ros2 topic echo`; without it, the command
gives up before the two sides have found each other.

If nothing arrives: are both routers running (see above), and does
`echo $RMW_IMPLEMENTATION` print `rmw_zenoh_cpp` on both sides? Programs using
different middlewares cannot see each other.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Connection timed out` | Check that the *Remote NDIS* adapter exists in Windows network connections and has `192.168.55.100`. If timeouts come back regularly, check the DHCP lease time (see "SSH and the USB link" under Lessons learned); a JetPack update may have reset it to 15 s. |
| `Permission denied (publickey,password)` | Wrong user, or the key is missing on the Orin. Copy it again: `Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub \| ssh fish@192.168.56.1 "cat >> ~/.ssh/authorized_keys"` |
| `Connection reset` during long commands | Same check as above. For long jobs (for example `docker pull`), start them detached on the Orin: `setsid nohup <command> > ~/job.log 2>&1 < /dev/null &` |
| `docker: permission denied` | Your user is missing from the `docker` group: `sudo usermod -aG docker fish`, then log out and back in. |
| `docker pull` fails with `server misbehaving` | DNS on the Orin is not ready yet after a reboot; wait a minute. |
| ROS 2 topics from the other side do not show up | Both Zenoh routers running? `RMW_IMPLEMENTATION=rmw_zenoh_cpp` on both sides? Inside WSL, `ping 192.168.56.1` must work. Run `ros2 daemon stop` to clear stale discovery results. Details in the ROS 2 section above. |
| `Error: No such container` or the container is gone | Recreate it: copy the repo's `docker/` folder to the Orin and run `bash run_orin.sh` there. |

## Lessons learned

Pitfalls found while setting this up (October 2026). Each one cost at least
one failed attempt.

### ROS 2

- **`ros2 topic echo` needs the message type.** Without it, the command looks
  the type up once, finds nothing because discovery takes a few seconds, and
  quits with *"topic does not appear to be published yet"*. With the type it
  waits: `ros2 topic echo /chatter std_msgs/msg/String`.
- **`ros2 topic list` right after starting is incomplete** for the same reason.
  Wait a few seconds, or list again.
- **The ROS 2 CLI daemon caches discovery results.** After changing network
  settings or `ROS_STATIC_PEERS`, run `ros2 daemon stop`; the next `ros2`
  command starts a fresh daemon with the new settings.
- **`ros2 topic pub --once` waits for a subscriber** before it publishes, so it
  hangs forever when nobody listens. Wrap it in `timeout` in scripts.
- **xacro defines `pi` itself.** Defining it again only warns, but
  `ros2 launch` treats any xacro output on stderr as an error and stops.
- **The `ros:lyrical-perception` image has no demo nodes** (`demo_nodes_cpp`
  is missing). Test with `ros2 topic pub` and `ros2 topic echo` instead.
- **Automatic discovery uses Wi-Fi**, and the laptop cannot reach the Orin over
  Wi-Fi (`192.168.1.100` does not answer; the router probably isolates Wi-Fi
  clients). `ROS_STATIC_PEERS` forces the USB link.
- **Both machines must run the same ROS 2 version.** Messages between
  different versions (for example Jazzy and Lyrical) are not guaranteed to
  work, and failures are silent. ROS 2 packages exist for one Ubuntu version
  each: Lyrical for 26.04, Jazzy for 24.04.

### WSL

- **WSL blocks all incoming traffic by default** (Hyper-V firewall,
  `DefaultInboundAction: Block`). Outgoing works, so pings and one direction
  can look fine while ROS 2 discovery fails. The `ROS2-Orin-USB` rule fixes it.
- **Default WSL networking cannot see the USB link.** `networkingMode=mirrored`
  is needed; it takes effect after `wsl --shutdown`, which closes all WSL
  windows.
- **Mirrored networking breaks Fast DDS between programs on the laptop.** A
  talker and a listener both in WSL see nothing over UDP, not even with
  `ROS_STATIC_PEERS=127.0.0.1` or `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`.
  Shared memory only (`FASTDDS_BUILTIN_TRANSPORTS=SHM`) works for small
  messages but not for camera images. The laptop to Orin link over Fast DDS
  did work. **Zenoh works locally** (`RMW_IMPLEMENTATION=rmw_zenoh_cpp` plus a
  running `ros2 run rmw_zenoh_cpp rmw_zenohd`); the simulation uses it.
- **Mirrored networking also breaks Gazebo's own transport** ("Exception
  sending a multicast message: Network is unreachable", and no data at all).
  Fix: `export GZ_IP=127.0.0.1`. The `odse_sim` launch file sets it.

### Docker on the Orin

- **`docker exec ... bash -c "..."` does not read `/root/.bashrc`**, so ROS 2
  is not sourced. Source it in the command:
  `docker exec ros2_lyrical bash -c "source /opt/ros/lyrical/setup.bash; ..."`.
  An interactive `docker exec -it ros2_lyrical bash` does read it. That is
  why the Zenoh settings live in the image as `ENV` (`docker/Dockerfile`)
  instead of in `.bashrc`: they apply to every `docker exec`.
- **Anything installed by hand inside a container is lost** when the container
  is recreated. Put it in `docker/Dockerfile` instead.
- **Processes inside the container belong to root**, so `pkill` from the Orin
  shell fails with *"Operation not permitted"*. Stop them from inside:
  `docker exec ros2_lyrical pkill -f "ros2 topic pub"`.
- **Joining the `docker` group needs a new login** (or a reboot) to take effect.
- **`~/ros2_ws` is owned by root**, because Docker created it. Files written in
  the container's `/root/ros2_ws` are root owned on the Orin as well.
- **DNS fails for about a minute after a reboot** (`server misbehaving`), even
  though the Orin is online.

### SSH and the USB link

- **The USB link dropped about every 27 seconds** (timeouts, `Connection
  reset`, transfers stalling after about 100 MB). It looked like the link
  "sleeping", but the cause was NVIDIA's DHCP lease of 15 s: Windows does not
  renew such short leases, so it lost its address over and over. **Fixed** by
  setting the lease to 60 s on the Orin (7 October 2026):
  ```bash
  sudo sed -i 's/^net_dhcp_lease_time=15$/net_dhcp_lease_time=60/' /opt/nvidia/l4t-usb-device-mode/nv-l4t-usb-device-mode-config.sh
  sudo systemctl restart nv-l4t-usb-device-mode.service
  ```
  **A JetPack update may reset it to 15.** Check with
  `grep net_dhcp_lease_time /opt/nvidia/l4t-usb-device-mode/nv-l4t-usb-device-mode-config.sh`.
  Healthy: the Orin's log (`journalctl | grep dhcpd`) shows `DHCPREQUEST`
  every 40 s or so, and no `DHCPDISCOVER` while the cable stays plugged in.
  Details in `docs/rmw_comparison.md`.
- **Run long jobs detached on the Orin anyway**, so a closed laptop or a
  pulled cable cannot kill them:
  `setsid nohup <command> > ~/job.log 2>&1 < /dev/null &`, then check the log.
- **Too many wrong passwords block SSH** for a while; the port then times out
  instead of refusing. Wait a few minutes.

### Running commands from PowerShell

- **Nested quotes break** when a command goes PowerShell, then `ssh`, then
  `bash`, then `docker exec`: PowerShell expands `$(...)` and `$VAR` itself, and
  quotes get lost. Put the commands in a script and pipe it in instead:
  `Get-Content script.sh | ssh fish@192.168.56.1 "tr -d '\r' | bash -s"`.
- **Scripts written on Windows may have CRLF line endings**, which bash
  rejects (`$'\r': command not found`). Strip them with `tr -d '\r'` or
  `sed -i 's/\r$//' script.sh`.
- **`wmic` no longer exists** on current Windows 11. Use PowerShell instead,
  for example `Get-CimInstance Win32_VideoController`.

## Short SSH names

`C:\Users\fisch\.ssh\config` defines (October 2026):

```
Host nvidia_orin             # Ethernet
    HostName 192.168.56.1
    User fish

Host nvidia_orin_usb         # USB-C cable
    HostName 192.168.55.1
    User fish
    ConnectionAttempts 5

Host nvidia_orin_wifi        # Wi-Fi, not reachable from the laptop so far
    HostName 192.168.1.100
    User fish
```

So `ssh nvidia_orin` logs in over Ethernet, and `scp file.txt nvidia_orin:/tmp/`
copies a file.
