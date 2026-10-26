"""Run the driving policy on the Orin while Gazebo runs on the laptop (Windows).

    python scripts\\orin_policy_windows.py deploy                  # copy and build odse_interfaces, odse_driving
    python scripts\\orin_policy_windows.py run                     # PPO, ball ahead
    python scripts\\orin_policy_windows.py run --model ppo_search
    python scripts\\orin_policy_windows.py run --target yolo      # YOLO and target node too
    python scripts\\orin_policy_windows.py stop

deploy copies the two packages from odse_ws/src to ~/ros2_ws/src on the Orin
(mounted in the ros2_lyrical container as /root/ros2_ws) and runs colcon build
in the container. Run it again after changing the code or the models.

run starts policy.launch.py in the container and shows its output; Ctrl+C
stops it. Then start the simulation in WSL:

    ros2 launch odse_driving experiment.launch.py policy:=orin
    ros2 launch odse_driving experiment.launch.py policy:=orin target:=orin   # with --target yolo

The nodes can stay running across experiments. Runs from Windows because the
SSH key for the Orin is there, not in WSL (docs/orin_ssh_docker.md).
"""

import argparse
import io
import subprocess
import sys
import tarfile
from pathlib import Path

HOST = "fish@192.168.56.1"
CONTAINER = "ros2_lyrical"
PACKAGES = ["odse_interfaces", "odse_driving"]
SRC = Path(__file__).resolve().parent.parent / "odse_ws" / "src"
SETUP = "source /opt/ros/lyrical/setup.bash && cd /root/ros2_ws"
# Brackets keep pkill from matching the bash -c command that runs it
NODE_PATTERN = '"[p]olicy.launch.py|[o]dse_driving/(policy|yolo|target)"'


def ssh(command: str, tty: bool = False, stdin: bytes | None = None) -> int:
    args = ["ssh", *(["-t"] if tty else []), HOST, command]
    return subprocess.run(args, input=stdin).returncode


def in_container(command: str, tty: bool = False) -> str:
    flags = "-it" if tty else ""
    return f"docker exec {flags} {CONTAINER} bash -c '{command}'"


def deploy() -> int:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for package in PACKAGES:
            tar.add(SRC / package, arcname=package, filter=lambda t: None if "__pycache__" in t.name else t)
    print(f"copying {', '.join(PACKAGES)} ({buffer.tell() / 1e3:.0f} kB)")
    # Unpacked inside the container: Docker created ~/ros2_ws, so root owns it
    remove = " ".join(f"src/{p}" for p in PACKAGES)
    unpack = f"cd /root/ros2_ws && rm -rf {remove} && mkdir -p src && tar xzf - -C src --no-same-owner"
    if ssh(f"docker exec -i {CONTAINER} bash -c '{unpack}'", stdin=buffer.getvalue()):
        return 1
    build = f"{SETUP} && colcon build --symlink-install --packages-select {' '.join(PACKAGES)}"
    return ssh(in_container(build))


def run(model: str, start_delay: float, target: str) -> int:
    stop()
    launch = (
        f"{SETUP} && source install/setup.bash && "
        f"ros2 launch odse_driving policy.launch.py model:={model} start_delay:={start_delay} target:={target}"
    )
    return ssh(in_container(launch, tty=True), tty=True)


def stop() -> int:
    """Stop policy, YOLO and target nodes left running, for example after a lost SSH connection.

    The nodes can take several seconds to end after SIGTERM, and a node left
    over from an earlier run breaks the next experiment (section 9.5 of
    docs/how_it_works.md), so wait up to 10 s and then kill what is left.
    """
    command = (
        f"pkill -f {NODE_PATTERN}; "
        f"for i in $(seq 20); do pgrep -f {NODE_PATTERN} > /dev/null || exit 0; sleep 0.5; done; "
        f"pkill -9 -f {NODE_PATTERN}; true"
    )
    return ssh(in_container(command))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["deploy", "run", "stop"])
    parser.add_argument("--model", default="ppo", help="ppo, ppo_search, bc, ppo_h4, ppo_search_h4 or an ONNX file name")
    parser.add_argument("--start-delay", type=float, default=5.0, help="simulated seconds before the car starts")
    parser.add_argument("--target", choices=["sim", "yolo"], default="sim",
                        help="yolo: also run YOLO and the target node on the Orin (laptop: target:=orin)")
    args = parser.parse_args()
    if args.action == "deploy":
        return deploy()
    if args.action == "run":
        return run(args.model, args.start_delay, args.target)
    return stop()


if __name__ == "__main__":
    sys.exit(main())
