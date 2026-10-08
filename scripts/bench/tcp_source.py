"""Send data over TCP to the Orin's port 5201 for 20 s, printing MB/s every 0.5 s."""

import socket
import sys
import time

host = sys.argv[1] if len(sys.argv) > 1 else "192.168.55.1"
block = bytes(64 << 10)
s = socket.create_connection((host, 5201), timeout=10)
t0 = last = time.monotonic()
total = window = 0
samples = []
try:
    while time.monotonic() - t0 < 20:
        s.sendall(block)
        total += len(block)
        window += len(block)
        now = time.monotonic()
        if now - last >= 0.5:
            samples.append(window / (now - last) / 1e6)
            window, last = 0, now
except OSError as e:
    print(f"error after {time.monotonic() - t0:.1f} s: {e}")
s.close()
span = time.monotonic() - t0
print("MB/s per 0.5 s:", " ".join(f"{x:.0f}" for x in samples))
print(f"SOURCE {total / 1e6:.0f} MB in {span:.1f} s = {total / span / 1e6:.1f} MB/s")
