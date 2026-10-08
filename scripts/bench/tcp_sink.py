"""Accept one TCP connection on port 5201 and report how fast data arrived."""

import socket
import time

srv = socket.socket()
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("0.0.0.0", 5201))
srv.listen(1)
srv.settimeout(60)
conn, _ = srv.accept()
total, t0 = 0, time.monotonic()
while chunk := conn.recv(1 << 20):
    total += len(chunk)
span = time.monotonic() - t0
print(f"SINK {total / 1e6:.0f} MB in {span:.1f} s = {total / span / 1e6:.1f} MB/s ({8 * total / span / 1e6:.0f} Mbit/s)")
