import sys
import os
import socket

# Ensure src is importable as a module from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from src.backend import app


def find_available_port(preferred_port: int, fallbacks=None):
    """Return the first free localhost port, starting from the preferred port."""
    candidates = list(fallbacks or [])
    if preferred_port not in candidates:
        candidates.insert(0, preferred_port)

    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue

    raise RuntimeError("No free localhost port was available for the EcoSense server.")

if __name__ == "__main__":
    preferred_port = int(os.getenv("PORT", "8000"))
    fallback_ports = [8001, 8002, 8010, 8080, 8082, 9000]
    selected_port = find_available_port(preferred_port, fallback_ports)

    uvicorn.run(
        "src.backend:app",
        host="127.0.0.1",
        port=selected_port,
        reload=False,
        log_level="info"
    )
