#!/usr/bin/env python3
"""Yield-stress AutoML live dashboard launcher."""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
import urllib.request
import webbrowser


EVENT_LOG = "agent_workspace/live_log.jsonl"
os.environ["AMLA_EVENT_LOG"] = EVENT_LOG
os.environ.setdefault("AMLA_MIRROR_EVENTS_TO_STDOUT", "1")
os.environ["YIELD_DASHBOARD"] = "1"
os.makedirs("agent_workspace", exist_ok=True)


def _is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        try:
            return sock.connect_ex((host, port)) == 0
        except OSError:
            return False


def _dashboard_responds(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=0.8) as resp:
            return 200 <= int(resp.status) < 500
    except Exception:
        return False


def _wait_for_dashboard(port: int, timeout: float = 8.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _dashboard_responds(port):
            return True
        time.sleep(0.2)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5052)
    args = parser.parse_args()

    if _dashboard_responds(args.port):
        url = f"http://localhost:{args.port}"
        print(f"[Yield Dashboard] Existing dashboard is already running at {url}.")
        webbrowser.open(url)
        return 0
    if _is_port_in_use(args.port):
        print(f"[Yield Dashboard] Port {args.port} is occupied by another service.")
        print(f"[Yield Dashboard] Try: python run_yield_live.py --port {args.port + 1}")
        return 1

    from live_dashboard import start_server

    print(f"[Yield Dashboard] Starting at http://localhost:{args.port}")
    start_server(port=args.port)
    if not _wait_for_dashboard(args.port):
        print(f"[Yield Dashboard] Failed to start at http://localhost:{args.port}.")
        return 1
    webbrowser.open(f"http://localhost:{args.port}")
    print("[Yield Dashboard] Running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
