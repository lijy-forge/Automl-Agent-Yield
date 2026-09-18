from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yieldmind.safety import RedactRequest, redact_payload
from yieldmind.tools import ToolRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the YieldMind no-network Docker sandbox.")
    parser.add_argument(
        "--allow-docker",
        action="store_true",
        help="Explicitly authorize one transient docker run. No image is pulled automatically.",
    )
    parser.add_argument("--image", default="python:3.11-slim")
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "agent_workspace" / "yieldmind" / "docker_sandbox"),
    )
    args = parser.parse_args()

    started = time.time()
    result = ToolRegistry().execute(
        "run_docker_sandboxed_python",
        {
            "argv": [
                "python",
                "-c",
                (
                    "import socket\n"
                    "print('docker-sandbox-python-ok')\n"
                    "try:\n"
                    "    connection = socket.create_connection(('1.1.1.1', 53), timeout=1)\n"
                    "except OSError:\n"
                    "    print('network-check=blocked')\n"
                    "else:\n"
                    "    connection.close()\n"
                    "    raise RuntimeError('network isolation failed')"
                ),
            ],
            "output_dir": "agent_workspace/yieldmind/docker_runs/smoke",
            "image": args.image,
            "timeout_seconds": 30,
            "allow_docker": args.allow_docker,
        },
    )
    status = "passed" if result.ok else ("skipped" if not args.allow_docker else "failed")
    report = {
        "status": status,
        "mode": result.mode,
        "real_llm_calls": 0,
        "simulated_model_calls": 0,
        "container_runs": 1 if result.result.get("container_name") else 0,
        "duration_seconds": round(time.time() - started, 4),
        "result": result.model_dump(),
    }
    report = redact_payload(RedactRequest(payload=report)).payload
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"yieldmind_docker_sandbox_{time.strftime('%Y%m%d_%H%M%S')}.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
