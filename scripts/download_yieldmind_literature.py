#!/usr/bin/env python3
"""Download and verify the curated YieldMind open-literature corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yieldmind.literature import download_literature, load_literature_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "knowledge_sources" / "literature" / "manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "knowledge_sources" / "literature" / "documents"),
    )
    args = parser.parse_args()
    manifest = load_literature_manifest(args.manifest)
    results = download_literature(manifest, args.output_dir)
    print(json.dumps({"documents": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
