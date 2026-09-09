#!/usr/bin/env python3
"""Download a running pod's artifacts over Runpod's HTTP proxy.

The pod serves /workspace/out on port 8000 (see bootstrap.sh), which is the
only channel back when the controlling environment has outbound HTTPS only.

    scripts/runpod/fetch_artifacts.py <pod-id> --dest .
"""

from __future__ import annotations

import argparse
import posixpath
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import requests

LINK_RE = re.compile(r'<a href="([^"?][^"]*)"')
#: Skip the per-error dump: hundreds of MB and not needed off-box.
SKIP = {"predictions.jsonl"}


def listing(base: str, path: str) -> list[str]:
    response = requests.get(posixpath.join(base, path), timeout=120)
    response.raise_for_status()
    return [unquote(m) for m in LINK_RE.findall(response.text)]


def walk(base: str, path: str, dest: Path, *, quiet: bool) -> int:
    count = 0
    for entry in listing(base, path):
        if entry in ("../", "./"):
            continue
        child = posixpath.join(path, entry) if path else entry
        if entry.endswith("/"):
            count += walk(base, child, dest, quiet=quiet)
            continue
        if entry in SKIP:
            continue
        target = dest / child
        target.parent.mkdir(parents=True, exist_ok=True)
        url = posixpath.join(base, child)
        with requests.get(url, stream=True, timeout=600) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                for block in response.iter_content(1 << 20):
                    handle.write(block)
        if not quiet:
            print(f"  {child}  ({target.stat().st_size:,} bytes)")
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pod_id")
    parser.add_argument("--dest", type=Path, default=Path("."))
    parser.add_argument("--subdir", default="", help="only fetch under this path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    base = f"https://{args.pod_id}-8000.proxy.runpod.net"
    try:
        n = walk(base, args.subdir, args.dest, quiet=args.quiet)
    except requests.HTTPError as exc:
        print(f"cannot reach {base}: {exc}", file=sys.stderr)
        return 1
    print(f"fetched {n} file(s) into {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
