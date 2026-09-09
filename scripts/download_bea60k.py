#!/usr/bin/env python3
"""Download the NeuSpell BEA-60K split. Do not commit these files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# NeuSpell train/test release (Google Drive folder
# https://drive.google.com/drive/folders/1ejKSkiHNOlupxXVDMg67rPdqwowsTq1i
# as referenced by neuspell/data/traintest/download_datafiles.py).
BEA_FILES = {
    "test.bea60k": "10VtrEThrDIiuFJf0gj4LeGDdP-y-yR--",
    "test.bea60k.noise": "16AMIb6FVltgRR8xv8h7qacDUX8cOQK9d",
}
SOURCE = {
    "name": "NeuSpell BEA-60K",
    "upstream": "https://github.com/neuspell/neuspell",
    "shared_task": "https://www.cl.cam.ac.uk/research/nl/bea2019st/",
    "drive_folder": "https://drive.google.com/drive/folders/1ejKSkiHNOlupxXVDMg67rPdqwowsTq1i",
    "files": BEA_FILES,
    "redistribution": "Do not commit these files; download locally for evaluation only.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_google_drive(file_id: str, dest: Path) -> None:
    session = requests.Session()
    url = "https://drive.google.com/uc?export=download"
    response = session.get(url, params={"id": file_id, "confirm": "t"}, stream=True, timeout=120)
    response.raise_for_status()
    # Large-file confirm cookie / HTML interstitial
    token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            token = value
    if token or "text/html" in response.headers.get("Content-Type", ""):
        params = {"id": file_id, "confirm": token or "t"}
        response = session.get(url, params=params, stream=True, timeout=120)
        response.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as handle:
        for chunk in response.iter_content(1024 * 64):
            if chunk:
                handle.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "bea60k")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    checksums = {}
    for name, file_id in BEA_FILES.items():
        dest = args.out_dir / name
        if dest.is_file() and not args.force:
            print(f"exists {dest}")
        else:
            print(f"downloading {name}")
            download_google_drive(file_id, dest)
        checksums[name] = {
            "path": str(dest),
            "bytes": dest.stat().st_size,
            "sha256": sha256_file(dest),
        }
        print(f"{name}: {checksums[name]['bytes']} sha256={checksums[name]['sha256']}")

    payload = {"source": SOURCE, "checksums": checksums}
    (args.out_dir / "checksums.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out_dir / 'checksums.json'}")
    print("These files must remain untracked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
