#!/usr/bin/env python3
"""Download WikiText-103 raw for synthetic training-data construction."""

from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Official Salesforce S3 host is gone; use documented mirrors.
# 1) HuggingFace copy of the original zip (same 191,984,949-byte archive).
# 2) Official Salesforce/wikitext parquet (reconstruct wiki.*.raw).
ZIP_URLS = [
    "https://huggingface.co/datasets/mattdangerw/wikitext-103-raw/resolve/main/wikitext-103-raw-v1.zip",
    "https://huggingface.co/datasets/mattdangerw/wikitext-103-raw/resolve/main/wikitext-103-raw-v1.zip?download=true",
]
PARQUET_BASE = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1"
PARQUET_FILES = {
    "wiki.train.raw": [
        "train-00000-of-00002.parquet",
        "train-00001-of-00002.parquet",
    ],
    "wiki.valid.raw": ["validation-00000-of-00001.parquet"],
    "wiki.test.raw": ["test-00000-of-00001.parquet"],
}
EXPECTED_FILES = (
    "wiki.train.raw",
    "wiki.valid.raw",
    "wiki.test.raw",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url} -> {dest}")
    with requests.get(url, stream=True, timeout=120, allow_redirects=True) as resp:
        resp.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in resp.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)


def extract_zip(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    for child in out_dir.rglob("wiki.train.raw"):
        return child.parent
    raise FileNotFoundError("wiki.train.raw not found after extract")


def reconstruct_from_parquet(out_dir: Path) -> Path:
    import pyarrow.parquet as pq

    raw_dir = out_dir / "wikitext-103-raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for dest_name, parts in PARQUET_FILES.items():
        dest = raw_dir / dest_name
        lines: list[str] = []
        for part in parts:
            local = out_dir / "hf_parquet" / part
            if not local.is_file():
                download(f"{PARQUET_BASE}/{part}", local)
            table = pq.read_table(local, columns=["text"])
            lines.extend(str(x) if x is not None else "" for x in table.column("text").to_pylist())
        dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {dest} ({dest.stat().st_size} bytes, {len(lines)} lines)")
    return raw_dir


def find_extracted(raw_dir: Path) -> Path | None:
    for child in raw_dir.rglob("wiki.train.raw"):
        return child.parent
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    extracted = None if args.force else find_extracted(args.out_dir)
    zip_path = args.out_dir / "wikitext-103-raw-v1.zip"

    if extracted is None:
        zip_ok = False
        if zip_path.is_file() and not args.force:
            zip_ok = True
        else:
            last_error: Exception | None = None
            for url in ZIP_URLS:
                try:
                    download(url, zip_path)
                    zip_ok = True
                    last_error = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    print(f"failed {url}: {exc}")
            if not zip_ok:
                print("zip mirrors failed; reconstructing from Salesforce parquet")
                try:
                    extracted = reconstruct_from_parquet(args.out_dir)
                except Exception as parquet_exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"could not download WikiText-103 raw (zip={last_error}; parquet={parquet_exc})"
                    ) from parquet_exc
        if extracted is None:
            extracted = extract_zip(zip_path, args.out_dir)

    for name in EXPECTED_FILES:
        path = extracted / name
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"{name}: {path.stat().st_size} bytes sha256={sha256_file(path)}")
    print(f"wikitext dir: {extracted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
