"""Deterministic Hunspell wrapper. Preserves suggestion order. NFC equality."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from spelling_reranker.byte_encoding import N_CANDIDATE_SLOTS, nfc
from spelling_reranker.candidates import first_ten_pool

DEFAULT_DIC = Path("/usr/share/hunspell/en_US.dic")
DEFAULT_AFF = Path("/usr/share/hunspell/en_US.aff")
#: Hunspell's suggestion list is kept to the width of the model's candidate
#: slots rather than truncated at 10. Most typos yield fewer suggestions than
#: this, so the extra slots are usually free.
MAX_CANDIDATES = N_CANDIDATE_SLOTS


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cmd_version(args: Sequence[str]) -> str | None:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    text = (proc.stdout or "") + (proc.stderr or "")
    return text.strip().splitlines()[0] if text.strip() else None


def collect_hunspell_metadata(
    dic_path: Path = DEFAULT_DIC,
    aff_path: Path = DEFAULT_AFF,
) -> dict:
    hunspell_version = _cmd_version(["hunspell", "-v"])
    try:
        import hunspell as hunspell_mod

        python_binding = getattr(hunspell_mod, "__file__", "hunspell")
    except ImportError:
        python_binding = None
    metadata = {
        "hunspell_cli_version": hunspell_version,
        "python_binding": python_binding,
        "dictionary_package": "hunspell-en-us",
        "dictionary_paths": {
            "dic": str(dic_path),
            "aff": str(aff_path),
        },
        "dictionary_hashes": {},
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }
    if dic_path.is_file():
        metadata["dictionary_hashes"]["en_US.dic"] = {
            "path": str(dic_path),
            "sha256": _sha256_file(dic_path),
            "size": dic_path.stat().st_size,
        }
    if aff_path.is_file():
        metadata["dictionary_hashes"]["en_US.aff"] = {
            "path": str(aff_path),
            "sha256": _sha256_file(aff_path),
            "size": aff_path.stat().st_size,
        }
    return metadata


def write_hunspell_metadata(path: Path, metadata: dict | None = None) -> dict:
    payload = metadata if metadata is not None else collect_hunspell_metadata()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


@dataclass
class HunspellEngine:
    """Fixed en_US Hunspell dictionary. Suggestion order is preserved."""

    dic_path: Path = DEFAULT_DIC
    aff_path: Path = DEFAULT_AFF
    max_candidates: int = MAX_CANDIDATES

    def __post_init__(self) -> None:
        self.dic_path = Path(self.dic_path)
        self.aff_path = Path(self.aff_path)
        if not self.dic_path.is_file() or not self.aff_path.is_file():
            raise FileNotFoundError(
                f"Hunspell dictionary not found: {self.dic_path} / {self.aff_path}"
            )
        try:
            import hunspell as hunspell_mod
        except ImportError as exc:
            raise ImportError(
                "The 'hunspell' Python package is required. "
                "Install libhunspell-dev and `pip install hunspell`."
            ) from exc
        self._handle = hunspell_mod.HunSpell(str(self.dic_path), str(self.aff_path))

    def spell(self, word: str) -> bool:
        return bool(self._handle.spell(nfc(word)))

    def suggest_raw(self, word: str) -> list[str]:
        """Hunspell suggestions in engine order, before dedup or truncation."""
        raw = self._handle.suggest(nfc(word)) or []
        out: list[str] = []
        for item in raw:
            if isinstance(item, bytes):
                item = item.decode("utf-8", errors="replace")
            out.append(item)
        return out

    def suggest(self, word: str) -> list[str]:
        raw = self.suggest_raw(word)
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            normalized = nfc(item)
            if normalized in seen:
                continue
            seen.add(normalized)
            out.append(normalized)
            if len(out) >= self.max_candidates:
                break
        return out

    def first_ten(self, typo: str) -> list[str]:
        """Raw first ten Hunspell suggestions, then NFC-dedup within that slice."""
        return first_ten_pool(self.suggest_raw(typo))

    def candidates(self, typo: str) -> list[str]:
        return self.suggest(typo)[: self.max_candidates]

    def gold_index(self, candidates: Sequence[str | None], gold: str) -> int | None:
        target = nfc(gold)
        for idx, cand in enumerate(candidates):
            if cand is None:
                continue
            if nfc(cand) == target:
                return idx
        return None

    def metadata(self) -> dict:
        return collect_hunspell_metadata(self.dic_path, self.aff_path)


def default_engine() -> HunspellEngine:
    dic = Path(os.environ.get("HUNSPELL_DIC", DEFAULT_DIC))
    aff = Path(os.environ.get("HUNSPELL_AFF", DEFAULT_AFF))
    return HunspellEngine(dic_path=dic, aff_path=aff)
