"""Aspell baseline wrapper. Preserves suggestion order."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass


def _first_line(args: list[str]) -> str | None:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    text = (proc.stdout or "") + (proc.stderr or "")
    return text.strip().splitlines()[0] if text.strip() else None


def aspell_metadata(lang: str = "en_US") -> dict:
    return {
        "aspell_version": _first_line(["aspell", "--version"]),
        "lang": lang,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "platform": platform.platform(),
        },
    }


@dataclass
class AspellEngine:
    lang: str = "en_US"

    def __post_init__(self) -> None:
        if _first_line(["aspell", "--version"]) is None:
            raise FileNotFoundError("aspell is not installed")
        self._proc = subprocess.Popen(
            ["aspell", "-a", "--lang", self.lang, "--encoding=utf-8"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        assert self._proc.stdout is not None
        self._proc.stdout.readline()

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    def suggest(self, word: str) -> list[str]:
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("aspell process is not running")
        self._proc.stdin.write(f"^{word}\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        # consume trailing blank
        self._proc.stdout.readline()
        line = line.strip()
        if not line or line.startswith("*") or line.startswith("+") or line.startswith("-"):
            return []
        if line.startswith("#"):
            return []
        if line.startswith("&"):
            # & original count offset: s1, s2, ...
            _, _, rest = line.partition(":")
            items = [item.strip() for item in rest.split(",") if item.strip()]
            return items
        return []

    def top1(self, word: str) -> str | None:
        suggestions = self.suggest(word)
        return suggestions[0] if suggestions else None

    def metadata(self) -> dict:
        return aspell_metadata(self.lang)
