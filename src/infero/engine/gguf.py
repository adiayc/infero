"""Minimal GGUF metadata reader.

Answers "what model is this file?" — architecture, layer count, trained context
length — without loading a weight, because filenames are only partly
trustworthy and the layer count in particular has to be exact.

Only the header and metadata block are read. Large arrays (tokenizer
vocabularies, most of the metadata by volume) are seeked past rather than
parsed, and the read stops once the wanted keys have been seen.

Format: ``magic | version | tensor_count | kv_count``, then ``kv_count`` pairs
of ``key | value_type | value``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

MAGIC = b"GGUF"

# Value type tags, per the GGUF spec.
_UINT8 = 0
_INT8 = 1
_UINT16 = 2
_INT16 = 3
_UINT32 = 4
_INT32 = 5
_FLOAT32 = 6
_BOOL = 7
_STRING = 8
_ARRAY = 9
_UINT64 = 10
_INT64 = 11
_FLOAT64 = 12

# tag -> (struct format, byte width) for everything that isn't a string or array.
_SCALARS: dict[int, tuple[str, int]] = {
    _UINT8: ("<B", 1),
    _INT8: ("<b", 1),
    _UINT16: ("<H", 2),
    _INT16: ("<h", 2),
    _UINT32: ("<I", 4),
    _INT32: ("<i", 4),
    _FLOAT32: ("<f", 4),
    _BOOL: ("<?", 1),
    _UINT64: ("<Q", 8),
    _INT64: ("<q", 8),
    _FLOAT64: ("<d", 8),
}

# Keys worth keeping. Architecture-scoped keys are matched by suffix because the
# prefix varies per model family (``qwen2.block_count``, ``llama.block_count``…).
_WANTED_EXACT = frozenset({"general.architecture", "general.name", "general.size_label"})
_WANTED_SUFFIX = (".block_count", ".context_length")

# Keys that, once all present, mean there is nothing left worth scanning for.
_SUFFICIENT = frozenset({"general.architecture", "block_count", "context_length"})


class GGUFError(Exception):
    """Raised when a file is not readable as GGUF."""


@dataclass(frozen=True)
class GGUFInfo:
    """The handful of GGUF metadata fields this app actually uses."""

    version: int
    tensor_count: int
    architecture: str | None = None
    name: str | None = None
    size_label: str | None = None
    block_count: int | None = None
    context_length: int | None = None


class _Reader:
    """Sequential little-endian reader over an open GGUF file."""

    def __init__(self, fh: BinaryIO) -> None:
        self._fh = fh

    def read(self, n: int) -> bytes:
        data = self._fh.read(n)
        if len(data) != n:
            raise GGUFError(f"unexpected end of file (wanted {n} bytes, got {len(data)})")
        return data

    def skip(self, n: int) -> None:
        if n < 0:
            raise GGUFError(f"refusing to skip backwards ({n} bytes)")
        self._fh.seek(n, 1)

    def scalar(self, tag: int) -> int | float | bool:
        fmt, width = _SCALARS[tag]
        return struct.unpack(fmt, self.read(width))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        length = self.u64()
        # Guard against a corrupt length turning into a multi-GB allocation.
        if length > 1 << 24:
            raise GGUFError(f"implausible string length {length}")
        return self.read(length).decode("utf-8", errors="replace")

    def skip_value(self, tag: int) -> None:
        """Advance past a value without materializing it."""

        if tag == _STRING:
            self.skip(self.u64())
        elif tag == _ARRAY:
            self.skip_array()
        elif tag in _SCALARS:
            self.skip(_SCALARS[tag][1])
        else:
            raise GGUFError(f"unknown value type {tag}")

    def skip_array(self) -> None:
        elem_tag = self.u32()
        count = self.u64()
        if elem_tag in _SCALARS:
            # Fixed-width elements: one seek for the whole array.
            self.skip(_SCALARS[elem_tag][1] * count)
        elif elem_tag == _STRING:
            # Variable-width: lengths must be walked, but the bytes are seeked.
            for _ in range(count):
                self.skip(self.u64())
        elif elem_tag == _ARRAY:
            for _ in range(count):
                self.skip_array()
        else:
            raise GGUFError(f"unknown array element type {elem_tag}")

    def read_value(self, tag: int) -> int | float | bool | str:
        if tag == _STRING:
            return self.string()
        if tag in _SCALARS:
            return self.scalar(tag)
        raise GGUFError(f"value type {tag} is not readable as a scalar")


def read_gguf_info(path: str | Path) -> GGUFInfo:
    """Read model metadata from a GGUF file.

    Raises :class:`GGUFError` if the file is missing, truncated, or not GGUF.
    """

    resolved = Path(path)
    try:
        fh = resolved.open("rb")
    except OSError as exc:
        raise GGUFError(f"cannot open {resolved}: {exc}") from exc

    with fh:
        reader = _Reader(fh)
        if reader.read(4) != MAGIC:
            raise GGUFError(f"{resolved} is not a GGUF file (bad magic)")

        version = reader.u32()
        if version < 2:
            # v1 sized its counts differently; no current tooling emits it.
            raise GGUFError(f"unsupported GGUF version {version}")

        tensor_count = reader.u64()
        kv_count = reader.u64()

        found: dict[str, int | float | bool | str] = {}
        for _ in range(kv_count):
            key = reader.string()
            tag = reader.u32()

            suffix = _matching_suffix(key)
            if key in _WANTED_EXACT or suffix:
                try:
                    found[suffix or key] = reader.read_value(tag)
                except GGUFError:
                    # A wanted key with an unexpected (array) type: skip it
                    # rather than fail the whole read.
                    reader.skip_value(tag)
            else:
                reader.skip_value(tag)

            if _SUFFICIENT <= found.keys():
                break

    return GGUFInfo(
        version=version,
        tensor_count=tensor_count,
        architecture=_as_str(found.get("general.architecture")),
        name=_as_str(found.get("general.name")),
        size_label=_as_str(found.get("general.size_label")),
        block_count=_as_int(found.get("block_count")),
        context_length=_as_int(found.get("context_length")),
    )


def _matching_suffix(key: str) -> str | None:
    """Return the bare name of an architecture-scoped key we want, else ``None``."""

    for suffix in _WANTED_SUFFIX:
        if key.endswith(suffix):
            return suffix.lstrip(".")
    return None


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _as_int(value: object) -> int | None:
    # bool is an int subclass; a boolean here would mean a malformed file.
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None
