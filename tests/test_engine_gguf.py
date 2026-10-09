"""GGUF header parsing.

Files are synthesized rather than shipped as fixtures: a real GGUF is gigabytes,
and hand-built bytes pin the exact format the reader claims to support.
"""

from __future__ import annotations

import struct

import pytest

from infero.engine.gguf import GGUFError, read_gguf_info

# Value type tags.
UINT32 = 4
FLOAT32 = 6
STRING = 8
ARRAY = 9


def _gguf_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _encode_value(tag: int, value) -> bytes:
    if tag == STRING:
        return _gguf_string(value)
    if tag == UINT32:
        return struct.pack("<I", value)
    if tag == FLOAT32:
        return struct.pack("<f", value)
    if tag == ARRAY:
        elem_tag, items = value
        out = struct.pack("<I", elem_tag) + struct.pack("<Q", len(items))
        for item in items:
            out += _encode_value(elem_tag, item)
        return out
    raise AssertionError(f"test helper cannot encode tag {tag}")


def build_gguf(kvs: list[tuple[str, int, object]], *, version: int = 3, tensors: int = 7) -> bytes:
    out = b"GGUF" + struct.pack("<I", version)
    out += struct.pack("<Q", tensors) + struct.pack("<Q", len(kvs))
    for key, tag, value in kvs:
        out += _gguf_string(key) + struct.pack("<I", tag) + _encode_value(tag, value)
    return out


def write_gguf(tmp_path, kvs, **kwargs):
    path = tmp_path / "model.gguf"
    path.write_bytes(build_gguf(kvs, **kwargs))
    return path


def test_reads_architecture_layers_and_context(tmp_path):
    path = write_gguf(
        tmp_path,
        [
            ("general.architecture", STRING, "qwen2"),
            ("general.name", STRING, "Qwen2.5 7B Instruct"),
            ("general.size_label", STRING, "7B"),
            ("qwen2.block_count", UINT32, 28),
            ("qwen2.context_length", UINT32, 32768),
        ],
    )

    info = read_gguf_info(path)

    assert info.architecture == "qwen2"
    assert info.name == "Qwen2.5 7B Instruct"
    assert info.size_label == "7B"
    assert info.block_count == 28
    assert info.context_length == 32768
    assert info.version == 3
    assert info.tensor_count == 7


def test_architecture_prefix_is_not_hardcoded(tmp_path):
    """Layer count is found for any architecture, not just the ones we've seen."""

    path = write_gguf(
        tmp_path,
        [
            ("general.architecture", STRING, "gemma3"),
            ("gemma3.block_count", UINT32, 34),
            ("gemma3.context_length", UINT32, 8192),
        ],
    )

    info = read_gguf_info(path)

    assert info.architecture == "gemma3"
    assert info.block_count == 34


def test_skips_large_arrays_to_reach_later_keys(tmp_path):
    """A tokenizer vocabulary must be seeked past, not parsed into memory."""

    vocab = [f"token_{i}" for i in range(5000)]
    path = write_gguf(
        tmp_path,
        [
            ("tokenizer.ggml.tokens", ARRAY, (STRING, vocab)),
            ("tokenizer.ggml.scores", ARRAY, (FLOAT32, [0.5] * 5000)),
            ("general.architecture", STRING, "llama"),
            ("llama.block_count", UINT32, 32),
            ("llama.context_length", UINT32, 4096),
        ],
    )

    info = read_gguf_info(path)

    assert info.architecture == "llama"
    assert info.block_count == 32


def test_stops_early_once_wanted_keys_are_seen(tmp_path):
    """Trailing metadata is never read — that's what keeps listing fast.

    The trailing key here declares an unknown value tag; reaching it would raise.
    """

    path = write_gguf(
        tmp_path,
        [
            ("general.architecture", STRING, "llama"),
            ("llama.block_count", UINT32, 32),
            ("llama.context_length", UINT32, 4096),
            ("trailing.garbage", UINT32, 1),
        ],
    )
    # Corrupt everything after the keys we need; a full scan would notice.
    data = bytearray(path.read_bytes())
    data[-4:] = b"\xff\xff\xff\xff"
    path.write_bytes(bytes(data))

    assert read_gguf_info(path).block_count == 32


def test_rejects_non_gguf_file(tmp_path):
    path = tmp_path / "not-a-model.gguf"
    path.write_bytes(b"PK\x03\x04 this is a zip")

    with pytest.raises(GGUFError, match="not a GGUF file"):
        read_gguf_info(path)


def test_rejects_truncated_file(tmp_path):
    path = tmp_path / "truncated.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<I", 3) + b"\x00\x00")

    with pytest.raises(GGUFError, match="unexpected end of file"):
        read_gguf_info(path)


def test_rejects_unsupported_version(tmp_path):
    path = tmp_path / "v1.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<I", 1) + struct.pack("<Q", 0) * 2)

    with pytest.raises(GGUFError, match="unsupported GGUF version"):
        read_gguf_info(path)


def test_missing_file_raises_gguf_error(tmp_path):
    with pytest.raises(GGUFError, match="cannot open"):
        read_gguf_info(tmp_path / "absent.gguf")


def test_missing_optional_keys_are_none(tmp_path):
    path = write_gguf(tmp_path, [("general.architecture", STRING, "phi3")])

    info = read_gguf_info(path)

    assert info.architecture == "phi3"
    assert info.block_count is None
    assert info.context_length is None
