"""GPU fit policy and device probing.

llama.cpp sizes GPU offload itself — ``--fit on`` adjusts any argument left
unset to fit device memory, using real introspection at load time. So this
module does **not** decide how many layers to offload; it renders a
:class:`FitPolicy` as arguments, probes the GPU via ``nvidia-smi`` so the app can
*report* what happened, and estimates placement for display.

Reporting uses VRAM figures rather than a layer count: llama.cpp's ``offloaded
X/Y layers`` line is absent at default verbosity and has changed format across
releases, and "2.9 of 4.0 GB VRAM in use" answers the real question anyway.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from infero.childproc import no_window

# llama.cpp's own default margin per device, in MiB.
DEFAULT_FIT_TARGET_MIB = 1024

# Ceiling on how long a device probe may block the caller.
_PROBE_TIMEOUT_S = 5.0

CommandRunner = Callable[[Sequence[str]], str]
"""Runs a command and returns its stdout. Injected so tests need no GPU."""


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    total_mib: int
    used_mib: int
    free_mib: int

    @property
    def total_gb(self) -> float:
        return self.total_mib / 1024

    @property
    def used_gb(self) -> float:
        return self.used_mib / 1024

    @property
    def free_gb(self) -> float:
        return self.free_mib / 1024


@dataclass(frozen=True)
class FitPolicy:
    """How to size a model onto the available devices.

    ``enabled`` maps to llama.cpp's ``--fit``, which only adjusts unset
    arguments — so a pinned ``n_gpu_layers`` or ``ctx_size`` stands while the
    rest is still fitted.
    """

    enabled: bool = True
    target_mib: int = DEFAULT_FIT_TARGET_MIB
    min_ctx: int = 4096
    n_gpu_layers: int | None = None
    ctx_size: int | None = None

    def to_args(self) -> list[str]:
        """Render as ``llama-server`` command-line arguments."""

        args: list[str] = []
        if self.n_gpu_layers is not None:
            args += ["-ngl", str(self.n_gpu_layers)]
        if self.ctx_size is not None:
            args += ["-c", str(self.ctx_size)]

        if not self.enabled:
            return args + ["--fit", "off"]

        args += ["--fit", "on", "--fit-target", str(self.target_mib)]
        # --fit-ctx only constrains a context size that --fit is choosing.
        if self.ctx_size is None:
            args += ["--fit-ctx", str(self.min_ctx)]
        return args

    def describe(self) -> str:
        if not self.enabled:
            pinned = self.n_gpu_layers if self.n_gpu_layers is not None else "default"
            return f"automatic fitting off (n_gpu_layers={pinned})"
        if self.n_gpu_layers is not None:
            return f"{self.n_gpu_layers} layers pinned to GPU, context fitted"
        return f"fitting to device memory, keeping {self.target_mib} MiB free"


#: llama.cpp's own behavior — fit automatically, keep 1 GiB per device free.
DEFAULT_FIT_POLICY = FitPolicy()


def detect_gpus(*, runner: CommandRunner | None = None) -> list[GpuInfo]:
    """Probe NVIDIA GPUs. Returns ``[]`` when none are found or usable.

    No GPU is a normal condition, so every failure path here is non-fatal.
    """

    run = runner or _run_nvidia_smi
    try:
        output = run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ]
        )
    except Exception:
        return []

    gpus: list[GpuInfo] = []
    for index, line in enumerate(output.splitlines()):
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 4:
            continue
        name, total, used, free = fields
        try:
            gpus.append(
                GpuInfo(
                    index=index,
                    name=name,
                    total_mib=int(total),
                    used_mib=int(used),
                    free_mib=int(free),
                )
            )
        except ValueError:
            continue
    return gpus


def _run_nvidia_smi(argv: Sequence[str]) -> str:
    if shutil.which(argv[0]) is None:
        raise FileNotFoundError(argv[0])
    completed = subprocess.run(
        list(argv),
        capture_output=True,
        text=True,
        timeout=_PROBE_TIMEOUT_S,
        check=True,
        # A probe this brief still flashes a console window when the caller is
        # the detached server.
        **no_window(),
    )
    return completed.stdout


def estimate_placement(
    model_size_bytes: int,
    gpus: Sequence[GpuInfo],
    policy: FitPolicy = DEFAULT_FIT_POLICY,
) -> str:
    """A one-line pre-load estimate, shown before a slow load starts.

    Weights against free VRAM less the fit margin, ignoring KV cache and compute
    buffers. llama.cpp decides the real split at load time.
    """

    size_gb = model_size_bytes / 1024**3
    if not gpus:
        return f"{size_gb:.1f} GB model, no GPU detected — CPU only"

    budget_mib = sum(max(0, g.free_mib - policy.target_mib) for g in gpus)
    budget_gb = budget_mib / 1024
    weights_gb = model_size_bytes / 1024**3
    device = gpus[0].name if len(gpus) == 1 else f"{len(gpus)} GPUs"

    if not policy.enabled and policy.n_gpu_layers is None:
        return f"{size_gb:.1f} GB model, {device} — automatic fitting off"
    if policy.n_gpu_layers is not None:
        return (
            f"{size_gb:.1f} GB model, {budget_gb:.1f} GB usable VRAM on {device} — "
            f"{policy.n_gpu_layers} layers pinned to GPU"
        )
    if weights_gb <= budget_gb:
        return (
            f"{size_gb:.1f} GB model, {budget_gb:.1f} GB usable VRAM on {device} — "
            "expect full offload"
        )
    return (
        f"{size_gb:.1f} GB model, {budget_gb:.1f} GB usable VRAM on {device} — "
        "expect partial offload (slower)"
    )
