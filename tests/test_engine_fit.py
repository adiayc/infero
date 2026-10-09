"""GPU fit policy and device probing."""

from __future__ import annotations

from infero.engine.fit import FitPolicy, GpuInfo, detect_gpus, estimate_placement

GB = 1024**3


def test_default_policy_delegates_sizing_to_llama_cpp():
    args = FitPolicy().to_args()

    assert args == ["--fit", "on", "--fit-target", "1024", "--fit-ctx", "4096"]
    # No -ngl: leaving it unset is what lets --fit choose it.
    assert "-ngl" not in args


def test_pinned_layers_coexist_with_fitting():
    """--fit only adjusts unset arguments, so a pin and fitting are compatible."""

    args = FitPolicy(n_gpu_layers=12).to_args()

    assert args[:2] == ["-ngl", "12"]
    assert "--fit" in args and args[args.index("--fit") + 1] == "on"


def test_pinned_context_suppresses_fit_ctx():
    """--fit-ctx bounds a context size llama.cpp is choosing; a pin makes it moot."""

    args = FitPolicy(ctx_size=8192).to_args()

    assert "-c" in args and args[args.index("-c") + 1] == "8192"
    assert "--fit-ctx" not in args


def test_disabled_policy_turns_fitting_off():
    args = FitPolicy(enabled=False, n_gpu_layers=99).to_args()

    assert args == ["-ngl", "99", "--fit", "off"]
    assert "--fit-target" not in args


def test_custom_margin_is_passed_through():
    args = FitPolicy(target_mib=512, min_ctx=2048).to_args()

    assert "--fit-target" in args and args[args.index("--fit-target") + 1] == "512"
    assert "--fit-ctx" in args and args[args.index("--fit-ctx") + 1] == "2048"


def test_describe_explains_the_policy():
    assert "keeping 1024 MiB free" in FitPolicy().describe()
    assert "20 layers pinned" in FitPolicy(n_gpu_layers=20).describe()
    assert "off" in FitPolicy(enabled=False).describe()


# --- device probing --------------------------------------------------------

# A real `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` line.
NVIDIA_SMI_OUTPUT = "NVIDIA GeForce RTX 3050 Laptop GPU, 4096, 565, 3398\n"


def test_detect_gpus_parses_nvidia_smi():
    gpus = detect_gpus(runner=lambda argv: NVIDIA_SMI_OUTPUT)

    assert len(gpus) == 1
    assert gpus[0].name == "NVIDIA GeForce RTX 3050 Laptop GPU"
    assert gpus[0].total_mib == 4096
    assert gpus[0].free_mib == 3398
    assert gpus[0].index == 0


def test_detect_gpus_handles_multiple_devices():
    output = "GPU A, 24576, 1024, 23552\nGPU B, 8192, 512, 7680\n"

    gpus = detect_gpus(runner=lambda argv: output)

    assert [g.index for g in gpus] == [0, 1]
    assert gpus[1].name == "GPU B"


def test_missing_nvidia_smi_is_not_an_error():
    """CPU-only inference is a normal configuration, not a failure."""

    def explode(argv):
        raise FileNotFoundError("nvidia-smi")

    assert detect_gpus(runner=explode) == []


def test_unparseable_lines_are_skipped():
    output = "GPU A, 24576, 1024, 23552\ngarbage line\nGPU B, not-a-number, 1, 2\n"

    gpus = detect_gpus(runner=lambda argv: output)

    assert [g.name for g in gpus] == ["GPU A"]


# --- placement estimate ----------------------------------------------------


def rtx3050(free_mib: int = 3398) -> GpuInfo:
    return GpuInfo(
        index=0,
        name="RTX 3050",
        total_mib=4096,
        used_mib=4096 - free_mib,
        free_mib=free_mib,
    )


def test_estimate_flags_partial_offload():
    """The 7B Q4 on a 4 GB card — the author's actual machine."""

    estimate = estimate_placement(int(4.36 * GB), [rtx3050()])

    assert "partial offload" in estimate
    assert "4.4 GB model" in estimate


def test_estimate_reports_full_offload_when_it_fits():
    estimate = estimate_placement(1 * GB, [rtx3050()])

    assert "full offload" in estimate


def test_estimate_subtracts_the_fit_margin():
    """2.4 GB of weights fits in 3.3 GB free only if the margin is ignored."""

    size = int(2.4 * GB)

    assert "partial offload" in estimate_placement(size, [rtx3050()])
    assert "full offload" in estimate_placement(size, [rtx3050()], FitPolicy(target_mib=64))


def test_estimate_without_a_gpu_says_cpu_only():
    assert "CPU only" in estimate_placement(4 * GB, [])


def test_estimate_reports_pinned_layers():
    estimate = estimate_placement(4 * GB, [rtx3050()], FitPolicy(n_gpu_layers=10))

    assert "10 layers pinned to GPU" in estimate
