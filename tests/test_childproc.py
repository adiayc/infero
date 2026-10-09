"""Child processes must not put a window on the user's desktop.

The failure this guards against is not a crash. `infero run` starts the server
detached, the detached server has no console to lend, and Windows hands every
console child a brand-new one — so loading a model opened an empty terminal
window that stayed until the model was evicted.
"""

from __future__ import annotations

import os
import subprocess

from infero.childproc import detached, no_window
from infero.engine.fit import _run_nvidia_smi, detect_gpus
from infero.engine.manager import SubprocessHandle

WINDOWS = os.name == "nt"


class FakePopen:
    """Captures the arguments a real spawn would have used."""

    last: FakePopen | None = None

    def __init__(self, argv, **kwargs):
        self.argv = list(argv)
        self.kwargs = kwargs
        self.pid = 999
        self.stdout = iter(())
        FakePopen.last = self

    def poll(self):
        return None


class FakeCompleted:
    stdout = ""


# --- the flags themselves ----------------------------------------------------


def test_no_window_is_a_windows_only_concern():
    """POSIX has no console to suppress, so the flags must stay empty there."""

    flags = no_window()

    if WINDOWS:
        assert flags["creationflags"] == subprocess.CREATE_NO_WINDOW
        assert flags["creationflags"] != 0
    else:
        assert flags == {}


def test_no_window_hands_out_a_fresh_dict():
    """It is splatted into Popen calls; a shared dict invites mutation."""

    assert no_window() is not no_window()


def test_detaching_never_leaves_a_process_without_a_console():
    """DETACHED_PROCESS is the bug, not the fix.

    A process with no console does not avoid a window — it defers one. The
    first console program below it (here the uv venv trampoline re-execing the
    interpreter) makes Windows allocate a console, and the default terminal app
    draws it. Its own windowless console detaches just as well and silently.
    """

    flags = detached()

    if not WINDOWS:
        assert flags == {"start_new_session": True}
        return

    creationflags = flags["creationflags"]
    assert creationflags & subprocess.CREATE_NO_WINDOW
    assert creationflags & subprocess.CREATE_NEW_PROCESS_GROUP
    assert not creationflags & subprocess.DETACHED_PROCESS


# --- the call sites ----------------------------------------------------------


def test_the_model_server_is_launched_without_a_window(monkeypatch):
    """llama-server is resident for hours, so its console is not a flash."""

    monkeypatch.setattr("infero.engine.manager.subprocess.Popen", FakePopen)

    SubprocessHandle(["llama-server", "--model", "x.gguf"])

    assert FakePopen.last is not None
    if WINDOWS:
        assert FakePopen.last.kwargs["creationflags"] == subprocess.CREATE_NO_WINDOW
    else:
        assert "creationflags" not in FakePopen.last.kwargs


def test_the_gpu_probe_is_run_without_a_window(monkeypatch):
    """Short-lived, but a window that flashes on every load is still a window."""

    seen: dict = {}

    def fake_run(argv, **kwargs):
        seen.update(kwargs)
        return FakeCompleted()

    monkeypatch.setattr("infero.engine.fit.shutil.which", lambda name: "nvidia-smi")
    monkeypatch.setattr("infero.engine.fit.subprocess.run", fake_run)

    _run_nvidia_smi(["nvidia-smi"])

    if WINDOWS:
        assert seen["creationflags"] == subprocess.CREATE_NO_WINDOW
    else:
        assert "creationflags" not in seen


def test_a_failed_probe_is_still_not_fatal(monkeypatch):
    """The flags must not turn a missing GPU into an error."""

    monkeypatch.setattr("infero.engine.fit.shutil.which", lambda name: None)

    assert detect_gpus() == []
