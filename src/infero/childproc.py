"""Spawning child processes without conjuring a terminal window.

Windows gives every console program a console. Inheriting the parent's costs
nothing visible, but a process that has *no* console is not spared — the first
console program below it makes Windows allocate a fresh one, and since Windows
11 the default terminal app draws that as a real window on the desktop.

That is the trap in ``DETACHED_PROCESS``: it detaches by leaving the process
with no console at all, so the window simply appears one generation down. The
background server never spawns a console program directly — but the uv venv
``python.exe`` is a trampoline that re-execs the real interpreter, and that is
not our code to pass flags to.

``CREATE_NO_WINDOW`` detaches the same way and is quieter about it: the process
gets its own console with no window, which descendants inherit in silence. It
is a different console from the shell's, so the server still outlives the
terminal that started it, which is the whole point of detaching.
"""

from __future__ import annotations

import os
import subprocess


def no_window() -> dict[str, int]:
    """``Popen``/``run`` arguments for a child that must not draw a window."""

    if os.name != "nt":
        return {}
    return {"creationflags": subprocess.CREATE_NO_WINDOW}


def detached() -> dict[str, object]:
    """``Popen`` arguments for a process that outlives the shell that spawned it.

    ``CREATE_NEW_PROCESS_GROUP`` keeps the shell's Ctrl+C from reaching it; the
    console it gets is its own, so closing the terminal does not close it.
    """

    if os.name != "nt":
        return {"start_new_session": True}
    return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
