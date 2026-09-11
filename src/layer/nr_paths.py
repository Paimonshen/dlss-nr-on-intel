"""Where the layer, the daemon and the tools meet: three paths and the settings file.

Two tools and a daemon have to agree on these. They were written out separately in
`nr-ctl` first, and `nr-toggle` would have been a third copy — this project's own notes
say a fact written down twice drifts, so it is written down once.

Nothing here imports anything but the standard library, so a tool can be a single file
with a shebang and still share this.
"""
import json
import os
import pathlib
import socket

SETTINGS = pathlib.Path(os.environ.get("NR_SETTINGS", "/tmp/nr_settings.json"))
TRIGGER = pathlib.Path(os.environ.get("NR_LAYER_TRIGGER", "/tmp/nr_trigger"))
SOCKET = pathlib.Path(os.environ.get("NR_LAYER_SOCKET", "/tmp/nr_layer.sock"))


def read():
    try:
        with SETTINGS.open() as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def write(values):
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    temporary = SETTINGS.with_suffix(SETTINGS.suffix + ".new")
    # written whole and renamed, so the daemon never reads a half-written file
    with temporary.open("w") as handle:
        json.dump(values, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(SETTINGS)


def alive():
    """Whether a daemon is listening. A stale socket file is not a daemon."""
    if not SOCKET.exists():
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            probe.connect(str(SOCKET))
        return True
    except OSError:
        return False
