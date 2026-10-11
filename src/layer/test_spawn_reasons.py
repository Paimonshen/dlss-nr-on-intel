#!/usr/bin/env python3
"""Whether the layer starts its daemon, on Windows, and when it does not, that it says why.

A tester's game loaded the layer and started no daemon, with nothing said (issue #12). Here a
Python stands in for the game: it makes one Vulkan instance through Windows' loader with NR's
layer named in VK_INSTANCE_LAYERS and its manifest in VK_LAYER_PATH, and the layer's stderr is
read. A stand-in daemon, which writes its arguments into the daemon's log, shows whether one
was started:
- NR_LAYER_SPAWN unset: nothing started, nothing said (vulkaninfo, a compositor);
- NR_LAYER_SPAWN without NR_LAYER_SOCKET: nothing started, and it says so;
- NR_LAYER_SPAWNED set: nothing started, and it says so, with the mark;
- a daemon already answering on the socket (a pipe this test holds): nothing started, and it
  says so;
- otherwise the stand-in starts, with the socket, settings and root, and the game's process id
  as its mark, and nothing is said.

    python src/layer/test_spawn_reasons.py [path to nr_layer.dll]

The DLL is work/nr_layer.dll, which build_win.bat builds. Exits 77, skipped, off Windows, from a
32-bit Python, or when the DLL is not built.
"""
import ctypes
import json
import os
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import uuid

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FAILURES = []
SAID = "[nr_layer] no daemon started"
# The game: one instance through System32's loader, then its process id and the result.
CHILD = r'''
import ctypes, os, sys
vk = ctypes.WinDLL("vulkan-1.dll")
class Application(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("name", ctypes.c_char_p),
                ("version", ctypes.c_uint32), ("engine", ctypes.c_char_p), ("engineVersion", ctypes.c_uint32),
                ("apiVersion", ctypes.c_uint32)]
class Create(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("application", ctypes.POINTER(Application)), ("layers", ctypes.c_uint32),
                ("layerNames", ctypes.c_void_p), ("extensions", ctypes.c_uint32),
                ("extensionNames", ctypes.c_void_p)]
application = Application(0, None, b"nr spawn test", 1, b"none", 1, (1 << 22) | (3 << 12))
info = Create(1, None, 0, ctypes.pointer(application), 0, None, 0, None)
instance = ctypes.c_void_p()
vk.vkDestroyInstance.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
result = vk.vkCreateInstance(ctypes.byref(info), None, ctypes.byref(instance))
if result == 0:
    vk.vkDestroyInstance(instance, None)
print(os.getpid(), result)
'''
# The daemon, as the layer starts it: what it was given, into the log the layer opened for it.
DAEMON = ('import os, sys; print("started", " ".join(sys.argv[1:]), "mark", '
          'os.environ.get("NR_LAYER_SPAWNED"), flush=True)\n')


def check(name, ok, detail=""):
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(name)


def hold_pipe(name):
    """A named pipe that answers, as a running daemon's does; closed by the caller."""
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateNamedPipeW.restype = wintypes.HANDLE
    kernel32.CreateNamedPipeW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                          wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p)
    # PIPE_ACCESS_DUPLEX; PIPE_TYPE_BYTE | PIPE_WAIT; PIPE_UNLIMITED_INSTANCES
    handle = kernel32.CreateNamedPipeW(name, 0x3, 0x0, 255, 65536, 65536, 0, None)
    if handle in (None, wintypes.HANDLE(-1).value):
        raise OSError(ctypes.get_last_error(), f"CreateNamedPipeW {name}")
    return handle


def close_pipe(handle):
    from ctypes import wintypes
    kernel32 = ctypes.WinDLL("kernel32")
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle(handle)


def run(layer, folder, case):
    """The game's stderr and output, and what the stand-in daemon wrote, for one case."""
    manifest = json.loads((HERE / "VkLayer_dlss_nr.json").read_text(encoding="utf-8"))
    manifest["layer"]["library_path"] = str(layer)
    (folder / "VkLayer_dlss_nr.json").write_text(json.dumps(manifest), encoding="utf-8")
    (folder / "daemon.py").write_text(DAEMON, encoding="utf-8")
    log = folder / "nr_daemon.log"
    log.unlink(missing_ok=True)
    socket = rf"\\.\pipe\nr_spawn_test_{uuid.uuid4().hex[:12]}"
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(("NR_", "VK_")) and key.upper() not in ("ENABLE_NR_LAYER", "DISABLE_NR_LAYER")}
    env.update({"VK_LAYER_PATH": str(folder / "VkLayer_dlss_nr.json"), "VK_INSTANCE_LAYERS": "VK_LAYER_dlssnr_intel",
                "NR_LAYER_SPAWN": "1", "NR_LAYER_SOCKET": socket, "NR_PYTHON": sys.executable,
                "NR_DAEMON": str(folder / "daemon.py"), "NR_LAYER_LOG": str(log),
                "NR_SETTINGS": str(folder / "settings.json"), "NR_ROOT": str(folder)})
    held = None
    if case == "unasked":
        del env["NR_LAYER_SPAWN"]
    elif case == "no socket":
        del env["NR_LAYER_SOCKET"]
    elif case == "marked":
        env["NR_LAYER_SPAWNED"] = "4242"
    elif case == "answering":
        held = hold_pipe(socket)
    try:
        done = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True,
                              timeout=60, cwd=folder)
    finally:
        if held is not None:
            close_pipe(held)
    # The stand-in starts on its own and may still be on its way.
    for _ in range(100):
        if log.is_file() and "started" in log.read_text(encoding="utf-8", errors="replace"):
            break
        time.sleep(0.1)
    written = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else None
    out = done.stdout.split()
    return {"code": done.returncode, "stderr": done.stderr, "pid": out[0] if out else None,
            "result": out[1] if len(out) > 1 else None, "socket": socket, "log": written}


def main():
    if os.name != "nt" or struct.calcsize("P") != 8:
        print("skipped: 64-bit Windows Python only")
        return 77
    layer = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "work" / "nr_layer.dll"
    if not layer.is_file():
        print(f"skipped: no {layer} (tools\\build_win.bat builds it)")
        return 77
    layer = layer.resolve()
    folder = pathlib.Path(tempfile.mkdtemp(prefix="nr spawn (test) "))
    try:
        for case, said in (("unasked", None),
                           ("no socket", "NR_LAYER_SPAWN is set and NR_LAYER_SOCKET is not"),
                           ("marked", "NR_LAYER_SPAWNED=4242 marks this process as one NR started"),
                           ("answering", "one already answers on {socket}")):
            print(f"{case}:")
            seen = run(layer, folder, case)
            check("the instance is made, through the layer", seen["result"] == "0"
                  and "[nr_layer] active" in seen["stderr"], f"result {seen['result']}: {seen['stderr'][-300:]}")
            check("no daemon is started", seen["log"] is None or "started" not in seen["log"], str(seen["log"]))
            lines = [line for line in seen["stderr"].splitlines() if line.startswith(SAID)]
            if said is None:
                check("and nothing is said", not lines, str(lines))
            else:
                check("and it says why", lines == [f"{SAID}: {said.format(socket=seen['socket'])}"], str(lines))

        print("asked, nothing in the way:")
        seen = run(layer, folder, "start")
        check("the instance is made, through the layer", seen["result"] == "0"
              and "[nr_layer] active" in seen["stderr"], f"result {seen['result']}: {seen['stderr'][-300:]}")
        check("nothing is said", SAID not in seen["stderr"], seen["stderr"][-300:])
        expected = (f"started --socket {seen['socket']} --settings {folder / 'settings.json'} --root {folder} "
                    f"mark {seen['pid']}")
        check("the daemon starts, with the game's id as its mark", (seen["log"] or "").strip() == expected,
              repr(seen["log"]))
    finally:
        shutil.rmtree(folder, ignore_errors=True)

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILED: " + ", ".join(FAILURES), flush=True)
        return 1
    print("\nall passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
