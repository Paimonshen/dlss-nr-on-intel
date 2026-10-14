#!/usr/bin/env python3
"""Which Vulkan device and queue the network runs on: libxmx's xmx_pick_device and xmx_init.

The choice first, on machines this one is not: an older Intel iGPU before an Arc card,
another vendor's card first, a virtual or a software device, XMX_DEVICE by index and by
name. Then this machine's own devices through xmx_init, in a child process each, with
XMX_DEVICE unset, naming device 0, and naming nothing here. Last the queue: each family
XMX_QUEUE_FAMILY can name and each XMX_QUEUE_PRIORITY, a GEMM checked on every one.
"""
import ctypes
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "gpu"))
import xmx  # noqa: E402

INTEGRATED, DISCRETE, VIRTUAL, CPU = 1, 2, 3, 4    # VkPhysicalDeviceType
ABLE, NONE_ABLE, ASKED, UNMATCHED = range(4)       # XMX_PICK_*
failures = []


def check(name, got, want):
    ok = got == want
    print(f"  {name:<66} {got}  {'OK' if ok else f'FAIL, wanted {want}'}")
    if not ok:
        failures.append(name)


def pick(lib, devices, want=None):
    n = len(devices)
    names = (ctypes.c_char_p * n)(*(name.encode() for name, _, _ in devices))
    types = (ctypes.c_int * n)(*(kind for _, kind, _ in devices))
    able = (ctypes.c_int * n)(*(int(can) for _, _, can in devices))
    how = ctypes.c_int(-1)
    index = lib.xmx_pick_device(n, types, able, names,
                                None if want is None else want.encode(), ctypes.byref(how))
    return index, how.value


def choices():
    print("the choice, on other machines:")
    lib = xmx.native_library("xmx")          # not initialised: no device is touched
    lib.xmx_pick_device.restype = ctypes.c_int
    lnl = ("Intel(R) Arc(TM) 140V GPU (8GB)", INTEGRATED, True)
    uhd = ("Intel(R) UHD Graphics 770", INTEGRATED, False)
    b580 = ("Intel(R) Arc(TM) B580 Graphics", DISCRETE, True)
    b570 = ("Intel(R) Arc(TM) B570 Graphics", DISCRETE, True)
    rtx = ("NVIDIA GeForce RTX 4070", DISCRETE, False)
    llvmpipe = ("llvmpipe (LLVM 19.1.7, 256 bits)", CPU, False)
    check("Lunar Lake alone", pick(lib, [lnl]), (0, ABLE))
    check("an older Intel iGPU before an Arc card", pick(lib, [uhd, b580]), (1, ABLE))
    check("another vendor's card first", pick(lib, [rtx, b580]), (1, ABLE))
    check("a virtual device first", pick(lib, [("Virtual GPU", VIRTUAL, False), lnl]),
          (1, ABLE))
    check("a discrete card before an integrated one, in either order",
          (pick(lib, [lnl, b580]), pick(lib, [b580, lnl])), ((1, ABLE), (0, ABLE)))
    check("two of a kind: the loader's order", pick(lib, [b570, b580]), (0, ABLE))
    check("a software device that could, after a GPU that can",
          pick(lib, [(llvmpipe[0], CPU, True), lnl]), (1, ABLE))
    check("none can: the first that is not a CPU, as before", pick(lib, [llvmpipe, rtx]),
          (1, NONE_ABLE))
    check("none can and all are CPUs: the first", pick(lib, [llvmpipe]), (0, NONE_ABLE))
    check("XMX_DEVICE by index", pick(lib, [uhd, b580], "0"), (0, ASKED))
    check("XMX_DEVICE by part of the name, in any case", pick(lib, [uhd, b580], "uhd"),
          (0, ASKED))
    check("XMX_DEVICE by digits that are a name, not an index",
          (pick(lib, [uhd, b580], "580"), pick(lib, [uhd, b580], "770")),
          ((1, ASKED), (0, ASKED)))
    check("XMX_DEVICE naming a device that cannot: as asked", pick(lib, [rtx, b580], "rtx"),
          (0, ASKED))
    check("XMX_DEVICE naming nothing here: as without it",
          (pick(lib, [uhd, b580], "Radeon"), pick(lib, [uhd, b580], "7")),
          ((1, UNMATCHED), (1, UNMATCHED)))
    check("XMX_DEVICE=1 on one device is no index, and no digit of its name",
          pick(lib, [lnl], "1"), (0, UNMATCHED))
    check("XMX_DEVICE empty: as without it", pick(lib, [uhd, b580], ""), (1, ABLE))


# The child's GEMM: a 64x96 by 96x80 product against NumPy's, its worst relative error printed
# after the device's name.
GEMM = ("import sys; sys.path.insert(0, sys.argv[1]); import numpy as np, xmx; "
        "print(xmx.device_name(), flush=True); r = np.random.default_rng(3); "
        "a = r.standard_normal((64, 96)).astype(np.float16); "
        "b = r.standard_normal((96, 80)).astype(np.float16); "
        "want = a.astype(np.float32) @ b.astype(np.float32); "
        "print(float(np.abs(xmx.gemm(a, b) - want).max() / np.abs(want).max()), flush=True)")


def start(device, **settings):
    """xmx_init in a child, as the daemon runs it; its device name and libxmx's lines, and
    with XMX_QUEUE_* settings the GEMM's error too."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("XMX_")}
    if device is not None:
        env["XMX_DEVICE"] = device
    env.update(settings)
    code = GEMM if settings else ("import sys; sys.path.insert(0, sys.argv[1]); import xmx; "
                                  "print(xmx.device_name(), flush=True)")
    got = subprocess.run([sys.executable, "-c", code, str(ROOT / "src" / "gpu")], env=env,
                         capture_output=True, text=True, timeout=300)
    lines = [line for line in got.stderr.splitlines() if line.startswith("libxmx: ")]
    if settings:
        out = got.stdout.split()
        return got.returncode, float(out[-1]) if got.returncode == 0 else None, lines
    return got.returncode, got.stdout.strip(), lines


def queues():
    print("this machine's queues:")
    # Above medium a driver may not offer the level for the family (Mesa, as Linux showed on
    # 10-14), or refuse it without a privilege (Linux: CAP_SYS_NICE); either way libxmx says so
    # and runs at its default. Intel's Windows driver takes every level.
    high = (", priority high", "queue priority high refused",
            "XMX_QUEUE_PRIORITY=high is not offered for this queue family")
    for settings, said in (
            ({"XMX_QUEUE_PRIORITY": "high"}, high),
            ({"XMX_QUEUE_PRIORITY": "medium"}, ("libxmx: queue family 0, priority medium",)),
            ({"XMX_QUEUE_FAMILY": "compute"}, ("libxmx: queue family ",)),
            ({"XMX_QUEUE_FAMILY": "compute", "XMX_QUEUE_PRIORITY": "high"}, high),
            ({"XMX_QUEUE_FAMILY": "graphics"}, ("libxmx: queue family 0",)),
            ({"XMX_QUEUE_PRIORITY": "urgent"}, ("is none of low, medium, high, realtime",)),
            ({"XMX_QUEUE_FAMILY": "99"}, ("names no compute queue family here",))):
        code, error, lines = start(None, **settings)
        name = " ".join(f"{k}={v}" for k, v in settings.items())
        print(f"    {name}: " + " | ".join(line for line in lines if "queue" in line.lower()))
        check(f"{name}: a correct GEMM", code == 0 and error is not None and error < 1e-2, True)
        check(f"{name}: says so", any(part in line for line in lines for part in said), True)


def this_machine():
    print("this machine, through xmx_init:")
    code, name, lines = start(None)
    check("starts with XMX_DEVICE unset", code, 0)
    listed = [line for line in lines if line.startswith("libxmx: device ")]
    print(f"    on {name}; " + ("; ".join(listed) if listed else "the only device, not listed"))
    if listed:      # more than one device: the choice is said
        check("with several devices, says which runs the network",
              any("runs on device" in line and name in line for line in lines), True)
    check("says nothing is missing on the device it picks",
          [line for line in lines if "although it lacks" in line], [])
    code, asked, lines = start("0")
    check("XMX_DEVICE=0 starts", code, 0)
    first = [line for line in lines if line.startswith("libxmx: device 0: ")]
    print("    " + (first[0] if first else "device 0 not listed"))
    check("device 0 is listed and can run the network",
          bool(first) and first[0].endswith("can run the network"), True)
    code, unmatched, lines = start("no such device")
    check("XMX_DEVICE naming nothing here starts on the same device", (code, unmatched),
          (0, name))
    check("and says it matched nothing",
          any("XMX_DEVICE=no such device matches no device here" in line for line in lines),
          True)


choices()
this_machine()
queues()
print("FAIL: " + ", ".join(failures) if failures else "all device choices OK")
sys.exit(1 if failures else 0)
