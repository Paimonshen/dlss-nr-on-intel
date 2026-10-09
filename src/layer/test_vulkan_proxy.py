#!/usr/bin/env python3
"""The vulkan-1.dll proxy that setup puts beside a game (nr_vulkan_proxy.c), on Windows.

Each case loads it in a Python of its own, since its DllMain runs once a process, and the
Python stands in for the game:
- named in nr-env.txt, however spelt, the process gets the variables, the launch is recorded with its pid,
  its integrity level and the daemon log's size, the folder it was started in is listed once, and Vulkan
  calls reach the loader;
- the same at Low integrity, so the level recorded is the process's own: Windows' Vulkan loader ignores
  VK_LAYER_PATH from High up, and setup warns from the record. High itself needs elevation, which a test
  cannot give itself;
- another executable in the same folder gets nothing, and Vulkan still works;
- DISABLE_NR_PROXY=1 turns it all off;
- NR_REAL_VULKAN sends the calls to the loader it names (a game's own copy, set aside).

    python src/layer/test_vulkan_proxy.py [path to nr_vulkan_proxy.dll]

The DLL is work/nr_vulkan_proxy.dll, which build_win.bat builds. Exits 77, skipped, off
Windows, from a 32-bit Python, or when the DLL is not built.
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

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FAILURES = []
LOW = 0x1000  # SECURITY_MANDATORY_LOW_RID
CHILD = r'''
import ctypes, json, os, pathlib, sys
from ctypes import wintypes
folder = pathlib.Path(sys.argv[1]); case = sys.argv[2]
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
advapi32 = ctypes.WinDLL("advapi32")
psapi = ctypes.WinDLL("psapi")
kernel32.GetCurrentProcess.restype = wintypes.HANDLE
advapi32.OpenProcessToken.argtypes = (wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE))
advapi32.GetTokenInformation.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                         ctypes.POINTER(wintypes.DWORD))
advapi32.GetSidSubAuthorityCount.restype = ctypes.POINTER(ctypes.c_ubyte)
advapi32.GetSidSubAuthorityCount.argtypes = (ctypes.c_void_p,)
advapi32.GetSidSubAuthority.restype = ctypes.POINTER(wintypes.DWORD)
advapi32.GetSidSubAuthority.argtypes = (ctypes.c_void_p, wintypes.DWORD)
# This process's integrity level as the proxy should record it: TokenIntegrityLevel's last RID.
token = wintypes.HANDLE(); label = ctypes.create_string_buffer(128); size = wintypes.DWORD()
advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token))
advapi32.GetTokenInformation(token, 25, label, 128, ctypes.byref(size))
sid = ctypes.cast(label, ctypes.POINTER(ctypes.c_void_p))[0]
integrity = advapi32.GetSidSubAuthority(sid, advapi32.GetSidSubAuthorityCount(sid)[0] - 1)[0]
kernel32.GetModuleFileNameW.argtypes = (wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD)
kernel32.GetEnvironmentVariableW.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
psapi.EnumProcessModules.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.HMODULE), wintypes.DWORD,
                                     ctypes.POINTER(wintypes.DWORD))
buffer = ctypes.create_unicode_buffer(32768)
kernel32.GetModuleFileNameW(None, buffer, 32768)
me = buffer.value
spelt = {"other": me + ".not-this-one", "spelling": me.upper()}.get(case, me)
lines = ["NR_GAME_EXE=" + spelt,
         "NR_ROOT=" + str(folder / "release"),
         "NR_LAYER_LOG=" + str(folder / "daemon.log"),
         "NR_LAUNCH_STATE=" + str(folder / "launch-state.json"),
         "NR_TEST_VALUE=it works " + "été"]
if case == "real":
    lines.append("NR_REAL_VULKAN=" + str(folder / "own" / "vulkan-1.dll"))
(folder / "dlss-nr").mkdir(exist_ok=True)
(folder / "dlss-nr" / "nr-env.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
proxy = ctypes.WinDLL(str(folder / "vulkan-1.dll"))
value = ctypes.create_unicode_buffer(256)
got = kernel32.GetEnvironmentVariableW("NR_TEST_VALUE", value, 256)
version = ctypes.c_uint32(0)
result = proxy.vkEnumerateInstanceVersion(ctypes.byref(version))
# Every vulkan-1.dll in the process: the proxy, and the loader it passes calls to.
loaded = []
modules = (wintypes.HMODULE * 1024)(); needed = wintypes.DWORD()
psapi.EnumProcessModules(kernel32.GetCurrentProcess(), modules, ctypes.sizeof(modules), ctypes.byref(needed))
for module in modules[:needed.value // ctypes.sizeof(wintypes.HMODULE)]:
    name = ctypes.create_unicode_buffer(32768)
    kernel32.GetModuleFileNameW(module, name, 32768)
    if name.value.lower().endswith("vulkan-1.dll"):
        loaded.append(name.value)
# Where the proxy looks for the real loader: SysWOW64 for a 32-bit process, by redirection.
system = ctypes.create_unicode_buffer(32768)
kernel32.GetSystemDirectoryW(system, 32768)
seen = json.dumps({"exe": me, "pid": os.getpid(), "integrity": integrity, "value": value.value if got else None,
                   "from_system": any(os.path.samefile(path, os.path.join(system.value, "vulkan-1.dll"))
                                      for path in loaded),
                   "result": result, "version": version.value, "loaded": loaded})
print(seen)
if len(sys.argv) > 3:  # a child started without pipes: its answer as a file
    pathlib.Path(sys.argv[3]).write_text(seen, encoding="utf-8")
'''


def check(name, ok, detail=""):
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(name)


def start_low(arguments, cwd):
    """Runs `arguments` at Low integrity, in this process's environment, and returns its exit
    code. The token is this process's own, duplicated and labelled Low, which needs no privilege
    a user lacks (Windows' sample for a low-integrity process does the same)."""
    from ctypes import wintypes

    class StartupInfo(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR), ("lpDesktop", wintypes.LPWSTR),
                    ("lpTitle", wintypes.LPWSTR), ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
                    ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD), ("dwXCountChars", wintypes.DWORD),
                    ("dwYCountChars", wintypes.DWORD), ("dwFillAttribute", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD), ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD),
                    ("lpReserved2", ctypes.c_void_p), ("hStdInput", wintypes.HANDLE),
                    ("hStdOutput", wintypes.HANDLE), ("hStdError", wintypes.HANDLE)]

    class ProcessInformation(ctypes.Structure):
        _fields_ = [("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
                    ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD)]

    class Label(ctypes.Structure):  # TOKEN_MANDATORY_LABEL: one SID_AND_ATTRIBUTES
        _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    kernel32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.LocalFree.argtypes = (ctypes.c_void_p,)
    advapi32.OpenProcessToken.argtypes = (wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE))
    advapi32.DuplicateTokenEx.argtypes = (wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, ctypes.c_int,
                                          ctypes.c_int, ctypes.POINTER(wintypes.HANDLE))
    advapi32.ConvertStringSidToSidW.argtypes = (wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p))
    advapi32.GetLengthSid.argtypes = (ctypes.c_void_p,)
    advapi32.SetTokenInformation.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    advapi32.CreateProcessAsUserW.argtypes = (
        wintypes.HANDLE, wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p, ctypes.c_void_p, wintypes.BOOL,
        wintypes.DWORD, ctypes.c_void_p, wintypes.LPCWSTR, ctypes.POINTER(StartupInfo),
        ctypes.POINTER(ProcessInformation))
    token, low, sid = wintypes.HANDLE(), wintypes.HANDLE(), ctypes.c_void_p()
    try:
        # TOKEN_ASSIGN_PRIMARY | TOKEN_DUPLICATE | TOKEN_QUERY | TOKEN_ADJUST_DEFAULT; a primary token
        if not (advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0001 | 0x0002 | 0x0008 | 0x0080,
                                          ctypes.byref(token))
                and advapi32.DuplicateTokenEx(token, 0, None, 2, 1, ctypes.byref(low))
                and advapi32.ConvertStringSidToSidW("S-1-16-4096", ctypes.byref(sid))):
            raise OSError(ctypes.get_last_error(), "no Low token")
        label = Label(sid, 0x20)  # SE_GROUP_INTEGRITY
        if not advapi32.SetTokenInformation(low, 25, ctypes.byref(label),  # TokenIntegrityLevel
                                            ctypes.sizeof(label) + advapi32.GetLengthSid(sid)):
            raise OSError(ctypes.get_last_error(), "the token kept its level")
        startup, process = StartupInfo(), ProcessInformation()
        startup.cb = ctypes.sizeof(startup)
        command = ctypes.create_unicode_buffer(subprocess.list2cmdline(arguments))
        if not advapi32.CreateProcessAsUserW(low, None, command, None, None, False, 0x08000000,  # no window
                                             None, str(cwd), ctypes.byref(startup), ctypes.byref(process)):
            raise OSError(ctypes.get_last_error(), "CreateProcessAsUserW")
        try:
            if kernel32.WaitForSingleObject(process.hProcess, 60000):
                kernel32.TerminateProcess(process.hProcess, 1)
                return "timed out"
            code = wintypes.DWORD()
            kernel32.GetExitCodeProcess(process.hProcess, ctypes.byref(code))
            return code.value
        finally:
            kernel32.CloseHandle(process.hProcess)
            kernel32.CloseHandle(process.hThread)
    finally:
        for handle in (low, token):
            if handle:
                kernel32.CloseHandle(handle)
        if sid:
            kernel32.LocalFree(sid)


def run(proxy, case, environment=None, runs=1, low=False):
    """What the last of `runs` processes saw, its launch record, and for each folder listed in
    start-folders.txt whether it is the one they were started in (None without the file).

    `low` runs it at Low integrity, in this process's environment and in a folder under
    LocalLow, the one place such a process may write."""
    base = pathlib.Path(os.environ["USERPROFILE"]) / "AppData" / "LocalLow" if low else None
    folder = pathlib.Path(tempfile.mkdtemp(prefix="nr proxy (test) ", dir=base))
    try:
        shutil.copyfile(proxy, folder / "vulkan-1.dll")
        (folder / "daemon.log").write_bytes(b"x" * 1234)
        if case == "real":
            (folder / "own").mkdir()
            shutil.copyfile(pathlib.Path(os.environ["SystemRoot"]) / "System32" / "vulkan-1.dll",
                            folder / "own" / "vulkan-1.dll")
        started = folder / "started here"
        started.mkdir()
        env = dict(os.environ, **(environment or {}))
        env.pop("NR_TEST_VALUE", None)
        for _ in range(runs):
            if low:
                answer = folder / "seen.json"
                code = start_low([sys.executable, "-c", CHILD, str(folder), case, str(answer)], started)
                if code or not answer.exists():
                    return {"error": f"exit code {code}, no answer"}, None, None
                seen = json.loads(answer.read_text(encoding="utf-8"))
                continue
            done = subprocess.run([sys.executable, "-c", CHILD, str(folder), case], env=env, cwd=started,
                                  capture_output=True, text=True, timeout=60)
            if done.returncode:
                return {"error": done.stderr.strip()[-400:]}, None, None
            seen = json.loads(done.stdout.strip().splitlines()[-1])
        state = folder / "launch-state.json"
        record = json.loads(state.read_text(encoding="utf-8")) if state.exists() else None
        listed = folder / "dlss-nr" / "start-folders.txt"
        # Compared as files while they exist: MSYS2's Python joins paths with /.
        folders = ([os.path.isdir(line) and os.path.samefile(line, started)
                    for line in listed.read_text(encoding="utf-8").splitlines()]
                   if listed.exists() else None)
        return seen, record, folders
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def main():
    if os.name != "nt" or struct.calcsize("P") != 8:
        print("skipped: 64-bit Windows Python only")
        return 77
    proxy = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "work" / "nr_vulkan_proxy.dll"
    if not proxy.is_file():
        print(f"skipped: no {proxy} (tools\\build_win.bat builds it)")
        return 77

    print("the game's own process, started twice:")
    seen, record, folders = run(proxy, "game", runs=2)
    check("it ran", "error" not in seen, seen.get("error", ""))
    check("the variable is set, UTF-8 read right", seen.get("value") == "it works été", repr(seen.get("value")))
    check("a Vulkan call reaches the loader", seen.get("result") == 0 and seen.get("version", 0) >= (1 << 22),
          f"result {seen.get('result')}, version {seen.get('version')}")
    # Compared as files, in the child while they exist: MSYS2's Python joins paths with /.
    check("System32's loader is the one behind it", seen.get("from_system") is True, str(seen.get("loaded")))
    check("the launch is recorded", record is not None and record.get("pid") == seen.get("pid")
          and record.get("daemon_start_bytes") == 1234 and record.get("exit_code") is None
          and record.get("game_exe") == seen.get("exe"), str(record))
    check("with the process's integrity level", isinstance(seen.get("integrity"), int)
          and record is not None and record.get("integrity") == seen.get("integrity"),
          f"{record and record.get('integrity')} for {seen.get('integrity')}")
    check("the folder it was started in is listed, once", folders == [True], str(folders))

    print("the game's own process at Low integrity:")
    seen, record, folders = run(proxy, "game", low=True)
    check("it ran at Low, with the variable", seen.get("integrity") == LOW and seen.get("value") == "it works été",
          seen.get("error") or f"{seen.get('integrity')}, {seen.get('value')!r}")
    check("the record says Low", record is not None and record.get("integrity") == LOW, str(record))

    print("the game's path spelt otherwise (upper case):")
    seen, record, folders = run(proxy, "spelling")
    check("it is still the game", seen.get("value") == "it works été" and record is not None,
          repr(seen.get("value")))

    print("another executable in the folder:")
    seen, record, folders = run(proxy, "other")
    check("nothing is set", "error" not in seen and seen.get("value") is None, str(seen.get("value")))
    check("Vulkan still works", seen.get("result") == 0)
    check("no launch is recorded, no folder listed", record is None and folders is None)

    print("DISABLE_NR_PROXY=1:")
    seen, record, folders = run(proxy, "game", {"DISABLE_NR_PROXY": "1"})
    check("nothing is set, nothing recorded", seen.get("value") is None and record is None
          and folders is None)
    check("Vulkan still works", seen.get("result") == 0)

    print("the game's own loader, set aside:")
    seen, record, folders = run(proxy, "real")
    check("calls go to the loader NR_REAL_VULKAN names",
          seen.get("result") == 0 and any(path.lower().endswith("\\own\\vulkan-1.dll") for path in seen.get("loaded", [])),
          str(seen.get("loaded")))

    if FAILURES:
        print(f"\n{len(FAILURES)} FAILED: " + ", ".join(FAILURES), flush=True)
        return 1
    print("\nall passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
