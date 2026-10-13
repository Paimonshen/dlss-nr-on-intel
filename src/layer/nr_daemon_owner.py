"""Exclusive daemon endpoint ownership, acquired before loading any GPU model.

The endpoint can be absent while the first daemon loads its model. Probing the
pipe/socket therefore cannot prevent concurrent model loads. Hold this lock for
the entire load and serve lifetime. It identifies the actual Python runtime PID;
a Windows venv launcher that stays alive as its parent is not another owner.

Windows uses a non-inheritable named mutex and a zero-timeout wait. An abandoned
mutex is acquired normally after a crash. POSIX uses flock on a stable sibling
file; never unlink that file, since recreating its inode would split the lock.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import sys
import threading
import uuid


class EndpointBusyError(RuntimeError):
    """Another runtime already owns this endpoint, including while loading."""


def normalize_endpoint(endpoint, *, windows=None):
    """Canonical lock key for the transport endpoint, without creating anything."""
    endpoint = os.fspath(endpoint)
    if not isinstance(endpoint, str) or not endpoint or "\0" in endpoint:
        raise ValueError("daemon endpoint must be a nonempty path without NUL")
    if windows is None:
        windows = os.name == "nt"
    if windows:
        # Named pipe names ignore case. Do not use filesystem normpath: pipe names
        # are opaque, so collapsing '..' or trimming spaces would change the key.
        return endpoint.casefold()
    return os.path.realpath(os.path.abspath(endpoint))


def endpoint_key(endpoint, *, windows=None):
    """Stable across interpreters; Python's randomized hash() must not be used."""
    canonical = normalize_endpoint(endpoint, windows=windows)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def mutex_name(endpoint):
    # Named pipes are visible across Windows sessions, so the lock must be too.
    # If creation/opening is denied, fail explicitly rather than silently loading
    # an unprotected model under a different namespace.
    return "Global\\DLSS-NR-owner-v1-" + endpoint_key(endpoint, windows=True)


_registry_pid = os.getpid()
_registry_lock = threading.Lock()
_registry = {}


def _process_registry():
    """Discard copied process-local state after fork (including a copied mutex)."""
    global _registry_pid, _registry_lock, _registry
    pid = os.getpid()
    if pid != _registry_pid:
        _registry_pid = pid
        _registry_lock = threading.Lock()
        _registry = {}
    return _registry_lock, _registry


def _kernel32():
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel.ReleaseMutex.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    return kernel


class EndpointOwner:
    """Nonblocking ownership, held and released by the daemon's serving thread.

    ``acquire()`` returns False for contention. It raises OSError if the OS lock
    cannot be created/used, and never treats those errors as safe ownership.
    ``with EndpointOwner(path) as owner:`` raises EndpointBusyError when busy.
    Repeated acquire/release on one object is idempotent; a different object in
    the same process cannot exploit Win32's recursive mutex ownership.
    """

    def __init__(self, endpoint):
        self.endpoint = os.fspath(endpoint)
        self.canonical = normalize_endpoint(endpoint)
        self.key = endpoint_key(endpoint)
        self.name = mutex_name(endpoint) if os.name == "nt" else self.canonical + ".nr-owner.lock"
        self.session_id = None
        self.abandoned = False
        self._owned = False
        self._pid = None
        self._thread = None
        self._handle = None
        self._kernel = None

    @property
    def owned(self):
        return self._owned and self._pid == os.getpid()

    def _detach_after_fork(self):
        if self._pid is None or self._pid == os.getpid():
            return
        # A forked child must not unlock the parent's flock. Closing its copied
        # descriptor leaves the parent's descriptor and ownership intact.
        if self._handle is not None and os.name != "nt":
            os.close(self._handle)
        self._handle = None
        self._owned = False
        self._pid = self._thread = None
        self.session_id = None
        self.abandoned = False

    def acquire(self):
        self._detach_after_fork()
        if self.owned:
            if self._thread != threading.get_ident():
                raise RuntimeError("daemon endpoint ownership belongs to another thread")
            return True
        lock, registry = _process_registry()
        with lock:
            if self.key in registry:
                return False
            session_id = uuid.uuid4().hex
            if os.name == "nt":
                import ctypes

                kernel = self._kernel = _kernel32()
                handle = kernel.CreateMutexW(None, False, self.name)
                if not handle:
                    raise ctypes.WinError(ctypes.get_last_error())
                result = kernel.WaitForSingleObject(handle, 0)
                if result == 0x102:  # WAIT_TIMEOUT: a live owner, including model load.
                    kernel.CloseHandle(handle)
                    return False
                if result not in (0, 0x80):  # WAIT_OBJECT_0 / WAIT_ABANDONED.
                    error = ctypes.get_last_error()
                    kernel.CloseHandle(handle)
                    raise ctypes.WinError(error)
                self.abandoned = result == 0x80
                self._handle = int(handle)
            else:
                import fcntl

                flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
                flags |= getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(self.name, flags, 0o600)
                try:
                    os.set_inheritable(descriptor, False)
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as error:
                    os.close(descriptor)
                    if error.errno in (errno.EACCES, errno.EAGAIN):
                        return False
                    raise
                self._handle = descriptor
            self._owned = True
            self._pid = os.getpid()
            self._thread = threading.get_ident()
            self.session_id = session_id
            registry[self.key] = self
        return True

    def release(self):
        self._detach_after_fork()
        if not self.owned:
            return
        if self._thread != threading.get_ident():
            raise RuntimeError("daemon endpoint must be released by its acquiring thread")
        lock, registry = _process_registry()
        with lock:
            if os.name == "nt":
                import ctypes

                if not self._kernel.ReleaseMutex(self._handle):
                    # Preserve the handle/state on release failure so the caller
                    # can retry; closing an owned mutex is not a valid unlock.
                    raise ctypes.WinError(ctypes.get_last_error())
                closed = self._kernel.CloseHandle(self._handle)
            else:
                import fcntl

                fcntl.flock(self._handle, fcntl.LOCK_UN)
                os.close(self._handle)
                closed = True
            self._handle = None
            self._owned = False
            self._pid = self._thread = None
            if registry.get(self.key) is self:
                registry.pop(self.key)
            if not closed:
                raise ctypes.WinError(ctypes.get_last_error())

    close = release

    def metadata(self):
        """The runtime actually claiming the endpoint, not a venv launcher PID."""
        return {
            "endpoint": self.endpoint,
            "endpoint_key": self.key,
            "runtime_pid": os.getpid(),
            "parent_pid": os.getppid(),
            "session_id": self.session_id,
            "owned": self.owned,
            "recovered_abandoned_owner": self.abandoned,
            "executable": sys.executable,
            "base_executable": getattr(sys, "_base_executable", sys.executable),
            "python_version": sys.version.split()[0],
            "virtual_environment": sys.prefix != sys.base_prefix,
        }

    def describe(self):
        return "daemon endpoint owner " + json.dumps(self.metadata(), ensure_ascii=True, sort_keys=True)

    def __enter__(self):
        if not self.acquire():
            raise EndpointBusyError(f"another daemon owns endpoint {self.endpoint!r}; model load skipped")
        return self

    def __exit__(self, *_):
        self.release()
