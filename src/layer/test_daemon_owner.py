"""Exercise endpoint ownership with independent Python runtimes, without a GPU."""
from __future__ import annotations

import ctypes
import json
import os
import pathlib
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
import uuid

from nr_daemon_owner import EndpointBusyError, EndpointOwner, endpoint_key, normalize_endpoint

BASE_PYTHON = getattr(sys, "_base_executable", sys.executable)
TEST_TEMP = pathlib.Path(__file__).resolve().parents[2] / "work" / "test-temp"

def _child(endpoint, mode):
    if mode == "launcher":
        # Like python.org's Windows venv launcher: the visible parent remains
        # alive but only the interpreter running the script claims the endpoint.
        process = subprocess.Popen([BASE_PYTHON, __file__, "--child", endpoint, "hold"],
            stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr,
            **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))
        return process.wait()
    owner = EndpointOwner(endpoint)
    acquired = owner.acquire()
    print(json.dumps({"acquired": acquired, "abandoned": owner.abandoned,
                      "metadata": owner.metadata()}), flush=True)
    if acquired and mode == "hold":
        command = sys.stdin.readline().strip()
        if command == "crash":
            os._exit(23)
    owner.release()
    return 0


class OwnerTests(unittest.TestCase):
    def setUp(self):
        TEST_TEMP.mkdir(parents=True, exist_ok=True)
        self.folder = tempfile.TemporaryDirectory(prefix="nr-owner-test-", dir=TEST_TEMP)
        self.addCleanup(self.folder.cleanup)
        suffix = uuid.uuid4().hex
        self.endpoint = ("\\\\.\\pipe\\nr-owner-test-" + suffix if os.name == "nt"
                         else str(pathlib.Path(self.folder.name) / (suffix + ".sock")))

    def spawn(self, endpoint=None, mode="hold", environment=None, interpreter=None):
        process = subprocess.Popen([interpreter or BASE_PYTHON, __file__, "--child", endpoint or self.endpoint, mode],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", env=environment,
            **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}))

        def cleanup():
            if process.poll() is None:
                try:
                    process.communicate("release\n", timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate(timeout=5)
            else:
                process.communicate(timeout=5)

        self.addCleanup(cleanup)
        ready = queue.Queue()

        def read_ready():
            ready.put(process.stdout.readline())

        thread = threading.Thread(target=read_ready, daemon=True)
        thread.start()
        try:
            line = ready.get(timeout=10)
        except queue.Empty:
            process.kill()
            self.fail("owner subprocess did not report readiness")
        if not line:
            _, error = process.communicate(timeout=5)
            self.fail(f"owner subprocess failed before readiness: {error}")
        return process, json.loads(line)

    def test_competing_runtimes_do_not_both_load(self):
        first, status = self.spawn()
        self.assertTrue(status["acquired"])
        self.assertEqual(status["metadata"]["runtime_pid"], first.pid)
        self.assertEqual(status["metadata"]["parent_pid"], os.getpid())
        self.assertTrue(status["metadata"]["session_id"])
        # The first owner holds only the load reservation; there is no pipe
        # server/socket yet. A second runtime must still be rejected immediately.
        second, status = self.spawn(mode="try")
        self.assertFalse(status["acquired"])
        self.assertEqual(second.wait(timeout=5), 0)

    def test_different_endpoints_are_independent(self):
        first, one = self.spawn()
        second, two = self.spawn(self.endpoint + "-other")
        self.assertTrue(one["acquired"] and two["acquired"])
        self.assertNotEqual(one["metadata"]["endpoint_key"], two["metadata"]["endpoint_key"])

    def test_normal_release_allows_retry(self):
        first, status = self.spawn()
        first.communicate("release\n", timeout=5)
        self.assertEqual(first.returncode, 0)
        second, status = self.spawn(mode="try")
        self.assertTrue(status["acquired"])
        self.assertEqual(second.wait(timeout=5), 0)

    def test_crashed_runtime_allows_retry(self):
        first, status = self.spawn()
        first.communicate("crash\n", timeout=5)
        self.assertEqual(first.returncode, 23)
        second, status = self.spawn(mode="try")
        self.assertTrue(status["acquired"])
        self.assertEqual(second.wait(timeout=5), 0)

    @unittest.skipUnless(os.name == "nt", "Win32 abandoned mutex state")
    def test_wait_abandoned_is_recovered(self):
        from nr_daemon_owner import _kernel32

        first, status = self.spawn()
        owner = EndpointOwner(self.endpoint)
        # Keep an observer handle alive, so killing the sole owner does not
        # destroy and recreate the mutex. Recovery must see WAIT_ABANDONED.
        kernel = _kernel32()
        handle = kernel.CreateMutexW(None, False, owner.name)
        self.assertTrue(handle, f"CreateMutexW failed: {ctypes.get_last_error()}")
        self.addCleanup(kernel.CloseHandle, handle)
        first.communicate("crash\n", timeout=5)
        self.assertEqual(first.returncode, 23)
        self.assertTrue(owner.acquire())
        self.addCleanup(owner.release)
        self.assertTrue(owner.abandoned)
        competing, status = self.spawn(mode="try")
        self.assertFalse(status["acquired"])

    def test_context_releases_on_exception_and_is_idempotent(self):
        owner = EndpointOwner(self.endpoint)
        with self.assertRaisesRegex(ValueError, "model failed"):
            with owner:
                self.assertTrue(owner.owned)
                self.assertTrue(owner.acquire())
                self.assertTrue(owner.acquire())
                raise ValueError("model failed")
        self.assertFalse(owner.owned)
        owner.release()
        owner.close()
        second, status = self.spawn(mode="try")
        self.assertTrue(status["acquired"])
        self.assertEqual(second.wait(timeout=5), 0)

    def test_same_process_cannot_recursively_claim_another_owner(self):
        first = EndpointOwner(self.endpoint)
        with first:
            second = EndpointOwner(self.endpoint)
            self.assertFalse(second.acquire())
            with self.assertRaises(EndpointBusyError):
                with second:
                    self.fail("duplicate reached model load")
        with second:
            self.assertTrue(second.owned)

    def test_releasing_from_another_thread_does_not_drop_ownership(self):
        with EndpointOwner(self.endpoint) as owner:
            failures = []

            def wrong_thread():
                try:
                    owner.release()
                except RuntimeError as error:
                    failures.append(str(error))

            thread = threading.Thread(target=wrong_thread)
            thread.start(); thread.join(timeout=5)
            self.assertEqual(len(failures), 1)
            second, status = self.spawn(mode="try")
            self.assertFalse(status["acquired"])

    def test_venv_style_launcher_is_not_an_endpoint_owner(self):
        launcher, status = self.spawn(mode="launcher")
        self.assertTrue(status["acquired"])
        runtime_pid = status["metadata"]["runtime_pid"]
        self.assertNotEqual(runtime_pid, launcher.pid)
        self.assertEqual(status["metadata"]["parent_pid"], launcher.pid)
        competitor, busy = self.spawn(mode="try")
        self.assertFalse(busy["acquired"])
        launcher.communicate("release\n", timeout=5)
        self.assertEqual(launcher.returncode, 0)
        retry, recovered = self.spawn(mode="try")
        self.assertTrue(recovered["acquired"])
        self.assertEqual(retry.wait(timeout=5), 0)

    @unittest.skipUnless(os.name == "nt" and sys.prefix != sys.base_prefix,
                         "requires a real Windows venv launcher")
    def test_native_venv_launcher_reports_actual_runtime(self):
        launcher, status = self.spawn(interpreter=sys.executable)
        self.assertTrue(status["acquired"])
        self.assertNotEqual(status["metadata"]["runtime_pid"], launcher.pid)
        self.assertEqual(status["metadata"]["parent_pid"], launcher.pid)
        self.assertTrue(status["metadata"]["virtual_environment"])
        competitor, busy = self.spawn(mode="try")
        self.assertFalse(busy["acquired"])
        launcher.communicate("release\n", timeout=5)
        self.assertEqual(launcher.returncode, 0)
        retry, recovered = self.spawn(mode="try")
        self.assertTrue(recovered["acquired"])
        self.assertEqual(retry.wait(timeout=5), 0)

    def test_deterministic_case_insensitive_windows_key(self):
        mixed = "\\\\.\\PIPE\\NR_Mixed-name.sock"
        self.assertEqual(endpoint_key(mixed, windows=True),
                         endpoint_key(mixed.lower(), windows=True))
        self.assertNotEqual(endpoint_key(mixed, windows=True),
                            endpoint_key(mixed + "-other", windows=True))
        self.assertNotEqual(normalize_endpoint(mixed + " ", windows=True),
                            normalize_endpoint(mixed, windows=True))
        # Forward slashes are legal characters within the opaque pipe name.
        self.assertNotEqual(normalize_endpoint(mixed + "/tail", windows=True),
                            normalize_endpoint(mixed + "\\tail", windows=True))
        values = []
        for seed in ("1", "12345"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = seed
            process, result = self.spawn(mode="try", environment=environment)
            self.assertTrue(result["acquired"])
            self.assertEqual(process.wait(timeout=5), 0)
            values.append(result["metadata"]["endpoint_key"])
        self.assertEqual(values[0], values[1])

    @unittest.skipUnless(os.name == "nt", "Windows pipe aliases")
    def test_windows_pipe_case_aliases_share_ownership(self):
        first, status = self.spawn()
        second, status = self.spawn(self.endpoint.upper(), mode="try")
        self.assertFalse(status["acquired"])

    @unittest.skipUnless(os.name != "nt" and hasattr(os, "fork"), "POSIX fork descriptors")
    def test_forked_child_cannot_unlock_parent(self):
        with EndpointOwner(self.endpoint) as owner:
            pid = os.fork()
            if pid == 0:
                try:
                    owner.release()
                    os._exit(0 if not EndpointOwner(self.endpoint).acquire() else 1)
                except BaseException:
                    os._exit(2)
            _, status = os.waitpid(pid, 0)
            self.assertEqual(os.waitstatus_to_exitcode(status), 0)
            second, result = self.spawn(mode="try")
            self.assertFalse(result["acquired"])

    @unittest.skipUnless(os.name != "nt", "POSIX stable flock inode")
    def test_posix_release_preserves_lock_inode(self):
        owner = EndpointOwner(self.endpoint)
        with owner:
            identity = os.stat(owner.name).st_ino
        self.assertEqual(os.stat(owner.name).st_ino, identity)
        with EndpointOwner(self.endpoint):
            self.assertEqual(os.stat(owner.name).st_ino, identity)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        raise SystemExit(_child(sys.argv[2], sys.argv[3]))
    unittest.main()
