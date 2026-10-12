#!/usr/bin/env python3
"""Exercise command-line proxy ownership, rollback and Unicode paths without a GPU/game."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import test_release as fixtures


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("nr_setup_proxy", ROOT / "src/layer/nr_setup_proxy.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)
core = setup.load_core(ROOT)


class ProxyOwnershipTests(unittest.TestCase):
    def setUp(self):
        room = tempfile.TemporaryDirectory(prefix="nr-cli-proxy-")
        self.addCleanup(room.cleanup)
        self.base = Path(room.name)
        self.root = self.base / "release & !Русский!"
        self.game = self.base / "game & !Игра!"
        (self.root / "work").mkdir(parents=True)
        self.game.mkdir()
        (self.root / "nr_vulkan_proxy.dll").write_bytes(b"NR proxy version one")
        exe = self.game / "Игра & test.exe"
        exe.write_bytes(b"executable fixture, never launched")
        self.profile = core.Profile(root=self.root, python=Path(sys.executable), game_exe=exe)
        self.destination = core.installed_path(self.profile)
        self.target = self.game / core.PROXY_NAME
        self.backup = self.destination / core.GAME_BACKUP / core.PROXY_NAME

    def install(self):
        setup.prepare(core, self.profile)
        setup.install(core, self.profile)

    def environment(self):
        return dict(line.split("=", 1) for line in
                    (self.destination / core.ENV_FILE).read_text(encoding="utf-8").splitlines()
                    if "=" in line and not line.startswith("#"))

    def remove(self):
        with mock.patch.object(core, "_install_process_guard", return_value=(True, "fixture")):
            return core.uninstall(self.profile)

    def test_reinstall_without_original_never_backs_up_our_proxy_and_gui_removes_it(self):
        self.install()
        self.install()
        self.assertFalse(self.backup.exists())
        self.assertNotIn("NR_REAL_VULKAN", self.environment())
        self.assertTrue(self.remove()["ok"])
        self.assertFalse(self.target.exists())
        self.assertFalse(self.destination.exists())

    def test_original_loader_survives_reinstall_and_proxy_update_then_gui_restores_it(self):
        self.target.write_bytes(b"game's original loader")
        self.install()
        self.install()
        (self.root / "nr_vulkan_proxy.dll").write_bytes(b"NR proxy version two")
        self.install()
        self.assertEqual(self.backup.read_bytes(), b"game's original loader")
        self.assertEqual(self.target.read_bytes(), b"NR proxy version two")
        self.assertEqual(self.environment()["NR_REAL_VULKAN"], str(self.backup))
        self.assertTrue(self.remove()["ok"])
        self.assertEqual(self.target.read_bytes(), b"game's original loader")

    def test_changed_game_loader_is_refused_before_either_copy_is_touched(self):
        self.target.write_bytes(b"game's original loader")
        self.install()
        self.target.write_bytes(b"a game update replaced the loader")
        env = (self.destination / core.ENV_FILE).read_bytes()
        with self.assertRaisesRegex(ValueError, "changed since NR"):
            setup.prepare(core, self.profile)
        self.assertEqual(self.target.read_bytes(), b"a game update replaced the loader")
        self.assertEqual(self.backup.read_bytes(), b"game's original loader")
        self.assertEqual((self.destination / core.ENV_FILE).read_bytes(), env)

    def test_unowned_installation_and_orphan_backup_are_refused(self):
        self.destination.mkdir()
        private = self.destination / "user-file.txt"
        private.write_text("leave this untouched")
        with self.assertRaisesRegex(ValueError, "not owned"):
            setup.prepare(core, self.profile)
        self.assertEqual(private.read_text(), "leave this untouched")
        self.assertFalse(self.target.exists())
        private.unlink()
        setup.prepare(core, self.profile)
        self.backup.parent.mkdir()
        self.backup.write_bytes(b"unknown backup")
        with self.assertRaisesRegex(ValueError, "unrecorded"):
            setup.install(core, self.profile)
        self.assertEqual(self.backup.read_bytes(), b"unknown backup")
        self.assertFalse(self.target.exists())

    def test_utf8_environment_keeps_unicode_and_shell_metacharacters(self):
        self.install()
        raw = (self.destination / core.ENV_FILE).read_bytes()
        self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
        environment = self.environment()
        self.assertEqual(environment["NR_ROOT"], str(self.root))
        self.assertEqual(environment["NR_GAME_EXE"], str(self.profile.game_exe))
        self.assertEqual(environment["NR_PYTHON"], sys.executable)
        self.assertEqual(environment["VK_LAYER_PATH"], str(self.destination / "VkLayer_dlss_nr.json"))
        self.assertNotIn("NR_LAYER_SPAWNED", environment)

    def test_environment_failure_restores_original_and_allows_retry(self):
        self.target.write_bytes(b"game's original loader")
        setup.prepare(core, self.profile)
        with mock.patch.object(core, "_write_proxy_environment", side_effect=OSError("write failed")):
            with self.assertRaisesRegex(OSError, "write failed"):
                setup.install(core, self.profile)
        self.assertEqual(self.target.read_bytes(), b"game's original loader")
        self.assertFalse(self.backup.exists())
        self.assertFalse((self.destination / core.GAME_FILES).exists())
        self.install()
        self.assertEqual(self.backup.read_bytes(), b"game's original loader")

    def test_update_failure_restores_previous_proxy_record_and_environment(self):
        self.install()
        before = {path: path.read_bytes() for path in
                  (self.target, self.destination / core.GAME_FILES, self.destination / core.ENV_FILE)}
        (self.root / "nr_vulkan_proxy.dll").write_bytes(b"new proxy")
        with mock.patch.object(core, "_write_proxy_environment", side_effect=OSError("write failed")):
            with self.assertRaises(OSError):
                setup.install(core, self.profile)
        for path, contents in before.items():
            self.assertEqual(path.read_bytes(), contents)
        self.assertFalse((self.destination / ".undo").exists())
        self.install()
        self.assertEqual(self.target.read_bytes(), b"new proxy")

    def test_multiple_executables_require_explicit_selection(self):
        second = self.game / "launcher.exe"
        second.write_bytes(b"launcher fixture")
        with self.assertRaisesRegex(ValueError, "Several executables"):
            setup.choose_game(self.game)
        self.assertEqual(setup.choose_game(self.game, self.profile.game_exe.name), self.profile.game_exe)


@unittest.skipUnless(os.name == "nt", "Windows installer requires cmd.exe")
class BatchProxyTests(unittest.TestCase):
    write = fixtures.ReleaseTests.write

    def setUp(self):
        fixtures.ReleaseTests.setUp(self)
        self.target = self.base / "package & !Русский!"
        fixtures.release.assemble(self.root, self.target, "windows")
        weights = self.target / "work/mlxw/dlssnr-logical.safetensors"
        weights.parent.mkdir()
        weights.write_bytes(b"fixture, not weights")
        self.game = self.base / "game & !Игра!"
        self.game.mkdir()
        self.exe = self.game / "game.exe"
        self.exe.write_bytes(b"fixture, never launched")
        self.environment = {key: value for key, value in os.environ.items() if not key.startswith("NR_")}
        self.environment.update(NR_PYTHON=sys.executable, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
        self.destination = self.game / "dlss-nr"

    def run_setup(self, *extra):
        return subprocess.run([
            os.environ.get("COMSPEC", "cmd.exe"), "/d", "/v:on", "/c", "call",
            str(self.target / "setup.bat"), "--game", str(self.game), "--skip-weights", *extra,
        ], capture_output=True, encoding="utf-8", errors="replace", timeout=40,
            env=self.environment, cwd=self.base)

    def test_real_batch_preserves_unicode_ampersand_and_exclamation_paths_on_reinstall(self):
        for _ in range(2):
            got = self.run_setup()
            self.assertEqual(got.returncode, 0, got.stdout + got.stderr)
        environment = dict(line.split("=", 1) for line in
                           (self.destination / "nr-env.txt").read_text(encoding="utf-8").splitlines()
                           if "=" in line and not line.startswith("#"))
        self.assertEqual(environment["NR_ROOT"], str(self.target))
        self.assertEqual(environment["NR_GAME_EXE"], str(self.exe))
        self.assertEqual(environment["NR_PYTHON"], sys.executable)
        self.assertNotIn("NR_REAL_VULKAN", environment)
        self.assertFalse((self.destination / "game-backup/vulkan-1.dll").exists())

    def test_real_batch_refuses_unknown_installation_before_overwriting_its_runtime(self):
        self.destination.mkdir()
        old = self.destination / "nr_layer.dll"
        old.write_bytes(b"unknown user's layer")
        got = self.run_setup()
        self.assertNotEqual(got.returncode, 0)
        self.assertNotIn("Done.", got.stdout)
        self.assertEqual(old.read_bytes(), b"unknown user's layer")
        self.assertFalse((self.game / "vulkan-1.dll").exists())

    def test_real_batch_selects_render_executable_from_multiple_names(self):
        renderer = self.game / "game-vulkan.exe"
        renderer.write_bytes(b"renderer fixture, never launched")
        got = self.run_setup()
        self.assertNotEqual(got.returncode, 0)
        self.assertFalse(self.destination.exists())
        got = self.run_setup("--exe", renderer.name)
        self.assertEqual(got.returncode, 0, got.stdout + got.stderr)
        self.assertIn("NR_GAME_EXE=" + str(renderer),
                      (self.destination / "nr-env.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
