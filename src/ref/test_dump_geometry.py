#!/usr/bin/env python3
"""Raw dumps survive resizing; conversion preserves source data on failure. CPU only.

The codecs are substituted with lossless array storage so these workflow checks require no
ImageMagick or GPU. test_dump_raw.py separately checks the real PNG codec's quantisation.
"""
import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "tools"))

import image_io
import comparisons
import dump_to_png


def quantised(array):
    return (np.clip(array, 0, 1) * 255 + 0.5).astype(np.uint8)


def save_codec(array, path):
    with open(path, "wb") as handle:
        np.save(handle, quantised(array), allow_pickle=False)


def load_codec(path):
    with open(path, "rb") as handle:
        return np.load(handle, allow_pickle=False).astype(np.float32) / 255


class DumpGeometry(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="nr_dump_geometry_")
        self.addCleanup(temporary.cleanup)
        self.folder = pathlib.Path(temporary.name)

    def write(self, name, shape):
        image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
        path = self.folder / name
        image_io.save_raw(image, path, metadata=True)
        return path, image

    def convert(self, *options, save=save_codec, load=load_codec):
        with mock.patch.object(sys, "argv", ["dump_to_png.py", str(self.folder), *options]), \
                mock.patch.object(image_io, "save", side_effect=save), \
                mock.patch.object(image_io, "load", side_effect=load), \
                contextlib.redirect_stdout(io.StringIO()):
            return dump_to_png.main()

    def assertPixels(self, expected, actual):
        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_array_equal(np.rint(actual * 255).astype(np.uint8), expected)

    def test_same_area_resize_uses_each_frames_geometry(self):
        # The old directory-level sidecar silently reshaped the first 2x6 frame to 3x4.
        for number, shape in ((1, (2, 6, 3)), (2, (3, 4, 3)), (3, (2, 3, 3))):
            path, image = self.write(f"{number:03d}_in.raw", shape)
            self.write(f"{number:03d}_out.raw", shape)
            (self.folder / "dump.json").write_text(json.dumps(
                {"width": shape[1], "height": shape[0], "per_frame_geometry": True}))
            self.assertPixels(image, image_io.load_raw(path))
            self.assertPixels(image, comparisons._frame(self.folder, number, "in"))
        self.assertEqual(self.convert(), 0)
        for number, shape in ((1, (2, 6, 3)), (2, (3, 4, 3)), (3, (2, 3, 3))):
            for side in ("in", "out"):
                path = self.folder / f"{number:03d}_{side}.raw"
                self.assertFalse(path.exists())
                self.assertFalse(image_io.raw_metadata_path(path).exists())
                self.assertPixels(np.arange(np.prod(shape), dtype=np.uint8).reshape(shape),
                                  load_codec(path.with_suffix(".png")))

    def test_legacy_directory_metadata_reads_and_keeps_original(self):
        image = np.arange(36, dtype=np.uint8).reshape(2, 6, 3)
        path = self.folder / "001_in.raw"
        image_io.save_raw(image, path)
        (self.folder / "dump.json").write_text(json.dumps({"width": 6, "height": 2}))
        self.assertPixels(image, image_io.load_raw(path))
        self.assertPixels(image, comparisons._frame(self.folder, 1, "in"))
        self.assertEqual(self.convert(), 0)
        self.assertTrue(path.exists())
        self.assertPixels(image, load_codec(path.with_suffix(".png")))

    def test_bad_sidecar_never_falls_back_to_latest_geometry(self):
        path, _ = self.write("001_in.raw", (2, 6, 3))
        (self.folder / "dump.json").write_text(json.dumps({"width": 4, "height": 3}))
        image_io.raw_metadata_path(path).write_text("{")
        with self.assertRaisesRegex(ValueError, "cannot read raw geometry"):
            comparisons._frame(self.folder, 1, "in")
        self.assertEqual(self.convert(), 1)
        self.assertTrue(path.exists())
        self.assertFalse(path.with_suffix(".png").exists())

    def test_interrupted_sidecar_is_refused_for_new_dump(self):
        path, _ = self.write("001_in.raw", (2, 6, 3))
        image_io.raw_metadata_path(path).unlink()
        (self.folder / "dump.json").write_text(json.dumps(
            {"width": 4, "height": 3, "per_frame_geometry": True}))
        with self.assertRaisesRegex(ValueError, "missing per-frame geometry"):
            image_io.load_raw(path)
        self.assertEqual(self.convert(), 1)
        self.assertTrue(path.exists())

    def test_truncated_raw_is_preserved_and_returns_failure(self):
        path, _ = self.write("001_in.raw", (2, 6, 3))
        path.write_bytes(b"short")
        self.assertEqual(self.convert(), 1)
        self.assertEqual(path.read_bytes(), b"short")
        self.assertTrue(image_io.raw_metadata_path(path).exists())

    def test_encoder_failure_preserves_raw_and_existing_png(self):
        path, _ = self.write("001_in.raw", (2, 6, 3))
        png = path.with_suffix(".png")
        png.write_bytes(b"previous PNG")

        def fail(array, target):
            pathlib.Path(target).write_bytes(b"partial PNG")
            raise OSError("encoder interrupted")

        self.assertEqual(self.convert(save=fail), 1)
        self.assertTrue(path.exists())
        self.assertEqual(png.read_bytes(), b"previous PNG")
        self.assertEqual(sorted(item.name for item in self.folder.iterdir()),
                         ["001_in.png", "001_in.raw", "001_in.raw.json"])

    def test_wrong_png_shape_and_pixels_preserve_raw(self):
        path, image = self.write("001_in.raw", (2, 6, 3))
        for decoded in (image.reshape(3, 4, 3).astype(np.float32) / 255,
                        np.zeros_like(image, dtype=np.float32)):
            self.assertEqual(self.convert(load=lambda _: decoded), 1)
            self.assertTrue(path.exists())
            self.assertFalse(path.with_suffix(".png").exists())
            self.assertTrue(image_io.raw_metadata_path(path).exists())

    def test_keep_and_output_directory_leave_raw_with_geometry(self):
        path, image = self.write("001_in.raw", (2, 6, 3))
        self.assertEqual(self.convert("--keep"), 0)
        self.assertTrue(path.exists())
        other = self.folder / "converted"
        self.assertEqual(self.convert("--out", str(other)), 0)
        self.assertTrue(path.exists())
        self.assertPixels(image, load_codec(other / "001_in.png"))

    def test_metadata_write_failure_leaves_no_false_geometry_or_tempfiles(self):
        path, _ = self.write("001_in.raw", (2, 6, 3))
        (self.folder / "dump.json").write_text(json.dumps(
            {"width": 6, "height": 2, "per_frame_geometry": True}))
        replace = image_io.os.replace

        def fail_metadata(source, target):
            if pathlib.Path(target) == image_io.raw_metadata_path(path):
                raise OSError("sidecar replace interrupted")
            return replace(source, target)

        with mock.patch.object(image_io.os, "replace", side_effect=fail_metadata):
            with self.assertRaises(OSError):
                image_io.save_raw(np.zeros((3, 4, 3), np.uint8), path, metadata=True)
        self.assertEqual(sorted(item.name for item in self.folder.iterdir()),
                         ["001_in.raw", "001_in.raw.json.pending", "dump.json"])
        self.assertEqual(self.convert(), 1)
        self.assertTrue(path.exists())

    def test_failed_geometry_publish_refuses_legacy_fallback_at_equal_area(self):
        path, _ = self.write("001_in.raw", (2, 6, 3))
        # A mixed directory can still have legacy global geometry, with no new-dump marker.
        (self.folder / "dump.json").write_text(json.dumps({"width": 6, "height": 2}))
        replacement = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
        replace = image_io.os.replace

        def fail_metadata(source, target):
            if pathlib.Path(target) == image_io.raw_metadata_path(path):
                raise OSError("sidecar publication failed")
            return replace(source, target)

        with mock.patch.object(image_io.os, "replace", side_effect=fail_metadata):
            with self.assertRaises(OSError):
                image_io.save_raw(replacement, path, metadata=True)
        self.assertEqual(path.read_bytes(), replacement.tobytes())
        self.assertFalse(image_io.raw_metadata_path(path).exists())
        with self.assertRaisesRegex(ValueError, "geometry update is pending"):
            image_io.load_raw(path)
        with self.assertRaisesRegex(ValueError, "geometry update is pending"):
            comparisons._frame(self.folder, 1, "in")
        self.assertEqual(self.convert(), 1)
        self.assertTrue(path.exists())
        self.assertFalse(path.with_suffix(".png").exists())
        # A successful retry publishes the new geometry and removes the refusal marker.
        image_io.save_raw(replacement, path, metadata=True)
        self.assertFalse(image_io.raw_metadata_pending_path(path).exists())
        self.assertPixels(replacement, image_io.load_raw(path))
        self.assertEqual(self.convert(), 0)
        self.assertFalse(path.exists())
        self.assertPixels(replacement, load_codec(path.with_suffix(".png")))

    def test_invalid_geometry_is_refused(self):
        path, _ = self.write("001_in.raw", (2, 6, 3))
        sidecar = image_io.raw_metadata_path(path)
        for values in ({"width": True, "height": 2}, {"width": 6, "height": 0},
                       {"width": "6", "height": 2}, {"width": 6, "height": 2, "channels": 4},
                       {"width": 6, "height": 2, "dtype": "float16"}):
            sidecar.write_text(json.dumps(values))
            with self.assertRaises(ValueError):
                image_io.load_raw(path)


if __name__ == "__main__":
    unittest.main()
