#!/usr/bin/env python3
"""
nr_daemon — runs DLSS-NR for the Vulkan layer.

The layer is a shared object inside the game's process and the implementation is
Python, so the frame crosses a Unix socket rather than a function call. For a photo
mode that costs nothing: the game is meant to stall while the frame is being looked
at. A per-frame pass would need the graph ported to C.

    python3 src/layer/nr_daemon.py                        # then, to trigger:
    touch /tmp/nr_trigger        # hold the enhanced frame
    rm /tmp/nr_trigger           # release it

    ENABLE_NR_LAYER=1 VK_LAYER_PATH=... VK_INSTANCE_LAYERS=VK_LAYER_dlssnr_intel \\
    NR_LAYER_SOCKET=/tmp/nr_layer.sock NR_LAYER_TRIGGER=/tmp/nr_trigger <game>

Protocol: 16 bytes of header — magic 'NRN0', width, height, VkFormat — then
width*height*4 bytes of pixels; the same number of bytes come back.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import socket
import stat
import struct
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src" / "ref"))
sys.path.insert(0, str(ROOT / "src" / "gpu"))

import nr_frame  # noqa: E402

MAGIC = 0x304E524E
MASK_COVERAGE_LIMIT = 0.55   # above this the mask is the scene, not the interface
MAGIC_MASKED = 0x314E524E     # the same, with a one-byte-per-pixel interface mask after the colour

# The swapchain formats a compositor or VKD3D actually hands out. A game writes
# sRGB-encoded values into a UNORM swapchain just as it does into an SRGB one, so both
# decode the same way; what differs is only how the display reads them back.
FORMATS = {
    37: ("rgba8", "R8G8B8A8_UNORM"),
    43: ("rgba8", "R8G8B8A8_SRGB"),
    44: ("bgra8", "B8G8R8A8_UNORM"),
    50: ("bgra8", "B8G8R8A8_SRGB"),
    64: ("a2b10g10r10", "A2B10G10R10_UNORM_PACK32"),
}


def decode(raw, width, height, vk_format):
    kind, _ = FORMATS[vk_format]
    if kind == "a2b10g10r10":
        packed = np.frombuffer(raw, dtype=np.uint32).reshape(height, width)
        channels = [((packed >> shift) & 0x3FF).astype(np.float32) / np.float32(1023.0)
                    for shift in (0, 10, 20)]
        return np.stack(channels, axis=-1)
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
    # `pixels[..., [2, 1, 0]]` gathers into a new array, then `.astype` copies it again,
    # then the divide copies a third time — 170 ms for a 1080p frame. A reversed slice is
    # a *view*, and one ufunc does the widening and the scale together: 3 passes to 1.
    channels = pixels[..., 2::-1] if kind == "bgra8" else pixels[..., :3]
    # `divide`, not a multiply by 1/255: that reciprocal is not representable and the two
    # disagree in the last bit, which a test caught.
    return np.divide(channels, np.float32(255.0), dtype=np.float32)


def encode(image, raw, vk_format):
    """Write `image` back into a copy of `raw`, leaving alpha as the game left it."""
    kind, _ = FORMATS[vk_format]
    if kind == "a2b10g10r10":
        image = np.clip(image, 0.0, 1.0)
        packed = np.frombuffer(raw, dtype=np.uint32).copy().reshape(image.shape[:2])
        quantised = (image * np.float32(1023.0) + 0.5).astype(np.uint32)
        packed &= np.uint32(0xC0000000)
        for index, shift in enumerate((0, 10, 20)):
            packed |= np.minimum(quantised[..., index], 1023) << np.uint32(shift)
        return packed.tobytes()
    pixels = np.frombuffer(raw, dtype=np.uint8).copy().reshape(*image.shape[:2], 4)
    # The old form built an int32 intermediate — four bytes a channel for a value that
    # ends up in one — and then copied each channel separately, seven full-frame passes
    # for 424 ms at 1080p. In place, into a strided view, is three.
    quantised = np.multiply(image, np.float32(255.0), dtype=np.float32)
    np.add(quantised, np.float32(0.5), out=quantised)
    np.clip(quantised, 0.0, 255.0, out=quantised)
    target = pixels[..., 2::-1] if kind == "bgra8" else pixels[..., :3]
    target[:] = quantised.astype(np.uint8)
    return pixels.tobytes()


def receive(connection, count):
    chunks, got = [], 0
    while got < count:
        chunk = connection.recv(min(1 << 20, count - got))
        if not chunk:
            raise EOFError("the layer closed the connection")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


SOLID_RADIUS = 3


class Settings:
    """The knobs, re-read from a JSON file whenever it changes.

    The daemon holds the model, so restarting it to try another profile costs a second
    and loses the layer's connection. These are all post-network or extent choices —
    nothing here invalidates the weights — so they can move between frames. `nr-ctl`
    writes the file; anything may, it is one flat object.
    """

    KNOBS = ("profile", "intensity", "detail_strength", "colour_strength", "render_scale")

    def __init__(self, args):
        self.path = args.settings
        self.stamp = None
        for knob in self.KNOBS:
            setattr(self, knob, getattr(args, knob))

    def refresh(self):
        if not self.path:
            return
        try:
            stamp = os.stat(self.path).st_mtime_ns
        except OSError:
            return
        if stamp == self.stamp:
            return
        self.stamp = stamp
        try:
            with open(self.path) as handle:
                given = json.load(handle)
        except (OSError, ValueError) as error:
            print(f"settings: {error}; keeping the current ones", flush=True)
            return
        if not isinstance(given, dict):
            print("settings: expected a JSON object; keeping the current ones", flush=True)
            return
        changed = []
        for knob in self.KNOBS:
            if knob not in given:
                continue
            value = given[knob]
            if knob == "profile":
                if value not in nr_frame.PROFILES:
                    print(f"settings: no profile {value!r}", flush=True)
                    continue
            else:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    print(f"settings: {knob} is not a number", flush=True)
                    continue
                if not math.isfinite(value):
                    print(f"settings: {knob} must be finite", flush=True)
                    continue
                if knob == "render_scale":
                    if not 0.05 <= value <= 1.0:
                        print("settings: render_scale must be between 0.05 and 1", flush=True)
                        continue
                elif not 0.0 <= value <= 2.0:
                    # the vendor's own panel stops at 2 (notes/phase30-control-atlas.md)
                    print(f"settings: {knob} must be between 0 and 2", flush=True)
                    continue
            if getattr(self, knob) != value:
                setattr(self, knob, value)
                changed.append(f"{knob}={value}")
        if changed:
            print("settings: " + ", ".join(changed), flush=True)


def resample(image, size):
    """Bilinear resize of (H, W, C) to `size`, area-averaging when it divides evenly.

    Pure numpy because this runs per frame: ImageMagick through a subprocess, which is
    what `image_io.load` uses, costs more than the network does at these extents.

    Separable — one pass down the rows, then one across the columns. The obvious
    two-dimensional form, `image[y0][:, x0] * ... + image[y1][:, x1] * ...`, materialises
    four full-size gathers and measured **193 ms** on an 854x480 head against 20 ms for
    this, which made it 55 % of the whole frame. Splitting the axes leaves two gathers
    of the intermediate size instead.

    Downscaling by a whole factor takes the area mean instead of point-sampling, since
    the alternative feeds the network aliasing it would then try to enhance.
    """
    height, width = image.shape[:2]
    new_height, new_width = int(size[0]), int(size[1])
    if (new_height, new_width) == (height, width):
        return image
    if (new_height and new_width and height % new_height == 0 and width % new_width == 0
            and height > new_height and width > new_width):
        fy, fx = height // new_height, width // new_width
        return image.reshape(new_height, fy, new_width, fx, -1).mean((1, 3)).astype(np.float32)

    def axis(source, count, along):
        """One bilinear pass along `along` (0 rows, 1 columns)."""
        extent = source.shape[along]
        if extent == count:
            return source
        centres = (np.arange(count, dtype=np.float32) + 0.5) * (extent / count) - 0.5
        low = np.clip(np.floor(centres), 0, extent - 1).astype(np.int32)
        high = np.clip(low + 1, 0, extent - 1)
        weight = np.clip(centres - low, 0.0, 1.0).astype(np.float32)
        weight = weight.reshape((-1, 1, 1) if along == 0 else (1, -1, 1))
        return (np.take(source, low, along) * (1.0 - weight)
                + np.take(source, high, along) * weight)

    return axis(axis(np.asarray(image, np.float32), new_height, 0), new_width, 1)


def solid_regions(held, radius=SOLID_RADIUS, majority=0.75):
    """Keep only the parts of the interface mask that sit inside a solid held-still block.

    The layer marks single pixels, and that is right for an interface: a health bar is a
    slab of pixels that do not move. But a nearly static scene — a round transition, a
    slow replay — makes the *subject* half-hold too, and the mask comes back as a fine
    speckle over the character. Composing that interleaves original and re-rendered
    pixels, and since the pass moves skin by 20-30 levels the two populations differ by
    up to 100, which reads as mottled crust. Measured in `notes/phase43`.

    A majority filter separates them: an interface keeps ~85% of its pixels across a
    whole window and survives, speckle at ~60% or less does not. It is a majority rather
    than an erosion because a real interface does lose scattered pixels — an animated
    shine, bloom from the fighters — and an all-or-nothing test would erase the bar along
    with the noise.
    """
    r = int(radius)
    if r < 1:
        return held
    height, width = held.shape
    padded = np.zeros((height + 2 * r, width + 2 * r), np.int32)
    padded[r:r + height, r:r + width] = held
    integral = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    k = 2 * r + 1
    counted = (integral[k:k + height, k:k + width] - integral[0:height, k:k + width]
               - integral[k:k + height, 0:width] + integral[0:height, 0:width])
    return counted >= majority * k * k


def process_connection(connection, backend, args):
    """One request. Reject invalid extents before allocating/receiving the body.

    Closing a rejected exchange makes the Vulkan layer retain its original frame.
    """
    magic, width, height, vk_format = struct.unpack("<4I", receive(connection, 16))
    if magic not in (MAGIC, MAGIC_MASKED):
        raise ValueError(f"bad magic {magic:#x}")
    if not width or not height or width * height > args.max_pixels:
        raise ValueError(f"rejected extent {width}x{height}; limit {args.max_pixels} pixels")
    payload = receive(connection, width * height * 4)
    interface = receive(connection, width * height) if magic == MAGIC_MASKED else None
    if vk_format not in FORMATS:
        print(f"unsupported VkFormat {vk_format}; passing the frame through",
              flush=True)
        connection.sendall(payload)
        return

    live = args.live
    live.refresh()
    clock = time.perf_counter()
    colour = decode(payload, width, height, vk_format)
    # The network's cost follows the extent it is given and nothing else, so a smaller
    # internal frame is the only lever that changes the frame rate (notes/phase37,
    # phase45). What comes back up is the *head* — the detail the network drew — which
    # is then composed against the full-resolution original, so the game's own pixels
    # are never resampled and only the synthesised part is interpolated.
    inner = colour
    if live.render_scale < 1.0:
        inner = resample(colour, (max(64, round(height * live.render_scale)),
                                  max(64, round(width * live.render_scale))))
    geometry = nr_frame.NetworkGeometry.vendor_aligned(inner.shape[1], inner.shape[0])
    features = nr_frame.make_features(
        inner, geometry=geometry, **nr_frame.PROFILES[live.profile])
    head = geometry.crop(backend.run_features(features))
    if head.shape[:2] != colour.shape[:2]:
        # Only the first three channels reach `compose`; the fourth is the temporal gate
        # and neither game mode supplies history (notes/phase48). Carrying it through the
        # upscale is a quarter of that pass for nothing.
        head = resample(head[..., :3], colour.shape[:2])
    control = None
    held = None
    if interface is not None:
        # Red scales the blend per pixel, so an interface pixel comes back exactly as
        # the game drew it. The network still runs over the whole frame — masking the
        # composition rather than the input keeps the numerical path untouched.
        held = solid_regions(
            np.frombuffer(interface, np.uint8).reshape(height, width) > 127)
        # An interface is a small part of the frame. When the held region is most of it,
        # the scene itself is standing still — a round transition, a replay pause — and
        # what is being protected is the subject, not the HUD. Composing that leaves the
        # character in patches of two different exposures, which is far worse than a
        # softened health bar. Measured in `notes/phase43`: 43% held gave a clean frame,
        # 69% and 75% both blotched. So the mask is dropped rather than trusted.
        if held.mean() > MASK_COVERAGE_LIMIT:
            print(f"  interface mask covers {100 * held.mean():.0f}% of the frame; "
                  f"that is the scene holding still, not a HUD — mask dropped",
                  flush=True)
            held = None
    if held is not None:
        control = np.ones((height, width, 3), np.float32)
        control[held, 0] = 0.0
    output = nr_frame.compose(head, colour, intensity=live.intensity,
                              detail_strength=live.detail_strength,
                              colour_strength=live.colour_strength,
                              control_mask=control)
    encoded = encode(output, payload, vk_format)
    if held is not None:
        # The control mask reaches `compose_head`, but `compose_detail` runs *after* it
        # and re-weights the whole frame: with either strength away from 1 it moved 89.6%
        # of the masked pixels, so the interface protection silently stopped working the
        # moment a knob was touched. Restoring the original wire bytes here is exact by
        # construction — it survives every later stage and does not depend on the codec
        # round-tripping. Taken from the parallel ProjectsCodex tree, which had it.
        protected = np.frombuffer(encoded, np.uint8).copy().reshape(height, width, 4)
        original = np.frombuffer(payload, np.uint8).reshape(height, width, 4)
        protected[held] = original[held]
        encoded = protected.tobytes()
        output[held] = colour[held]          # so a --dump shows what was actually sent
    connection.sendall(encoded)
    if args.dump:
        import image_io
        try:
            destination = pathlib.Path(args.dump)
            destination.mkdir(parents=True, exist_ok=True)
            # Numbered, so walking through a game and pressing the trigger repeatedly
            # keeps every shot instead of overwriting the last one.
            index = 1 + max((int(path.stem.split("_")[0]) for path in destination.glob("*_in.png")
                             if path.stem.split("_")[0].isdigit()), default=0)
            image_io.save(colour, destination / f"{index:03d}_in.png")
            image_io.save(output, destination / f"{index:03d}_out.png")
            print(f"  -> {destination}/{index:03d}_{{in,out}}.png", flush=True)
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            print(f"frame returned, but dump failed: {error}", flush=True)
    note = "" if held is None else f"  interface {100 * held.mean():.0f}% left alone"
    print(f"{width}x{height} {FORMATS[vk_format][1]} in "
          f"{time.perf_counter() - clock:.2f}s  "
          f"change {np.abs(output - colour).mean():.5f}{note}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--socket", default="/tmp/nr_layer.sock")
    parser.add_argument("--profile", default="standard", choices=sorted(nr_frame.PROFILES))
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--detail-strength", type=float, default=1.0)
    parser.add_argument("--colour-strength", type=float, default=1.0)
    parser.add_argument("--settings", default=None,
                        help="a JSON file of knobs, re-read whenever it changes; "
                             "`nr-ctl` writes it")
    parser.add_argument("--render-scale", type=float, default=1.0,
                        help="run the network on this fraction of each side; the head is "
                             "scaled back and composed against the full-size frame")
    parser.add_argument("--max-pixels", type=int, default=1 << 22,
                        help="refuse frames larger than this, rather than thrash")
    parser.add_argument("--dump", help="write each frame in and out as PNG, for a look")
    parser.add_argument("--timeout", type=float, default=60,
                        help="socket inactivity timeout in seconds")
    args = parser.parse_args()
    if args.max_pixels <= 0 or args.timeout <= 0:
        parser.error("--max-pixels and --timeout must be positive")
    if not 0.05 <= args.render_scale <= 1.0:
        parser.error("--render-scale must be between 0.05 and 1")
    args.live = Settings(args)

    started = time.perf_counter()
    backend = nr_frame.ResidentBackend()
    print(f"model ready in {time.perf_counter() - started:.1f}s", flush=True)

    if os.path.lexists(args.socket):
        if not stat.S_ISSOCK(os.lstat(args.socket).st_mode):
            backend.close()
            raise SystemExit(f"socket path is occupied by a non-socket: {args.socket}")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            try:
                probe.connect(args.socket)
            except ConnectionRefusedError:
                os.unlink(args.socket)
            else:
                backend.close()
                raise SystemExit(f"a daemon is already listening at {args.socket}")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.socket)
    server.listen(4)
    print(f"listening on {args.socket}", flush=True)

    try:
        while True:
            connection, _ = server.accept()
            try:
                connection.settimeout(args.timeout)
                process_connection(connection, backend, args)
            except (EOFError, OSError, ValueError, RuntimeError) as error:
                print(f"frame rejected/failed; game keeps original: {error}", flush=True)
            finally:
                connection.close()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        backend.close()
        if os.path.exists(args.socket):
            os.unlink(args.socket)


if __name__ == "__main__":
    main()
