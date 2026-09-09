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
import os
import pathlib
import socket
import stat
import struct
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "src" / "ref"))
sys.path.insert(0, str(ROOT / "src" / "gpu"))

import nr_frame  # noqa: E402

MAGIC = 0x304E524E

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
    order = [2, 1, 0] if kind == "bgra8" else [0, 1, 2]
    return pixels[..., order].astype(np.float32) / np.float32(255.0)


def encode(image, raw, vk_format):
    """Write `image` back into a copy of `raw`, leaving alpha as the game left it."""
    kind, _ = FORMATS[vk_format]
    image = np.clip(image, 0.0, 1.0)
    if kind == "a2b10g10r10":
        packed = np.frombuffer(raw, dtype=np.uint32).copy().reshape(image.shape[:2])
        quantised = (image * np.float32(1023.0) + 0.5).astype(np.uint32)
        packed &= np.uint32(0xC0000000)
        for index, shift in enumerate((0, 10, 20)):
            packed |= np.minimum(quantised[..., index], 1023) << np.uint32(shift)
        return packed.tobytes()
    pixels = np.frombuffer(raw, dtype=np.uint8).copy().reshape(*image.shape[:2], 4)
    order = [2, 1, 0] if kind == "bgra8" else [0, 1, 2]
    quantised = np.minimum((image * np.float32(255.0) + 0.5).astype(np.int32), 255)
    for index, channel in enumerate(order):
        pixels[..., channel] = quantised[..., index].astype(np.uint8)
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


def process_connection(connection, backend, args):
    """One request. Reject invalid extents before allocating/receiving the body.

    Closing a rejected exchange makes the Vulkan layer retain its original frame.
    """
    magic, width, height, vk_format = struct.unpack("<4I", receive(connection, 16))
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic:#x}")
    if not width or not height or width * height > args.max_pixels:
        raise ValueError(f"rejected extent {width}x{height}; limit {args.max_pixels} pixels")
    payload = receive(connection, width * height * 4)
    if vk_format not in FORMATS:
        print(f"unsupported VkFormat {vk_format}; passing the frame through",
              flush=True)
        connection.sendall(payload)
        return

    clock = time.perf_counter()
    colour = decode(payload, width, height, vk_format)
    geometry = nr_frame.NetworkGeometry.vendor_aligned(width, height)
    features = nr_frame.make_features(
        colour, geometry=geometry, **nr_frame.PROFILES[args.profile])
    head = geometry.crop(backend.run_features(features))
    output = nr_frame.compose(head, colour, intensity=args.intensity,
                              detail_strength=args.detail_strength,
                              colour_strength=args.colour_strength)
    connection.sendall(encode(output, payload, vk_format))
    if args.dump:
        import image_io
        try:
            destination = pathlib.Path(args.dump)
            destination.mkdir(parents=True, exist_ok=True)
            image_io.save(colour, destination / "in.png")
            image_io.save(output, destination / "out.png")
        except (OSError, image_io.subprocess.CalledProcessError) as error:
            print(f"frame returned, but dump failed: {error}", flush=True)
    print(f"{width}x{height} {FORMATS[vk_format][1]} in "
          f"{time.perf_counter() - clock:.2f}s  "
          f"change {np.abs(output - colour).mean():.5f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--socket", default="/tmp/nr_layer.sock")
    parser.add_argument("--profile", default="standard", choices=sorted(nr_frame.PROFILES))
    parser.add_argument("--intensity", type=float, default=1.0)
    parser.add_argument("--detail-strength", type=float, default=1.0)
    parser.add_argument("--colour-strength", type=float, default=1.0)
    parser.add_argument("--max-pixels", type=int, default=1 << 22,
                        help="refuse frames larger than this, rather than thrash")
    parser.add_argument("--dump", help="write each frame in and out as PNG, for a look")
    parser.add_argument("--timeout", type=float, default=60,
                        help="socket inactivity timeout in seconds")
    args = parser.parse_args()
    if args.max_pixels <= 0 or args.timeout <= 0:
        parser.error("--max-pixels and --timeout must be positive")

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
