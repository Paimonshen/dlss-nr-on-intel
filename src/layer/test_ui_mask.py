#!/usr/bin/env python3
"""The interface mask, over the real socket: masked pixels must come back untouched.

The shipped feature never has to ask which pixels are interface — it inserts the pass
before the interface is drawn. A layer at `vkQueuePresentKHR` sees the composed frame,
so it marks the pixels that held still between two presents and the daemon leaves those
exactly as the game drew them. This checks that "exactly" means bit-identical, and that
an unmasked frame still goes through the old path.
"""
import pathlib, socket, struct, subprocess, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
MAGIC, MAGIC_MASKED = 0x304E524E, 0x314E524E
FORMAT_B8G8R8A8 = 44
SOCKET = "/tmp/nr_ui_mask_test.sock"
FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}", flush=True)
    if not ok:
        FAILURES.append(name)


def request(payload, header):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(180)
        client.connect(SOCKET)
        client.sendall(header + payload)
        want, chunks = len(payload) - (len(payload) - 4 * WIDTH * HEIGHT), b""
        want = 4 * WIDTH * HEIGHT
        while len(chunks) < want:
            piece = client.recv(want - len(chunks))
            if not piece:
                break
            chunks += piece
        return chunks


WIDTH, HEIGHT = 384, 256


def main():
    pathlib.Path(SOCKET).unlink(missing_ok=True)
    daemon = subprocess.Popen([sys.executable, str(ROOT / "src" / "layer" / "nr_daemon.py"),
                               "--socket", SOCKET], stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(300):
            if pathlib.Path(SOCKET).exists():
                break
            if daemon.poll() is not None:
                raise RuntimeError("daemon exited: " + (daemon.stdout.read() or ""))
            time.sleep(0.5)

        rng = np.random.default_rng(4)
        colour = rng.integers(0, 256, (HEIGHT, WIDTH, 4), dtype=np.uint8)
        colour[..., 3] = 255
        payload = colour.tobytes()

        # a band down the middle is "interface"
        mask = np.zeros((HEIGHT, WIDTH), np.uint8)
        mask[:, WIDTH // 3: 2 * WIDTH // 3] = 255
        header = struct.pack("<4I", MAGIC_MASKED, WIDTH, HEIGHT, FORMAT_B8G8R8A8)
        masked = np.frombuffer(request(payload + mask.tobytes(), header),
                               np.uint8).reshape(HEIGHT, WIDTH, 4)

        header = struct.pack("<4I", MAGIC, WIDTH, HEIGHT, FORMAT_B8G8R8A8)
        plain = np.frombuffer(request(payload, header), np.uint8).reshape(HEIGHT, WIDTH, 4)

        held = mask > 127
        check("a masked pixel comes back bit-identical",
              np.array_equal(masked[held], colour[held]),
              f"{held.sum()} pixels, max |d| {int(np.abs(masked[held].astype(int) - colour[held]).max())}")
        check("an unmasked pixel is still processed",
              not np.array_equal(masked[~held], colour[~held]),
              f"mean |d| {np.abs(masked[~held].astype(int) - colour[~held]).mean():.2f}")
        check("masking changes nothing outside the mask",
              np.array_equal(masked[~held], plain[~held]),
              "identical to the unmasked request")
        check("the unmasked path still works",
              not np.array_equal(plain, colour),
              f"mean |d| {np.abs(plain.astype(int) - colour).mean():.2f}")
    finally:
        daemon.terminate()
        daemon.wait(timeout=30)
        pathlib.Path(SOCKET).unlink(missing_ok=True)

    print("\n" + ("the interface mask is exact" if not FAILURES
                  else f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
