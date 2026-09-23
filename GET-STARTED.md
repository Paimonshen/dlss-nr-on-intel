# Getting started — DLSS 5 Neural Rendering on Intel

This is the **code-only** port of NVIDIA's DLSS 5 Neural Rendering model onto an
Intel XMX (Xe2 / Battlemage) GPU. It runs the recovered 71-block graph on Intel's
cooperative-matrix units, either:

- **Linux** — injected into any Vulkan-presenting game through a Vulkan layer
  (`vkQueuePresentKHR`), speaking to the daemon over a Unix socket.
- **Windows** — the same daemon, speaking to a client over a **named pipe**
  (the `nr_transport.h` / `nr_pipe.py` transport), so a Windows host that drives
  the pass can reach it.

> **You bring the weights.** This tree contains no NVIDIA binaries and no weights
> derived from them. `nvngx_dlssnr.dll` is NVIDIA's and is **not redistributed
> here** — every project in this space requires you to supply your own copy, and
> so does this one. Nothing runs until you extract the logical weight file from a
> DLL you already have. See [Weights](#weights-you-supply-these) below.

> **The tracked source is NVIDIA-private-free.** Everything that parses or recovers
> internals of `nvngx_dlssnr.dll` — the PE/`.rsrc`/`WEIGHTS_HT`/fatbin/PTX readers and
> the superseded graph-recovery research code — has been moved **out of the tracked
> tree** into `work/nvidia-private/` (git-ignored, never published). What remains in
> `src/` is the inference path that only *consumes* the user-supplied
> `dlssnr-logical.safetensors` (an Apache-2.0 MLX-DLSS layout), plus the publishing
> guard tools. A release built here is therefore complete except for the weights;
> [Build a release](#build-a-release) + [Weights](#weights-you-supply-these) below.

---

## 0. Prerequisites

Both platforms need:

- A GPU that exposes `VK_KHR_cooperative_matrix` with an `fp16 × fp16 → fp32`
  configuration. Developed on **Arc 140V / Xe2 (Mesa ANV)**; an **Arc B580**
  (discrete Battlemage) reports the same six configurations.
- Python 3 with **NumPy** and **safetensors**.
- A C compiler, `glslangValidator`, and the Vulkan loader/headers.
- About **2.3 GiB** of memory for the device buffers at 720p (iGPU shares it with
  system RAM).

Verify the matrix unit before anything else — the probe needs neither weights nor
the rest of the build:

```sh
# Linux (headers may already be on your system; drop the -I if so)
gcc -Iwork/vulkan-headers/include src/probe/coopmat_probe.c -o /tmp/probe -lvulkan
/tmp/probe
```

On Windows the equivalent is a Vulkan-capable Intel driver (Intel Graphics
Software) plus the Vulkan SDK; the same cooperative-matrix capability is what the
daemon checks for at load.

---

## 1. Build

```sh
mkdir -p work
git clone --depth 1 --branch v1.4.321 \
          https://github.com/KhronosGroup/Vulkan-Headers.git work/vulkan-headers
git clone https://github.com/iamwavecut/MLX-DLSS.git work/mlx-dlss
git -C work/mlx-dlss checkout 06a3e11a8b68817127406ace5c764463543f699b
make
```

This produces the resident runtime (`work/libxmx.so`), the host-pass library
(`work/libnr_image.so`), the shaders (`work/*.spv`), and — on Linux — the game
layer (`work/libnr_layer.so`). See `Makefile` for the full target list.

### Windows builds

The layer and the daemon transport are portable to Windows; the game layer is
built with MSVC (the `.def` exports the three symbols the Vulkan loader needs):

```bat
:: the daemon side: no C build needed, it is pure Python (+ NumPy)
:: the layer (a Vulkan layer DLL) on Windows:
cl /O2 /LD /Iwork/vulkan-headers/include src\layer\nr_layer.c nr_layer.def /link vulkan-1.lib /OUT:work\nr_layer.dll
```

The daemon itself is launched the same way on either OS:

```sh
python3 src/layer/nr_daemon.py --settings work/nr_settings.json --socket <endpoint>
```

- Linux endpoint: a filesystem path, e.g. `/tmp/nr_layer.sock`.
- Windows endpoint: a named-pipe name, e.g. `\\.\pipe\nr_dlssnr_intel`.

---

## 1b. Build a release

`make` (or the MSVC line above) builds the pieces into `work/`. To assemble a
**distributable, near-complete release** that is only missing the weights, use the
release script:

```sh
# Linux / WSL
scripts/build_release.sh
# Windows (PowerShell, with MSVC + Git + Vulkan SDK on PATH)
scripts/build_release.ps1
```

It collects the compiled runtime, the daemon/layer/inference source, and the
Apache-2.0 MLX-DLSS numpy halves (cloned and pinned) into `dist/`. It
**deliberately omits** `work/mlxw/dlssnr-logical.safetensors` and any
`nvngx_dlssnr.dll` — those are NVIDIA's and never shipped. `dist/README-release.txt`
states exactly what the user must still add. After the weights step below, the
release runs unchanged.

---

## 2. Weights (you supply these)

You need your **own** `nvngx_dlssnr.dll` (version 310.8.0.0). It is not provided
here and must not be committed (see `.gitignore`: `/ref/`, `/work/`, `*.dll`,
`*.safetensors`, … are all ignored).

Once you have the DLL, extract the **logical** weight file. The helper script
`scripts/get_weights.py` does exactly the README's two-step extraction for you:

```sh
# point it at your DLL; it writes work/mlxw/dlssnr-logical.safetensors
python3 scripts/get_weights.py /path/to/nvngx_dlssnr.dll

# or, the manual equivalent:
mkdir -p work/mlxw
python3 work/mlx-dlss/python/mlxdlss/tools/extract_dlssnr_weights.py \
        /path/to/nvngx_dlssnr.dll work/mlxw/dlssnr-packed.safetensors
python3 work/mlx-dlss/python/mlxdlss/tools/unpack_dlssnr_weights.py \
        work/mlxw/dlssnr-packed.safetensors work/mlxw/dlssnr-logical.safetensors
```

The result is **649 named tensors, 145 755 123 parameters**. The reader refuses
anything but `fully_logical=true` — the packed file is *not* a substitute, and
reading it as dense FP16 gives values correlating −0.02 with the truth.

After extraction the daemon starts cleanly:

```sh
python3 src/ref/nr_frame.py IN.png OUT.png --resident   # one still, no game
```

If you forget this step, the daemon exits with a clear message instead of a
traceback:

```
no weights at work/mlxw/dlssnr-logical.safetensors.
They are NVIDIA's and are not distributed here: extract them from your own
copy of nvngx_dlssnr.dll as the README's Build section describes.
```

---

## 3. Run

### Linux — in a Vulkan game (photo mode or live)

```sh
python3 src/layer/nr_daemon.py --settings /tmp/nr_settings.json
src/layer/nr-photo --steam <appid>   # prints the VK_LAYER_PATH launch option
```

The printed launch option carries **your** clone's absolute path; paste what it
prints, never a copied one. Leave out `NR_LAYER_LIVE` for photo mode, set
`NR_LAYER_LIVE=1` for live (a slideshow: every Nth frame re-rendered).

### Windows — named-pipe client

Start the daemon listening on a pipe, then point your client (whatever drives the
pass on the Windows side) at the same pipe name:

```bat
python src\layer\nr_daemon.py --settings work\nr_settings.json --socket \\.\pipe\nr_dlssnr_intel
```

The bytes on the wire are identical on both platforms (16-byte header, then
colour, then optional guides), so a Linux-built client and a Windows daemon — or
vice versa — interoperate. The `NRN2` guide frame (engine motion + depth) lets the
daemon reproject history instead of the identity fallback.

---

## 4. Controls

Every knob is post-network or an extent choice; none invalidates the weights, and
all move between frames. `src/layer/nr-ctl` / `nr-panel` / `nr-toggle` drive them
live. `nr-ctl report` prints everything a bug report needs, including the frames
the daemon **refused** (a refused frame is invisible while you play: the game
simply shows its own picture).

---

## What is deliberately NOT here

- `nvngx_dlssnr.dll` and any weights — **yours to supply** (see §2).
- Anything under `ref/` or `work/` — build output and the immutable original,
  both git-ignored.
- Build artifacts: `*.so`, `*.spv`, `*.obj`, `*.dll` (post-build), archives.

The full reference, findings, and the traps to avoid live in `notes/HANDOFF.md`
(it overrides `notes/CLAUDE.md`). For the architecture and the recovery story,
read `README.md` and `docs/ARCHITECTURE.md`.
