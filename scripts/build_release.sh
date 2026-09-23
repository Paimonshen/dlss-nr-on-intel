#!/usr/bin/env bash
# build_release.sh — assemble a near-complete release that is only missing the weights.
#
# What this produces (under dist/):
#   - the resident Vulkan runtime   (work/libxmx.so)
#   - the host-pass library          (work/libnr_image.so)
#   - compute shaders                (work/*.spv)
#   - the daemon + layer + tools      (src/layer/*, src/ref/*, src/gpu/*, src/bench/*)
#   - the Apache-2.0 MLX-DLSS numpy halves the daemon loads by path
#     (python/mlxdlss: features.py / composition.py / temporal.py / motion_quality.py …)
#
# What is DELIBERATELY ABSENT:
#   - work/mlxw/dlssnr-logical.safetensors  (the model weights)
#   - any nvngx_dlssnr.dll
# These are NVIDIA's and are never shipped. Run scripts/get_weights.py against YOUR
# OWN DLL to create work/mlxw/dlssnr-logical.safetensors; the release then runs as-is.
#
# The result is a "release-grade" tree: built artifacts, no NVIDIA code or weights.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="$ROOT/dist"
MLX_PIN="06a3e11a8b68817127406ace5c764463543f699b"

echo "== building resident runtime, host passes and shaders =="
( cd "$ROOT" && make )

echo "== fetching Apache-2.0 MLX-DLSS numpy halves (pinned) =="
MLX="$ROOT/work/mlx-dlss"
if [ ! -d "$MLX/python/mlxdlss" ]; then
  git clone https://github.com/iamwavecut/MLX-DLSS.git "$MLX"
  git -C "$MLX" checkout "$MLX_PIN"
fi

echo "== assembling dist/ =="
rm -rf "$DIST"
mkdir -p "$DIST/src" "$DIST/work/mlx-dlss" "$DIST/scripts"

# built artifacts
cp -r "$ROOT/work"/*.so "$ROOT/work"/*.spv "$DIST/work/" 2>/dev/null || true
# The layer library goes at the deploy root: it auto-spawns the daemon and searches
# src/layer/nr_daemon.py relative to itself. The other .so/.spv are runtime deps and
# stay under work/.
cp "$ROOT/work/libnr_layer.so" "$DIST/" 2>/dev/null || true

# source (the clean, weights-free inference code only)
cp -r "$ROOT/src/layer" "$ROOT/src/ref" "$ROOT/src/gpu" "$ROOT/src/bench" "$DIST/src/"
cp "$ROOT/src/tools/publish_check.py" "$ROOT/src/tools/claims_check.py" "$DIST/src/tools/" 2>/dev/null || \
  mkdir -p "$DIST/src/tools" && cp "$ROOT/src/tools/publish_check.py" "$ROOT/src/tools/claims_check.py" "$DIST/src/tools/"

# Apache-2.0 third-party numpy halves the daemon imports by path
cp -r "$MLX/python/mlxdlss" "$DIST/work/mlx-dlss/python/mlxdlss"

# helpers
cp "$ROOT/scripts/get_weights.py" "$DIST/scripts/"
cp "$ROOT/GET-STARTED.md" "$ROOT/README.md" "$ROOT/LICENSE" "$DIST/"

# a release note that states, in the package, what the user must still add
cat > "$DIST/README-release.txt" <<'EOF'
DLSS-NR-on-Intel release build.

This package is complete EXCEPT for the model weights, which are NVIDIA's property
and are not redistributed. The Vulkan layer (nr_layer.so at the root) auto-spawns
the daemon (src/layer/nr_daemon.py) on load, so you do NOT start the daemon by hand.

To make it run:

  1. Obtain your own copy of nvngx_dlssnr.dll (version 310.8.0.0).
  2. python3 scripts/get_weights.py /path/to/nvngx_dlssnr.dll
     -> writes work/mlxw/dlssnr-logical.safetensors (649 logical tensors)
  3. python3 src/ref/nr_frame.py IN.png OUT.png --resident   # sanity check, no game

Then launch the game with the layer enabled (VK_LAYER_PATH / the proxy DLL).
See GET-STARTED.md for the full Linux / Windows walkthrough.
EOF

echo "== release assembled at $DIST =="
echo "   weights present? $( [ -f "$DIST/work/mlxw/dlssnr-logical.safetensors" ] && echo yes || echo NO -- expected NO )"
echo "   run:  python3 scripts/get_weights.py <your dll>   to complete it"
