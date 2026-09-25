#!/usr/bin/env bash
# ============================================================================
#  deploy.sh  -  run DLSS-NR on Intel against a game (Linux / macOS)
#
#  What this does, and what it deliberately does not:
#
#    * It does NOT copy the model weights anywhere. The launcher points the daemon
#      at this checkout with NR_ROOT, so the weights stay in work/mlxw/ and nothing
#      large can be passed on by accident with the game folder.
#    * It DOES copy the runtime the daemon needs to start: work/mlx-dlss (the MLX
#      extractor the daemon imports), work/libxmx.so (the GPU runtime),
#      work/libnr_image.so and work/*.spv (the shaders). Without them the daemon
#      stops at start.
#    * It installs the layer next to a manifest whose library_path is its name.
#
#  Why the manifest matters: the Vulkan loader finds a layer through its manifest
#  plus VK_LAYER_PATH, not through the dynamic loader's search order, so the manifest
#  has to name the file and VK_LAYER_PATH has to point at the folder holding both.
#
#  Usage:
#    ./deploy.sh --game /path/to/Game [--name libnr_layer.so]
#                [--exe /path/to/Game/game] [--skip-build]
# ============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$REPO/work"
GAME=""
NAME="libnr_layer.so"
EXE=""
SKIP_BUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --game)       GAME="$2"; shift 2 ;;
    --name)       NAME="$2"; shift 2 ;;
    --exe)        EXE="$2";  shift 2 ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$GAME" ]] || { echo "ERROR: --game is required" >&2; exit 2; }
[[ -d "$GAME" ]] || { echo "ERROR: game directory not found: $GAME" >&2; exit 2; }

# ---- build ----
if [[ $SKIP_BUILD -eq 1 ]]; then
  echo "[skip] build (--skip-build)"
else
  echo "[1/2] building the layer ..."
  ( cd "$REPO" && make )
fi
[[ -f "$WORK/libnr_layer.so" ]] || { echo "ERROR: $WORK/libnr_layer.so missing; build it first" >&2; exit 3; }

# ---- install into the game folder ----
echo "[2/2] installing into $GAME ..."
DEPLOY="$GAME/dlss-nr"
mkdir -p "$DEPLOY/src" "$DEPLOY/work"

cp "$WORK/libnr_layer.so" "$DEPLOY/$NAME"
echo "  layer:      $DEPLOY/$NAME"

MANIFEST_SRC="$REPO/src/layer/VkLayer_dlss_nr.json"
MANIFEST_DST="$DEPLOY/VkLayer_dlss_nr.json"
[[ -f "$MANIFEST_SRC" ]] || { echo "ERROR: manifest template not found: $MANIFEST_SRC" >&2; exit 3; }
sed "s/LIBRARY_PATH_PLACEHOLDER/$NAME/g" "$MANIFEST_SRC" > "$MANIFEST_DST"
echo "  manifest:   $MANIFEST_DST  library_path = $NAME"

for d in layer ref gpu bench; do
  [[ -d "$REPO/src/$d" ]] && cp -r "$REPO/src/$d" "$DEPLOY/src/$d"
done

# The runtime the daemon imports. The weights are NOT copied.
if [[ -d "$WORK/mlx-dlss" ]]; then
  cp -r "$WORK/mlx-dlss" "$DEPLOY/work/mlx-dlss"
  echo "  runtime:    work/mlx-dlss"
else
  echo "  WARNING: $WORK/mlx-dlss is missing; the daemon needs it (see README)"
fi
for f in "$WORK"/libxmx.so* "$WORK"/libnr_image.so* "$WORK"/*.spv; do
  [[ -e "$f" ]] && cp "$f" "$DEPLOY/work/" || true
done
echo "  runtime:    libxmx, libnr_image, shaders"

find "$DEPLOY/src" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null

# ---- launcher ----
LAUNCH="$DEPLOY/launch-nr.sh"
{
  echo '#!/usr/bin/env bash'
  echo 'set -e'
  echo '# The loader finds the layer through VK_LAYER_PATH plus the manifest.'
  echo "export VK_LAYER_PATH=\"$DEPLOY\""
  echo 'export VK_INSTANCE_LAYERS=VK_LAYER_dlssnr_intel'
  echo 'export ENABLE_NR_LAYER=1'
  echo 'export NR_LAYER_SPAWN=1'
  echo 'export NR_LAYER_LIVE=1'
  echo 'export NR_LAYER_SOCKET="${NR_LAYER_SOCKET:-/tmp/nr_layer.sock}"'
  echo '# The weights stay in the checkout; only this path points at them.'
  echo "export NR_ROOT=\"$REPO\""
  echo ''
  if [[ -n "$EXE" ]]; then
    echo "exec \"$EXE\" \"\$@\""
  else
    echo 'GAME_EXE="${GAME_EXE:-}"'
    echo 'if [[ -z "$GAME_EXE" ]]; then echo "Set GAME_EXE in this file" >&2; exit 1; fi'
    echo 'exec "$GAME_EXE" "$@"'
  fi
} > "$LAUNCH"
chmod +x "$LAUNCH"
echo "  launcher:   $LAUNCH"

echo
echo "Done. Install folder: $DEPLOY"
echo "  $NAME, VkLayer_dlss_nr.json, src/, work/ (runtime only - no weights)"
echo
echo "Weights stay in $WORK/mlxw; the launcher points NR_ROOT at the checkout."
echo "Run the game through $LAUNCH so VK_LAYER_PATH is set."
echo
echo "COMPLIANCE: nothing NVIDIA's is shipped. The weights come from a DLL you own and"
echo "are never copied into the game folder."
