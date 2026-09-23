# build_release.ps1 -- assemble a near-complete Windows release (only weights missing).
#
# Mirrors scripts/build_release.sh. On Windows the daemon is pure Python (+ NumPy),
# and the Vulkan layer is compiled with MSVC into work/nr_layer.dll.
#
# DELIBERATELY ABSENT from dist/:
#   - work/mlxw/dlssnr-logical.safetensors   (the model weights)
#   - any nvngx_dlssnr.dll
# Run scripts/get_weights.py against YOUR OWN DLL to create the weights file; the
# release then runs unchanged.

$ErrorActionPreference = "Stop"
$ROOT = Resolve-Path (Join-Path $PSScriptRoot "..")
$DIST = Join-Path $ROOT "dist"
$MLX_PIN = "06a3e11a8b68817127406ace5c764463543f699b"

Write-Host "== building the Vulkan layer (MSVC) =="
$nrLayer = Join-Path $ROOT "src\layer\nr_layer.c"
$nrDef    = Join-Path $ROOT "src\layer\nr_layer.def"
New-Item -ItemType Directory -Force -Path (Join-Path $ROOT "work") | Out-Null
& cl.exe /O2 /LD "/I$ROOT\work\vulkan-headers\include" $nrLayer $nrDef /link vulkan-1.lib "/OUT:$(Join-Path $ROOT 'work\nr_layer.dll')" 2>&1 | Write-Host

Write-Host "== fetching Apache-2.0 MLX-DLSS numpy halves (pinned) =="
$MLX = Join-Path $ROOT "work\mlx-dlss"
if (-not (Test-Path (Join-Path $MLX "python\mlxdlss"))) {
  git clone https://github.com/iamwavecut/MLX-DLSS.git $MLX
  git -C $MLX checkout $MLX_PIN
}

Write-Host "== assembling dist/ =="
Remove-Item -Recurse -Force $DIST -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path (Join-Path $DIST "src")               | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $DIST "work\mlx-dlss\python") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $DIST "scripts")            | Out-Null

# built artifacts
Copy-Item -Recurse -Force (Join-Path $ROOT "work\*.dll") $DIST\work\  -ErrorAction SilentlyContinue
Copy-Item -Recurse -Force (Join-Path $ROOT "work\*.so")  $DIST\work\  -ErrorAction SilentlyContinue
Copy-Item -Recurse -Force (Join-Path $ROOT "work\*.spv") $DIST\work\  -ErrorAction SilentlyContinue

# clean inference source (no N-private extraction code: that lives under work/nvidia-private)
foreach ($d in @("layer","ref","gpu","bench")) {
  Copy-Item -Recurse -Force (Join-Path $ROOT "src\$d") (Join-Path $DIST "src\$d")
}
New-Item -ItemType Directory -Force -Path (Join-Path $DIST "src\tools") | Out-Null
Copy-Item -Force (Join-Path $ROOT "src\tools\publish_check.py") (Join-Path $DIST "src\tools\")
Copy-Item -Force (Join-Path $ROOT "src\tools\claims_check.py")  (Join-Path $DIST "src\tools\")

# Apache-2.0 third-party numpy halves the daemon imports by path
Copy-Item -Recurse -Force (Join-Path $MLX "python\mlxdlss") (Join-Path $DIST "work\mlx-dlss\python\mlxdlss")

# helpers + docs
Copy-Item -Force (Join-Path $ROOT "scripts\get_weights.py") (Join-Path $DIST "scripts\")
Copy-Item -Force (Join-Path $ROOT "GET-STARTED.md") $DIST\
Copy-Item -Force (Join-Path $ROOT "README.md")      $DIST\
Copy-Item -Force (Join-Path $ROOT "LICENSE")        $DIST\

$releaseNote = @"
DLSS-NR-on-Intel release build (Windows).

This package is complete EXCEPT for the model weights, which are NVIDIA's property
and are not redistributed. To make it run:

  1. Obtain your own copy of nvngx_dlssnr.dll (version 310.8.0.0).
  2. python scripts\get_weights.py \path\to\nvngx_dlssnr.dll
     -> writes work\mlxw\dlssnr-logical.safetensors (649 logical tensors)
  3. python src\ref\nr_frame.py IN.png OUT.png --resident   # sanity check, no game

Then start the daemon (src\layer\nr_daemon.py) and point your client at its pipe.
See GET-STARTED.md for the full walkthrough.
"@
Set-Content -Path (Join-Path $DIST "README-release.txt") -Value $releaseNote

$weightsPresent = Test-Path (Join-Path $DIST "work\mlxw\dlssnr-logical.safetensors")
Write-Host "== release assembled at $DIST =="
Write-Host "   weights present? $(if ($weightsPresent) {'yes'} else {'NO -- expected NO'})"
Write-Host "   run:  python scripts\get_weights.py <your dll>   to complete it"
