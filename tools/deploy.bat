@echo off
setlocal
rem ===========================================================================
rem  deploy.bat  -  run DLSS-NR on Intel against a game (Windows)
rem
rem  What this does, and what it deliberately does not:
rem
rem    * It does NOT copy the model weights anywhere. The launcher points the daemon
rem      at this checkout with NR_ROOT, so the weights stay in work/mlxw/ where
rem      scripts/get_weights.py put them, and nothing large can be passed on by
rem      accident with the game folder.
rem    * It DOES copy the runtime the daemon needs to start: work/mlx-dlss (the MLX
rem      extractor the daemon imports), work/libxmx.* (the GPU runtime), work/*.spv
rem      (the shaders). Without them the daemon stops at start.
rem    * It installs the layer under a name you choose (default nr_layer.dll) next to
rem      a manifest whose library_path is that name.
rem
rem  Why the manifest matters: the Vulkan loader finds a layer through its manifest
rem  plus VK_LAYER_PATH, not through the DLL search order, so the manifest has to name
rem  the file and VK_LAYER_PATH has to point at the folder holding both. That is also
rem  why the default name is NOT version.dll: on Windows a module called version.dll,
rem  once loaded, is what later imports of version.dll resolve to, so it can silently
rem  stand in for the system library. The manifest makes the name irrelevant anyway.
rem
rem  Usage:
rem    deploy.bat --game "C:\Games\MyGame" [--name nr_layer.dll]
rem              [--exe "C:\Games\MyGame\game.exe"] [--skip-build]
rem ===========================================================================

set "GAME="
set "NAME=nr_layer.dll"
set "EXE="
set "SKIP_BUILD=0"
rem Capture the script directory BEFORE any shift: shift moves %0 and %~dp0 with it.
set "TOOLSDIR=%~dp0"

:parse
if "%~1"=="" goto :done_parse
if /i "%~1"=="--game"        ( set "GAME=%~2" & shift & shift & goto :parse )
if /i "%~1"=="--name"        ( set "NAME=%~2" & shift & shift & goto :parse )
if /i "%~1"=="--exe"         ( set "EXE=%~2"  & shift & shift & goto :parse )
if /i "%~1"=="--skip-build"  ( set "SKIP_BUILD=1" & shift & goto :parse )
echo Unknown argument: %~1
exit /b 2
:done_parse

set "REPO=%TOOLSDIR%.."
rem Resolve the .. so NR_ROOT and every path below carry no relative segment
pushd "%REPO%" >nul 2>&1
set "REPO=%CD%"
popd
set "WORK=%REPO%\work"

if "%GAME%"=="" (
  echo ERROR: --game is required, the game directory to install into
  exit /b 2
)
if not exist "%GAME%" (
  echo ERROR: game directory not found: %GAME%
  exit /b 2
)

rem ---- build: delegated to build_win.bat, which also builds libxmx.dll and the shaders ----
rem Kept in one place on purpose: deploy only has to know that the runtime exists, and
rem a second copy of the MSVC invocation here is a second thing to get wrong.
if "%SKIP_BUILD%"=="1" goto :after_build
echo [1/2] building with build_win.bat ...
if not exist "%TOOLSDIR%build_win.bat" (
  echo ERROR: tools\build_win.bat not found
  exit /b 3
)
call "%TOOLSDIR%build_win.bat"
if errorlevel 1 (
  echo ERROR: the build failed
  exit /b 3
)
:after_build
if not exist "%WORK%\nr_layer.dll" (
  echo ERROR: %WORK%\nr_layer.dll is missing; build it first or drop --skip-build
  exit /b 3
)
if not exist "%WORK%\libxmx.dll" (
  echo WARNING: %WORK%\libxmx.dll is missing - the daemon cannot start without it
)

rem ---- install into the game folder ----
echo [2/2] installing into %GAME% ...
set "DEPLOY=%GAME%\dlss-nr"
if not exist "%DEPLOY%" mkdir "%DEPLOY%"

copy /Y "%WORK%\nr_layer.dll" "%DEPLOY%\%NAME%" >nul || exit /b 3
echo   layer:      %DEPLOY%\%NAME%

set "MANIFEST_SRC=%REPO%\src\layer\VkLayer_dlss_nr.json"
set "MANIFEST_DST=%DEPLOY%\VkLayer_dlss_nr.json"
if not exist "%MANIFEST_SRC%" (
  echo ERROR: manifest template not found: %MANIFEST_SRC%
  exit /b 3
)
rem An ABSOLUTE library_path. A bare name makes the loader fail with error 87
rem (ERROR_INVALID_PARAMETER) when the layer is found through VK_LAYER_PATH: measured
rem here, the same manifest loads with the full path and not without it.
powershell -NoProfile -Command "$lib = (Join-Path '%DEPLOY%' '%NAME%'); (Get-Content '%MANIFEST_SRC%') -replace 'LIBRARY_PATH_PLACEHOLDER', $lib.Replace('\','\\') | Set-Content '%MANIFEST_DST%'"
echo   manifest:   %MANIFEST_DST%  library_path = %DEPLOY%\%NAME%

rem The runtime the daemon needs at start: the MLX extractor it imports, the resident
rem Vulkan runtime, and the shaders. Without these the daemon stops before it listens.
rem The weights are NOT copied - the launcher points NR_ROOT at this checkout instead,
rem so nothing large can travel with the game folder.
if not exist "%DEPLOY%\src" mkdir "%DEPLOY%\src"
if not exist "%DEPLOY%\work" mkdir "%DEPLOY%\work"
xcopy /E /I /Y /Q "%REPO%\src\layer" "%DEPLOY%\src\layer" >nul
xcopy /E /I /Y /Q "%REPO%\src\ref"   "%DEPLOY%\src\ref"   >nul
xcopy /E /I /Y /Q "%REPO%\src\gpu"   "%DEPLOY%\src\gpu"   >nul
xcopy /E /I /Y /Q "%REPO%\src\bench" "%DEPLOY%\src\bench" >nul

if exist "%WORK%\mlx-dlss" (
  xcopy /E /I /Y /Q "%WORK%\mlx-dlss" "%DEPLOY%\work\mlx-dlss" >nul
  echo   runtime:    work\mlx-dlss
) else (
  echo   WARNING: %WORK%\mlx-dlss is missing; the daemon cannot start without it
)
if exist "%WORK%\libxmx.dll"    copy /Y "%WORK%\libxmx.dll"    "%DEPLOY%\work\" >nul
if exist "%WORK%\libnr_image.dll" copy /Y "%WORK%\libnr_image.dll" "%DEPLOY%\work\" >nul
copy /Y "%WORK%\*.spv" "%DEPLOY%\work\" >nul 2>&1
echo   runtime:    libxmx.dll, libnr_image.dll, shaders

rem Keep local junk out of the deployed tree
for /d /r "%DEPLOY%\src" %%J in (__pycache__) do @if exist "%%J" rmdir /S /Q "%%J" 2>nul
del /S /Q "%DEPLOY%\src\*.obj" >nul 2>&1
del /S /Q "%DEPLOY%\src\*.lib" >nul 2>&1
del /S /Q "%DEPLOY%\src\*.exp" >nul 2>&1

rem Drop local junk the copy may have brought along
for /d /r "%DEPLOY%\src" %%J in (__pycache__) do @if exist "%%J" rmdir /S /Q "%%J" 2>nul
del /S /Q "%DEPLOY%\src\*.obj" >nul 2>&1
del /S /Q "%DEPLOY%\src\*.lib" >nul 2>&1
del /S /Q "%DEPLOY%\src\*.exp" >nul 2>&1

rem ---- launcher ----
set "LAUNCH=%DEPLOY%\launch-nr.bat"
(
  echo @echo off
  echo setlocal
  echo.
  echo rem Launcher for DLSS-NR on Intel.
  echo rem The loader finds the layer through VK_LAYER_PATH plus the manifest, so the
  echo rem library name above does not matter; the manifest names it.
  echo set "VK_LAYER_PATH=%DEPLOY%"
  echo set "VK_INSTANCE_LAYERS=VK_LAYER_dlssnr_intel"
  echo set "ENABLE_NR_LAYER=1"
  echo set "NR_LAYER_SPAWN=1"
  echo set "NR_LAYER_LIVE=1"
  echo rem Windows talks to the daemon over a named pipe, not a Unix socket.
  echo set "NR_LAYER_SOCKET=\\.\pipe\nr_dlssnr_intel"
  echo rem The weights stay in the checkout; only this path points at them.
  echo set "NR_ROOT=%REPO%"
  echo.
) > "%LAUNCH%"
if defined EXE (
  echo start "" "%EXE%" %%* >> "%LAUNCH%"
) else (
  echo set "GAME_EXE=GAME_EXE_NOT_SET" >> "%LAUNCH%"
  echo if "%%GAME_EXE%%"=="GAME_EXE_NOT_SET" ^( echo Set GAME_EXE in this file ^& exit /b 1 ^) >> "%LAUNCH%"
  echo start "" "%%GAME_EXE%%" %%* >> "%LAUNCH%"
)
echo   launcher:   %LAUNCH%

echo.
echo Done. Install folder: %DEPLOY%
echo   %NAME%, VkLayer_dlss_nr.json, src\, work\ (runtime only - no weights)
echo.
echo Weights stay in %WORK%\mlxw; the launcher points NR_ROOT at the checkout.
echo Run the game through %LAUNCH% so VK_LAYER_PATH is set.
echo.
echo COMPLIANCE: nothing NVIDIA's is shipped. The weights come from a DLL you own and
echo are never copied into the game folder.
endlocal
exit /b 0
