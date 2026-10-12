## The first Windows releases, and a tester's game that got the proxy but never the layer (2026-10-09, later)

**Released at the owner's word**, under versions the owner named; neither carries the NVIDIA DLL
or weights:
- **v0.0.1**, tagged at `7f9c364`: `dlss-nr-windows-0.0.1-x64.zip`, 19 445 355 bytes, SHA-256
  `b9f5af1b…6437`.
- **v0.0.1.1**, tagged at `f9ec5f1`, the latest: `dlss-nr-windows-0.0.1.1-x64.zip`,
  19 448 154 bytes, SHA-256 `bbcee183…fda2`. Against 0.0.1 only `release-metadata.json`,
  `scripts/windows_wizard_core.py` and `windows-wizard.ps1` differ; the DLLs are the same. From
  the unpacked ZIP: the window's self-test, Check (the DLL as 310.8.0.0), Install NR on Dead or
  Alive 5 with the weights extracted, Save report (19 checks, the DLL tested, the extractor's
  log) and Remove NR, the folder as before. The game was not started.

`master` is `windows` again, at `f9ec5f1`. `2cba658` (the README: Windows beside Linux, and what
differs between them) went through PR #11; the rest went straight to `master`, which the owner
prefers to a pull request to oneself. `137fa78` is the issue form, `.github/ISSUE_TEMPLATE/problem.yml`:
platform, release, step, the window's words and its report.

**`a243ef8`, from the first tester's report.** An Arc B580 (issue #12) failed to install 0.0.1
twice, and its report said neither which check failed nor what the weight extractor printed. Now:
- the window names each failed check and why;
- Save report carries the checks, the DLL's version and SHA-256 and whether it is the tested
  310.8.0.0, and the tail of `work/logs/get-weights.log`, where install now writes the extractor's
  output;
- the Python and DLL probes wait 30 s, not 4, which a fresh environment's first NumPy import can
  need;
- a `VCOMP140.DLL` that does not load is named, with the Visual C++ Redistributable's link.

With 0.0.1.1 the B580 installed: its DLL is 310.8.0.0, 649 tensors, every check passed. Which
change did it is not known.

**Then no effect, in Need for Speed Heat (DX11 through DXVK) or in DOOM (2016).** The report from
the first: the proxy's launch record for `NeedForSpeedHeat.exe`, still running; the trigger on;
and **no `nr_daemon.log` at all**. The layer opens that log and spawns the daemon in
`nr_CreateInstance`. So the proxy loaded and set the variables, and the layer never made an
instance in the game. The window cannot tell: the launch record is there, so Enable NR's warning
stays silent.

**The likely reason: Windows' Vulkan loader ignores `VK_LAYER_PATH` in an elevated process.** In
Vulkan-Loader's source (`main`, read on 10-09):
- `VK_LAYER_PATH`, `VK_ADD_LAYER_PATH` and the implicit-layer and driver variables go through
  `loader_secure_getenv`, which on Windows returns nothing in a process at High integrity or above
  (`is_high_integrity`);
- HKCU's layer keys are skipped then too (`use_secondary_hive`), and only HKLM is read;
- `VK_INSTANCE_LAYERS` is plain `loader_getenv`, so the layer is still asked for. Its absence is
  one error line in the loader's debug output, and the instance is made without it.

So a game run as administrator gets the proxy and never the layer. So does every game when UAC is
off, and every game started by a launcher that runs as administrator. The tester's account is
named "Admin", on Windows 11 Pro build 28000. Asked in the issue at 11:40 UTC, with Task
Manager's Elevated column to check. On Linux `secure_getenv` refuses only setuid and setgid
processes, and games are neither.

**Confirmed here, with vkcube through the proxy** (`nr-game\elevated-test.ps1`, the release built
from `644814b`). As the user: 437 frames answered in 20 s, the record at integrity 8192, the
daemon gone with vkcube. As administrator, through one UAC prompt: the record at 12288, no daemon
started and nothing in its log, and setup's backend, as the window's buttons call it, said the
game runs elevated, in its status and at Enable NR (the window's words for it are in its
self-test). The same files and the same folder both times. For the tester it is still a question.

**And DOOM is set up through `DOOMx64vk.exe`**, with the game's API on Vulkan. The proxy acts only
in the executable setup names, and Steam's `DOOMx64.exe` starts `DOOMx64vk.exe` only for Vulkan.
Setup does not say so. Asked as well.

**The owner's call (10-09): a warning, no registry for now.** `644814b`, with the quick start
in `61e71a8`:
1. The proxy puts the game's integrity level in its launch record. The status says when the game
   runs elevated, and Enable NR warns: start it, and Steam or the launcher that starts it,
   without administrator rights; with UAC off, that NR needs UAC on.
2. Check and Install NR warn, without stopping an installation, when this window runs as
   administrator or UAC is off; and, for a Vulkan game, when the chosen executable does not load
   `vulkan-1.dll` itself, by import, delay-load or name, naming the one beside it that imports it.
   On DOOM's own files: `DOOMx64.exe` is warned about and `DOOMx64vk.exe` named; `DOOMx64vk.exe`
   is not.
3. Save report carries the warnings, the window's level and UAC, and the game's DXVK logs from
   `work/logs`, the game's folder and the folders it was started in, which would show whether
   DXVK drew the game at all. Need for Speed Heat could have run DirectX 12, or loaded System32's
   `d3d11.dll`; nothing in its report says which.

Tested: `test_vulkan_proxy.py` with the child at Medium and at Low integrity (8192, 4096), the x86
proxy through a 32-bit probe, 72 setup tests, the window's self-test (137 keys in three
languages). From the release's ZIP, through Steam's Play button: Mortal Kombat 11 740 frames and
Dead or Alive 5 850, none rejected, each record at 8192 with no warning, DXVK's logs in Save
report, the folders as before after Remove NR. Not pushed and not released: the owner's word
first.

Kept for later: the layer's manifest under `HKLM\SOFTWARE\Khronos\Vulkan\ExplicitLayers`, removed
by Remove NR, the one place an elevated process's loader still looks; with UAC off, the only way
NR reaches a game at all. An explicit layer loads only where it is asked for, but the key is
machine-wide, needs administrator rights to write, and would load a file a user can change into
elevated games: what the loader's check is there to prevent.

**Next on Linux:** the Windows tool suites at the head. `test_windows_wizard.py` changed again in
`a243ef8` and `644814b`; its `_file_version`, integrity and UAC tests skip off Windows, and
`test_vulkan_proxy.py` does entirely. Nothing Linux builds or runs changed.
