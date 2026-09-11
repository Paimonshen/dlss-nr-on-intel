# Phase 55 — one switch, on a key, over a fullscreen game

2026-09-11. `nr-ctl` is a console, and a console needs the game to lose the screen. This
is the other half: `src/layer/nr-toggle`, a single flip with a desktop notification for an
answer, small enough to hang off a key. `Meta+N` turns the rendering on and off, `Meta+Shift+N`
turns the temporal path on and off so the flicker of `phase54` can be seen coming and going.

Proven end to end by firing the shortcut the way a keypress does —
`kglobalaccel.Component.invokeShortcut` — not by pressing anything.

## Why the compositor has to do it

Nothing else can. On Wayland an application cannot grab a key for itself, and reading
`/dev/input/event*` needs the `input` group, which this account is not in (`uzbekunknown
wheel`, and nothing under `/dev/input` is readable). python-evdev and tkinter are not
installed either. The compositor is the only thing that sees a key while a fullscreen game
holds focus, so the binding is a Plasma global shortcut and the script is what it launches.

## Three things about Plasma 6 that are not in any obvious place

Each of these cost a wrong turn, and each looks like success while doing nothing.

**1. Writing `kglobalshortcutsrc` binds nothing.** `kwriteconfig6` puts
`[services][nr-toggle.desktop] _launch=Meta+N` in the file, `kreadconfig6` reads it back,
System Settings shows it — and `allComponents` lists eighteen components, none of them
ours, through a service restart and a `kbuildsycoca6`. kglobalaccel creates a component
when something **registers** one, over D-Bus: `doRegister(as)` then
`setShortcut(as ai u)`. After that first registration the binding does survive a restart,
and a reboot, from the file.

**2. Registering needs the launcher to be in the service cache, not merely on disk.** A
`.desktop` component is resolved through KService. With a stale cache, `doRegister`
returns success, creates nothing, and `setShortcut` then returns an **empty list** rather
than an error — which reads exactly like "that key is already taken", and was diagnosed as
that twice before `isGlobalShortcutAvailable` said the key was free. `kbuildsycoca6`
between writing the file and registering is the whole fix.

**3. Unbinding has to happen while the service is stopped.** kglobalacceld holds the
bindings in memory and writes the config back **as it exits**, so deleting the keys and
then restarting deletes nothing: the file comes back with the shortcut in it. Stop, delete,
start. Verified by watching the file across a `systemctl --user stop`: the entry this
session had cleared over D-Bus was gone from the written-back file, and the one it had not
was still there.

A shortcut is one integer to kglobalaccel — the Qt key code with the modifier bits on
(Shift `0x02000000`, Ctrl `0x04000000`, Alt `0x08000000`, Meta `0x10000000`), so `Meta+N`
is `0x1000004E`. `nr-toggle` parses the spelling and **refuses** anything it cannot name,
because a shortcut that silently becomes a different key is worse than no shortcut.

## Two things the tool does because a key is not a terminal

**Turning it on starts the daemon.** A toggle that answers "no daemon, go and start one"
sends the user to the terminal it exists to avoid. One press, from nothing, is under a
second — the model is in page cache after the first load — and the daemon refuses to start
when one is already listening, so a second press is harmless. Turning it *off* leaves the
daemon up on purpose: the trigger is separate from the model precisely so the picture can
come and go without paying the load again. With no settings file at all it writes
`render_scale = 0.55` first, this project's measured compromise (`phase51`, `phase52`);
anything already chosen wins.

**Asking whether the daemon is alive no longer looks like a failure.** The probe is a
connect and a close, and the daemon logged every one as `frame rejected/failed`. Every
status line and every press of the toggle did it. `receive(..., probe_ok=True)` now
distinguishes "closed before a byte" from a truncated frame; a truncated frame still
reports.

## What was shared rather than copied

`nr-ctl` and `nr-toggle` have to agree on three paths and how the settings file is written.
They were in `nr-ctl` first and `nr-toggle` would have been a second copy, so they moved to
`src/layer/nr_paths.py`. `test_toggle.py` checks the two tools resolve to the same files —
and, separately, that `nr-ctl` can reach **every** knob the daemon has, which it could not:
`hold` was added to the daemon in `phase54` and never to the control tool. The test also
found `start_daemon` passing `--settings` without `--socket`, which would have started a
daemon nothing else was talking to whenever `NR_LAYER_SOCKET` was set.
