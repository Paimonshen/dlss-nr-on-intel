# Phase 58 — what a repository takes with it

2026-09-12. The question was whether this is fit to publish. The answer was "not yet,
three things", and this is the two of them that are mine. The third — whether to publish
reverse-engineering output at all — is the owner's and is not decided here.

## It had no licence

Without a `LICENSE` file a public repository is not open source: default copyright means
nobody may use it. **Apache-2.0**, because MLX-DLSS — whose numpy modules this reads at
runtime and whose weight specification it reproduces — is Apache-2.0, and because Apache
carries an explicit patent grant, which is worth having in a project that reimplements a
vendor's inference pass. `NOTICE` states what is not included and credits what is.

## Publishing a repository publishes its history

A file deleted from the tip still ships in every clone, and so does every line edited out
of a file that stayed. `src/tools/publish_check.py` reads the tracked tree on every
`make test`; `make publish-check` walks every blob in every commit. It looks for shapes
rather than names — home directories, mail addresses, mounted volumes, private keys,
tokens, vendor and model file extensions, anything over 200 KB — because the check is
published too, and one that spelled out what it was hiding would be the leak.

It found four things. Three were known:

| | |
| --- | --- |
| `notes/ptx-demangled.txt` | 259 KB of NVIDIA's demangled symbol names, verbatim |
| `notes/morning-doa5.md` | a home directory and a Steam library path, in a Russian log for the *other* tree |
| `notes/phase0-acquire.md` | a named community mirror, which tag to take, and how |

The fourth was not: **a 242 KB JPEG made in paint.net, sitting in the object database and
reachable from nothing.** A game frame staged once and never committed. Unreachable
objects are not pushed, so it was never a publication risk — but it is in this clone, and
nobody knew it was there.

## What was removed, and the line it was removed on

The raw demangled symbol list is gone; `notes/ptx-kernel-configs.md` and
`notes/MODEL-SPEC.txt` stay. **A record of what was learned from a binary is a different
thing from a transcription of the binary**, and the dump was the second. Anyone with the
DLL regenerates it in a minute from the commands still in `phase3-ptx-unlock.md`.

`phase0-acquire.md` keeps the half that is useful — how to tell whether the copy in your
hands is NVIDIA's original bytes, by version record, Authenticode chain and section
entropy — and loses the half that is a sourcing guide for a pre-release binary. Two other
notes carried the same pointer and were scrubbed with it.

`morning-doa5.md` is deleted outright: it documented a different tree, in a language the
rest of the repository is not written in, pointed at screenshots that were never
committed, and carried two absolute paths.

## What is still to decide, and it is not a code question

Both removals are only removals from the **tip**. Those bytes remain in the 120 commits
behind it, and publishing the repository publishes them. `make publish-check` says so
every time it is run. The choice is a rewrite that keeps the commit messages and changes
every hash, or a squashed snapshot that keeps neither — and it is one to make knowingly,
at publication, not as a side effect of a hygiene pass.

The third item is not a hygiene question at all. `MODEL-SPEC.txt`, `phase3-architecture.md`
and `ptx-kernel-configs.md` are the recovered architecture of somebody else's product.
That is the substance of the project and the first thing a reader will look at. The
precedent is public — OptiScaler, `DLSS-NR-on-AMD`, MLX-DLSS — and precedent is not
permission.
